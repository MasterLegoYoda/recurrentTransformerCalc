import torch
import torch.nn as nn
import torch.nn.functional as F

def rotate_half(x):
    x = x.view(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return x.view(*x.shape[:-2], -1)

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", self.inv_freq)

    def _get_rotary_embedding(self, seq_len, device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)

    def forward(self, q, k):
        batch, seq_len, _, _ = q.shape
        sin = self._get_rotary_embedding(seq_len, q.device).sin()[None, None, ...]
        cos = self._get_rotary_embedding(seq_len, q.device).cos()[None, None, ...]
        
        q_rot = (q * cos) + (rotate_half(q) * sin)
        k_rot = (k * cos) + (rotate_half(k) * sin)
        return q_rot, k_rot

class MultiheadAttentionWithRoPE(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.rotary = RotaryPositionalEmbedding(self.head_dim)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        q_rot, k_rot = self.rotary(q, k)
        
        attn = torch.matmul(q_rot, k_rot.transpose(-1, -2)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        
        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.out_proj(output)
        return output
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, is_recurrent=False):
        super().__init__()
        self.attention = MultiheadAttentionWithRoPE(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention with Pre-LN
        x_norm = self.norm1(x)
        attn_out = self.attention(x_norm)
        x = x + self.dropout(attn_out)
        
        # FFN with Pre-LN
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + self.dropout(ffn_out)
        return x

class RecurrentTransformer(nn.Module):
    def __init__(self, L_recurrent, D_recurrent, D_initial, num_heads, dropout=0.1):
        super().__init__()
        self.fusion = nn.Linear(D_recurrent + D_initial, D_recurrent)
        self.layers = nn.ModuleList(
            [TransformerBlock(D_recurrent, num_heads, dropout, is_recurrent=True) 
             for _ in range(L_recurrent)]
        )

    def forward(self, iteration_input, context_input):
        fused = self.fusion(torch.cat([iteration_input, context_input], dim=-1))
        for layer in self.layers:
            fused = layer(fused)
        return fused

class HaltingUnit(nn.Module):
    def __init__(self, H_layers, D_halt, D_recurrent, num_heads):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerBlock(D_halt, num_heads) 
             for _ in range(H_layers)]
        )
        self.proj_in = nn.Linear(D_recurrent, D_halt)
        self.proj_out = nn.Linear(D_halt, 1)

    def forward(self, x):
        x = self.proj_in(x)
        x = torch.mean(x, dim=1)  # Global pooling
        for layer in self.layers:
            x = layer(x)
        return torch.sigmoid(self.proj_out(x)).squeeze(-1)

class HybridTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.initial_layers = nn.Sequential(*[
            TransformerBlock(config.D_initial, config.num_heads) 
            for _ in range(config.N_initial)
        ])
        
        self.recurrent_block = RecurrentTransformer(
            config.L_recurrent, config.D_recurrent, 
            config.D_initial, config.num_heads
        )
        
        self.halting = HaltingUnit(
            config.H_layers, config.D_halt,
            config.D_recurrent, config.num_heads
        )
        
        self.final_layers = nn.Sequential(*[
            TransformerBlock(config.D_final, config.num_heads)
            for _ in range(config.N_final)
        ])
        
        self.proj_final = nn.Linear(config.D_recurrent, config.D_final)
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize dimensions from config
        self.D_initial = config.D_initial
        self.D_recurrent = config.D_recurrent
        self.D_final = config.D_final

    def forward(self, x, max_iters=None, n_k=None):
        # Initial feature extraction
        context = self.initial_layers(x)
        
        # Recurrent processing
        B, T, _ = context.shape
        h_state = torch.zeros(B, T, self.D_recurrent, device=x.device)
        h_probs = []
        halts = torch.zeros(B, device=x.device)
        
        iters = 0
        max_iters = max_iters if max_iters else self.config.m
        
        while (iters < max_iters) and (halts.mean() < 0.99):
            # Reset state if using truncated BPTT
            if n_k and (iters - n_k[0] == n_k[1]):
                h_state = h_state.detach()

            with torch.set_grad_enabled(not self.training or (iters >= n_k[0] if n_k else True)):
                h_state = self.recurrent_block(
                    self.dropout(h_state), 
                    context.detach() if iters > 0 else context
                )
            
            halt_prob = self.halting(h_state)
            h_probs.append(halt_prob)
            halts = halts + (1 - halts) * halt_prob
            iters += 1

        # Final projection
        out = self.proj_final(h_state)
        return {
            "output": self.final_layers(out),
            "halting_probs": torch.stack(h_probs),
            "num_iters": iters
        }
    

def training_step(batch, config, α=0.3):
    model = HybridTransformer(config)
    X, y = batch
    
    # First pass (n + k steps)
    n = torch.randint(0, config.m-1, ())
    k = torch.randint(1, config.m-n, ())
    output_trunc = model(X, n_k=(n, k))
    L_prog = loss_fn(output_trunc["output"], y)
    
    # Second pass (full sequence)
    with torch.inference_mode():
        output_full = model(X, max_iters=config.m)
    L_max = loss_fn(output_full["output"], y)
    
    # Weighted loss
    total_loss = (1 - α)*L_max + α*L_prog
    
    # Backpropagate through the truncated pathway
    total_loss.backward()
    return total_loss