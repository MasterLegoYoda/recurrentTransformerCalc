import torch
import torch.nn as nn
import torch.nn.functional as F

class DYNMOELayer(nn.Module):
    def __init__(self, input_dim, expert_dim, max_experts=16):
        """
        Dynamic Mixture of Experts (DYNMOE) layer
        
        Args:
            input_dim: Dimension of input token embeddings
            expert_dim: Hidden dimension of expert networks
            max_experts: Maximum number of allowed experts
        """
        super().__init__()
        self.input_dim = input_dim
        self.expert_dim = expert_dim
        self.max_experts = max_experts
        
        # Expert selection parameters
        self.Wg = nn.Parameter(torch.Tensor(max_experts, input_dim))  # Expert representations
        self.G = nn.Parameter(torch.Tensor(max_experts))              # Thresholds
        
        # Expert networks (FFNs)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_dim),
                nn.GELU(),
                nn.Linear(expert_dim, input_dim)
            ) for _ in range(max_experts)
        ])
        
        # Active expert mask (starts with first expert active)
        self.register_buffer('active_experts', torch.zeros(max_experts, dtype=torch.bool))
        self.active_experts[0] = True
        self.num_active = 1
        
        # Routing statistics
        self.register_buffer('RE', torch.zeros(max_experts, dtype=torch.int))  # Activation counts
        self.register_buffer('RS', torch.zeros(input_dim))                     # Accumulated embeds
        
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights correctly"""
        nn.init.orthogonal_(self.Wg)
        nn.init.zeros_(self.G)
        for expert in self.experts:
            for layer in expert:
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, std=0.02)
                    nn.init.zeros_(layer.bias)

    def forward(self, x):
        """
        Forward pass with dynamic expert selection
        
        Args:
            x: (batch_size, seq_len, input_dim)
        Returns:
            out: (batch_size, seq_len, input_dim)
        """
        batch_size, seq_len, d = x.shape
        x_flat = x.reshape(-1, d)
        
        # 1. TOP-ANY GATING MECHANISM
        # Get active expert parameters
        active_indices = torch.where(self.active_experts)[0]
        Wg_active = self.Wg[active_indices]
        G_active = self.G[active_indices]
        num_active = len(active_indices)
        if num_active == 0:  # All experts disabled -> return original
            return x
        
        # Compute cosine similarities
        x_norm = F.normalize(x_flat, p=2, dim=-1)
        Wg_norm = F.normalize(Wg_active, p=2, dim=-1)
        similarities = torch.einsum('bd,ed->be', x_norm, Wg_norm)  # (B*S, num_active)
        
        # Gating logic with straight-through estimator
        scores = torch.sigmoid(similarities)
        gates = (scores > torch.sigmoid(G_active)).float()  # Binary gates
        gates = gates + (scores - scores.detach())  # STE for backprop
        
        # 2. EXPERT PROCESSING
        outputs = []
        for i, idx in enumerate(active_indices.tolist()):
            expert_out = self.experts[idx](x_flat) * gates[:, i].unsqueeze(-1)
            outputs.append(expert_out)
        
        # Combine expert outputs
        combined = sum(outputs)  # (B*S, input_dim)
        k = gates.sum(dim=-1)  # Number of active experts per token
        scale = torch.where(k > 0, 1/(k + 1e-12), 1.0).unsqueeze(-1)
        output = combined * scale
        
        # 3. HANDLE ZERO-ACTIVATION TOKENS (test-time modification during training)
        zero_mask = (k == 0)
        if torch.any(zero_mask):
            # Get top-1 expert for failed tokens
            top1_gate = torch.zeros_like(gates)
            top1_gate[torch.arange(gates.size(0)), scores.argmax(dim=-1)] = 1.0
            outputs = [self.experts[idx](x_flat) * top1_gate[:, i].unsqueeze(-1) 
                      for i, idx in enumerate(active_indices)]
            output[zero_mask] = sum(outputs)[zero_mask]

        # 4. UPDATE ROUTING STATISTICS
        if self.training:
            # Update activation counts
            self.RE[active_indices] += gates.sum(dim=0).detach().to(torch.int)
            
            # Accumulate failed embeddings
            if zero_mask.any():
                self.RS += x_flat[zero_mask].sum(dim=0).detach()
        
        return output.reshape(batch_size, seq_len, d)

    def auxiliary_loss(self):
        """Sparse and Simple Gating Loss"""
        active_Wg = self.Wg[self.active_experts]
        if len(active_Wg) == 0:
            return torch.tensor(0.0, device=self.Wg.device)
        
        # Diversity loss (encourage orthogonality)
        active_Wg_norm = F.normalize(active_Wg, p=2, dim=1)
        identity = torch.eye(len(active_Wg), device=active_Wg.device)
        div_loss = torch.norm(active_Wg_norm @ active_Wg_norm.T - identity) ** 2
        
        # Simplicity loss (control magnitudes)
        simp_loss = torch.sum(active_Wg ** 2)
        
        return div_loss + simp_loss

    def adapt_experts(self, step, adapt_interval=100):
        """Dynamic expert adjustment (call periodically during training)"""
        if not self.training or step % adapt_interval != 0:
            return
        
        # Current active experts
        active_indices = torch.where(self.active_experts)[0]
        
        # 1. ADD NEW EXPERT IF NEEDED
        if self.num_active < self.max_experts and self.RS.norm() > 1e-6:
            inact_indices = torch.where(~self.active_experts)[0]
            new_idx = inact_indices[0]
            
            # Initialize to accumulated embeddings
            ws_new = self.RS / self.RS.norm()
            self.Wg.data[new_idx] = ws_new
            self.G.data[new_idx] = 0.0
            self.active_experts[new_idx] = True
            self.num_active += 1
            
            # Reset RS
            self.RS.zero_()
            
        # 2. REMOVE UNDERUSED EXPERTS
        usage = self.RE[active_indices]
        dead_experts = (usage == 0)
        if dead_experts.any():
            dead_indices = active_indices[dead_experts]
            self.active_experts[dead_indices] = False
            self.num_active -= len(dead_indices)
        
        # Reset counters
        self.RE.zero_()


class TransformerWithDYNMOE(nn.Module):
    """Example Transformer using DYNMOE layers"""
    def __init__(self, num_layers, input_dim, expert_dim=256, num_heads=8, max_experts=16):
        super().__init__()
        
        self.encoder = nn.Embedding(10000, input_dim)
        
        # Standard Transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=num_heads,
                dim_feedforward=expert_dim*2,
                activation='gelu',
                batch_first=True
            ) for _ in range(num_layers//2)
        ])
        
        # DYNMOE layers (replace FFNs)
        self.moe_layers = nn.ModuleList([
            DYNMOELayer(input_dim, expert_dim, max_experts)
            for _ in range(num_layers//2)
        ])
        
    def forward(self, x):
        x = self.encoder(x)
        layer_idx = 0
        for std_layer in self.layers:
            x = std_layer(x)
            
            # Add DYNMOE layer 
            moe_out = self.moe_layers[layer_idx](x)
            x = x + moe_out  # Simple residual connection
            layer_idx += 1
        return x


# Example Training Loop
model = TransformerWithDYNMOE(num_layers=6, input_dim=512)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

for batch_idx, (inputs, targets) in enumerate(train_loader):
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(inputs)
    
    # Calculate main loss (example: cross-entropy for sequence prediction)
    # Note: You may need to add a task-specific head to the model
    logits = outputs @ model.encoder.weight.T  # Simple weight tying example
    loss_main = F.cross_entropy(logits.view(-1, logits.size(-1)), 
                               targets.view(-1), 
                               ignore_index=-100)
    
    # Calculate auxiliary losses from DYNMOE layers
    aux_loss = sum([moe.auxiliary_loss() for moe in model.moe_layers])
    
    # Total loss (adjust auxiliary loss scaling factor as needed)
    total_loss = loss_main + 0.1 * aux_loss
    
    # Backward pass and optimize
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
    optimizer.step()
    
    # Dynamic expert adaptation (perform after optimization step)
    for moe in model.moe_layers:
        moe.adapt_experts(batch_idx, adapt_interval=100)  # Check for adaptation
    
    # Log training statistics
    if batch_idx % 100 == 0:
        active_experts = sum([moe.num_active for moe in model.moe_layers])
        avg_usage = sum([moe.RE.float().mean().item() for moe in model.moe_layers])/len(model.moe_layers)
        
        print(f"Batch {batch_idx:04d} | "
              f"Loss: {total_loss.item():.4f} | "
              f"Active Experts: {active_experts} | "
              f"Avg Usage: {avg_usage:.1f}")

# Important Implementation Notes:
# 1. Initialization: The first expert starts active; use warmup steps to reach steady state
# 2. Threshold Warming: Gradually decrease learning rate for G thresholds after initialization
# 3. Batch Size: Use larger batch sizes for better expert utilization (>=64 recommended)
# 4. Expert Scaling: Adjust max_experts parameter based on available compute resources
# 5. Scheduling: Consider learning rate warmup when using adaptive expert counts

# Sample Model Usage:
# transformer = TransformerWithDYNMOE(
#     num_layers=6, 
#     input_dim=512,
#     expert_dim=1024,
#     max_experts=8
# )
# optimizer = torch.optim.AdamW(transformer.parameters(), lr=5e-4, weight_decay=0.1)

# Key Hyperparameters to Tune:
# - Auxiliary loss scaling factor (found 0.1 to be reasonable in tests)
# - Adaptation interval (start with 100 steps)
# - Expert network scale (hidden_dim/expert_dim ratio)
# - Maximum number of experts (balance between capacity and compute)