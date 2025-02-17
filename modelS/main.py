# --- Notebook Setup ---
import random
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence

# --- Model Definition (from recall_transformer_moe.py) ---
class RotaryPositionalEmbeddings(nn.Module):
    """Rotational Positional Embeddings (RoPE)."""
    def __init__(self, dim: int, max_seq_len: int = 4096, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self._rope_init(max_seq_len, base)

    def _rope_init(self, max_seq_len, base):
        theta = 1.0 / (base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        t = torch.arange(max_seq_len, device=theta.device)
        freqs = torch.outer(t, theta).repeat(1, 2)
        self.register_buffer("_cos_cached", freqs.cos().unsqueeze(0).unsqueeze(0))
        self.register_buffer("_sin_cached", freqs.sin().unsqueeze(0).unsqueeze(0))

    def _rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor, input_pos: torch.Tensor = None):
        batch_size, seq_len, num_heads, head_dim = x.shape
        if input_pos is not None:
            cos = self._cos_cached[:, :, input_pos]
            sin = self._sin_cached[:, :, input_pos]
        else:
            if seq_len > self.max_seq_len:
                self._rope_init(seq_len, self.base)
            cos = self._cos_cached[:, :, :seq_len]
            sin = self._sin_cached[:, :, :seq_len]
        cos = cos.transpose(1, 2)
        sin = sin.transpose(1, 2)
        return (x * cos) + (self._rotate_half(x) * sin)


class DenseTransformerLayer(nn.Module):
    """Dense Transformer Layer with RoPE (no MoE)"""
    def __init__(self, input_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        
        # Self-attention components
        self.q_proj = nn.Linear(input_dim, input_dim)
        self.k_proj = nn.Linear(input_dim, input_dim)
        self.v_proj = nn.Linear(input_dim, input_dim)
        self.rope = RotaryPositionalEmbeddings(self.head_dim)
        self.out_proj = nn.Linear(input_dim, input_dim)
        
        # FFN components
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Linear(4 * input_dim, input_dim)
        )
        
        # Normalization
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)

    def forward(self, x):
        # Self-attention
        b, s, d = x.shape
        q = self.q_proj(x).view(b, s, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(b, s, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(b, s, self.num_heads, self.head_dim)
        
        q = self.rope(q)
        k = self.rope(k)
        
        # Attention computation
        attn_output = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=True
        ).transpose(1, 2).reshape(b, s, d)
        
        attn_output = self.out_proj(attn_output)
        x = self.norm1(x + attn_output)
        
        # FFN
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        return x



class DYNMOELayer(nn.Module):
    """Dynamic Mixture of Experts (DYNMOE) layer."""
    def __init__(self, input_dim, expert_dim, max_experts=16):
        super().__init__()
        self.input_dim = input_dim
        self.expert_dim = expert_dim
        self.max_experts = max_experts
        self.Wg = nn.Parameter(torch.Tensor(max_experts, input_dim))
        self.G = nn.Parameter(torch.Tensor(max_experts))
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_dim),
                nn.GELU(),
                nn.Linear(expert_dim, input_dim)
            ) for _ in range(max_experts)
        ])
        self.register_buffer('active_experts', torch.zeros(max_experts, dtype=torch.bool))
        self.active_experts[0] = True
        self.num_active = 1
        self.register_buffer('RE', torch.zeros(max_experts, dtype=torch.int))
        self.register_buffer('RS', torch.zeros(input_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.Wg)
        nn.init.zeros_(self.G)
        for expert in self.experts:
            for layer in expert:
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, std=0.02)
                    nn.init.zeros_(layer.bias)

    def forward(self, x):
        batch_size, seq_len, d = x.shape
        x_flat = x.reshape(-1, d) # Flatten for batched processing
        active_indices = torch.where(self.active_experts)[0]
        Wg_active = self.Wg[active_indices]
        G_active = self.G[active_indices]
        num_active = len(active_indices)

        if num_active == 0:
            return x

        x_norm = F.normalize(x_flat, p=2, dim=-1)
        Wg_norm = F.normalize(Wg_active, p=2, dim=-1)
        similarities = torch.einsum('bd,ed->be', x_norm, Wg_norm) # Batch similarity computation
        scores = torch.sigmoid(similarities)
        gates = (scores > torch.sigmoid(G_active)).float()
        gates = gates + (scores - scores.detach()) # Add back gradient for routing

        # Batched expert computation
        active_experts_list = [self.experts[i] for i in active_indices.tolist()]

        # Prepare weights and biases for batched computation
        w1 = torch.stack([e[0].weight for e in active_experts_list], dim=0)  # (K, d_in, d_exp)
        b1 = torch.stack([e[0].bias for e in active_experts_list], dim=0)    # (K, d_exp)
        w2 = torch.stack([e[2].weight for e in active_experts_list], dim=0)  # (K, d_exp, d_in)
        b2 = torch.stack([e[2].bias for e in active_experts_list], dim=0)    # (K, d_in)

        # Batched computation across experts
        x_exp = x_flat.unsqueeze(1)  # (B*S, 1, d_in), reshape for batched matmul
        h = torch.bmm(x_exp, w1.to(x_exp.dtype).transpose(1,2)) + b1.to(x_exp.dtype).unsqueeze(0)  # (B*S, K, d_exp) - Batched linear 1
        h = F.gelu(h)
        expert_outputs = torch.bmm(h, w2.to(h.dtype).transpose(1,2)) + b2.to(h.dtype).unsqueeze(0)  # (B*S, K, d_in) - Batched linear 2

        # Weight expert outputs by gates and sum
        combined = (expert_outputs * gates.unsqueeze(-1).unsqueeze(-1)).sum(dim=1) # (B*S, d_in)

        k = gates.sum(dim=-1)
        scale = torch.where(k > 0, 1/(k + 1e-12), 1.0).unsqueeze(-1)
        output = combined * scale
        zero_mask = (k == 0)
        if torch.any(zero_mask):
            top1_gate = torch.zeros_like(gates)
            top1_gate[torch.arange(gates.size(0)), scores.argmax(dim=-1)] = 1.0
            expert_outputs_zero_mask = [] # Compute outputs only for zero-mask indices based on top-1 gate
            for i, idx in enumerate(active_indices.tolist()):
                expert_out_zm = self.experts[idx](x_flat[zero_mask]) * top1_gate[zero_mask, i].unsqueeze(-1)
                expert_outputs_zero_mask.append(expert_out_zm)
            output_zm = sum(expert_outputs_zero_mask)
            output[zero_mask] = output_zm # Assign outputs for zero-mask indices

        if self.training:
            self.RE[active_indices] += gates.sum(dim=0).detach().to(torch.int)
            if zero_mask.any():
                self.RS += x_flat[zero_mask].sum(dim=0).detach()

        return output.reshape(batch_size, seq_len, d)

    def auxiliary_loss(self):
        active_Wg = self.Wg[self.active_experts]
        if len(active_Wg) == 0:
            return torch.tensor(0.0, device=self.Wg.device)
        active_Wg_norm = F.normalize(active_Wg, p=2, dim=1)
        identity = torch.eye(len(active_Wg), device=active_Wg.device)
        div_loss = torch.norm(active_Wg_norm @ active_Wg_norm.T - identity) ** 2
        simp_loss = torch.sum(active_Wg ** 2)
        return div_loss + simp_loss

    def adapt_experts(self, step, adapt_interval=100):
        if not self.training or step % adapt_interval != 0:
            return
        active_indices = torch.where(self.active_experts)[0]
        if self.num_active < self.max_experts and self.RS.norm() > 1e-6:
            inact_indices = torch.where(~self.active_experts)[0]
            new_idx = inact_indices[0]
            ws_new = self.RS / self.RS.norm()
            self.Wg.data[new_idx] = ws_new
            self.G.data[new_idx] = 0.0
            self.active_experts[new_idx] = True
            self.num_active += 1
            self.RS.zero_()
        usage = self.RE[active_indices]
        dead_experts = (usage == 0)
        if dead_experts.any():
            dead_indices = active_indices[dead_experts]
            self.active_experts[dead_indices] = False
            self.num_active -= len(dead_indices)
        self.RE.zero_()


class InternalLayer(nn.Module):
    def __init__(self, input_dim, expert_dim, max_experts, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        
        self.q_proj = nn.Linear(input_dim, input_dim)
        self.k_proj = nn.Linear(input_dim, input_dim)
        self.v_proj = nn.Linear(input_dim, input_dim)
        
        self.rope = RotaryPositionalEmbeddings(self.head_dim)
        self.moe = DYNMOELayer(input_dim, expert_dim, max_experts)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.out_proj = nn.Linear(input_dim, input_dim)
        
    def forward(self, x):
        b, s, d = x.shape
        
        q = self.q_proj(x).view(b, s, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(b, s, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(b, s, self.num_heads, self.head_dim)
        
        q = self.rope(q)
        k = self.rope(k)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        attn_output = F.scaled_dot_product_attention(
            q, k, v, is_causal=True
        )
        attn_output = attn_output.transpose(1, 2).reshape(b, s, d)
        attn_output = self.out_proj(attn_output)
        attn_out = self.norm1(x + attn_output)
        
        moe_out = self.moe(attn_out)
        output = self.norm2(attn_out + moe_out)
        return output



class RecurrentBlockWithMoE(nn.Module):
    """Recurrent Block with Mixture of Experts and RoPE."""
    def __init__(self, input_dim, expert_dim, max_experts, num_heads, num_layers=3):
        super().__init__()
        self.recall_proj = nn.Linear(2 * input_dim, input_dim)
        self.layers = nn.ModuleList([
            InternalLayer(input_dim, expert_dim, max_experts, num_heads)
            for _ in range(num_layers)
        ])
        
    def forward(self, phi_prev, x_emb):
        combined = torch.cat([phi_prev, x_emb], dim=-1)
        combined = self.recall_proj(combined)
        for layer in self.layers:
            combined = layer(combined)
        return combined

class RecallTransformerWithMoE(nn.Module):
    """Modified with configurable pre/post dense layers"""
    def __init__(self, num_layers, input_dim, vocab_size, expert_dim=64,
                 num_heads=4, max_experts=4, max_iters=4,
                 num_pre_layers=2, num_post_layers=2):  # New hyperparams
        super().__init__()
        self.encoder = nn.Embedding(vocab_size, input_dim)

        # Pre-recurrent dense layers
        self.pre_layers = nn.ModuleList([
            DenseTransformerLayer(input_dim, num_heads)
            for _ in range(num_pre_layers)
        ])

        # Recurrent block
        self.recurrent_block = RecurrentBlockWithMoE(
            input_dim, expert_dim, max_experts, num_heads, num_layers
        )
        self.max_iters = max_iters

        # Post-recurrent dense layers
        self.post_layers = nn.ModuleList([
            DenseTransformerLayer(input_dim, num_heads)
            for _ in range(num_post_layers)
        ])

        # Output heads
        self.output_head = nn.Linear(input_dim, vocab_size)
        self.value_head = nn.Linear(input_dim, 1)

    def forward(self, input_ids, num_iters=None):
        x_emb = self.encoder(input_ids)

        # Process through pre-layers
        for layer in self.pre_layers:
            x_emb = layer(x_emb)

        phi = x_emb
        num_iters = num_iters if num_iters is not None else self.max_iters

        # Recurrent processing
        for _ in range(num_iters):
            phi = self.recurrent_block(phi, x_emb)

        # Process through post-layers
        for layer in self.post_layers:
            phi = layer(phi)

        logits = self.output_head(phi)
        value = self.value_head(phi).squeeze(-1)
        return logits, value

    def generate(self, input_ids, max_length=20, temperature=1.0, num_iters=None):
        """Autoregressively generates tokens."""
        generated_tokens = input_ids
        for _ in range(max_length):
            logits, _= self.forward(generated_tokens, num_iters=num_iters)
            next_token_logits = logits[:, -1, :] / temperature # Use logits of the last token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens = torch.cat([generated_tokens, next_token], dim=-1)
            if next_token.item() == self.encoder.num_embeddings - 1: # Assuming last token ID is EOS
                break
        return generated_tokens

def generate_expression(num_ops):
    if num_ops == 0:
        return str(random.randint(0, 9))
    left_ops = random.randint(0, num_ops - 1)
    right_ops = num_ops - 1 - left_ops
    left_expr = generate_expression(left_ops)
    right_expr = generate_expression(right_ops)
    op = random.choice(['+', '-', '*', '/'])
    expr = "(" + left_expr + op + right_expr + ")"
    return expr

def generate_expressions(num_expressions, num_ops):
    expr_list = []
    while len(expr_list) < num_expressions:
        expr = generate_expression(num_ops)
        try:
            result = eval(expr)
        except ZeroDivisionError:
            continue
        except Exception as e:
            continue
        expr_list.append((expr, result))
    return expr_list

# --- Tokenizer (in notebook) ---
class ExpressionTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.token_to_id = {token: id for id, token in enumerate(vocab)}
        self.id_to_token = {id: token for id, token in enumerate(vocab)}

    def encode(self, text):
        return [self.token_to_id[char] for char in text]

    def decode(self, token_ids):
        return "".join([self.id_to_token[id] for id in token_ids])

# --- Reward Calculation (in notebook) ---
def calculate_reward(current_seq_tokens, last_action_token, target_result, tokenizer, vocab):
    generated_expression_tokens = current_seq_tokens[0, len(tokenizer.encode("=")) : -1]
    generated_expression_str = tokenizer.decode(generated_expression_tokens.tolist())

    prefix_expression_tokens = current_seq_tokens[0, :len(tokenizer.encode("="))]
    prefix_expression_str = tokenizer.decode(prefix_expression_tokens.tolist())
    full_expression_str = prefix_expression_str + generated_expression_str

    try:
        evaluated_result = eval(full_expression_str[:-1])
        if abs(evaluated_result - target_result) < 1e-6:
            return 1.0
        else:
            return 0.0
    except:
        return 0.0

# --- PPO Iteration (in notebook) ---
def ppo_iteration(model, optimizer, scaler, ppo_epochs, batch_size, dataset, tokenizer, vocab, gamma=0.99, gae_lambda=0.95, clip_param=0.2, value_loss_coef=0.5, entropy_coef=0.01, moe_aux_loss_coef=0.01):
    trajectories = []
    total_reward = 0

    for problem, target_result in dataset:
        input_seq_str = problem + "="
        input_seq_tokens = tokenizer.encode(input_seq_str)
        current_seq_tokens = torch.tensor([input_seq_tokens], dtype=torch.long).to(next(model.parameters()).device)

        max_len = 20
        eos_token_id = vocab.index('EoS')

        actions = []
        log_probs = []
        values = []
        rewards = []
        masks = []
        states = []

        for step in range(max_len):
            states.append(current_seq_tokens.clone())

            with torch.no_grad():
                with torch.cuda.amp.autocast(): # FP16 context for forward pass
                    logits, value = model(current_seq_tokens)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                dist = Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)

            actions.append(action)
            log_probs.append(log_prob)
            values.append(value[:, -1])

            next_token_id = action.item()
            if next_token_id == eos_token_id:
                masks.append(0)
                current_reward = calculate_reward(current_seq_tokens, action, target_result, tokenizer, vocab)
                rewards.append(current_reward)
                total_reward += current_reward
                break
            else:
                masks.append(1)
                rewards.append(0.0)
                current_seq_tokens = torch.cat([current_seq_tokens, action.unsqueeze(0)], dim=1)
        else:
            masks[-1] = 0
            rewards[-1] = calculate_reward(current_seq_tokens, None, target_result, tokenizer, vocab)
            total_reward += rewards[-1]

        values_tensor = torch.stack(values).squeeze(-1).to(device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
        masks_tensor = torch.tensor(masks, dtype=torch.float32, device=device)

        next_values = torch.zeros_like(values_tensor)
        next_values[:-1] = values_tensor[1:]
        delta_t = rewards_tensor + gamma * masks_tensor * next_values - values_tensor

        advantage_buffer = 0
        advantages = []
        for delta, mask in zip(reversed(delta_t.tolist()), reversed(masks)):
            advantage_buffer = delta + gamma * gae_lambda * mask * advantage_buffer
            advantages.insert(0, advantage_buffer)
        advantages = torch.tensor(advantages, dtype=torch.float32, device=device).unsqueeze(1)

        trajectory = {
            'states': states,
            'input_sequences': input_seq_tokens,
            'actions': torch.stack(actions).to(device),
            'log_probs': torch.stack(log_probs).to(device),
            'values': values_tensor.unsqueeze(1),
            'advantages': advantages,
            'returns': (advantages + values_tensor.unsqueeze(1)).detach()
        }
        trajectories.append(trajectory)

    avg_reward = total_reward / len(dataset)
    print(f"Average reward for this iteration: {avg_reward:.4f}")

    for _ in range(ppo_epochs):
        for trajectory in trajectories:
            model.train()
            optimizer.zero_grad()

            states_list = [s.squeeze(0) for s in trajectory['states']]
            states_batch = pad_sequence(states_list, batch_first=True, padding_value=tokenizer.token_to_id['EoS']).to(device)

            # FP16 context for model forward and loss calculation
            with torch.cuda.amp.autocast():
                logits, values_new = model(states_batch)
                last_indices = torch.tensor([len(s) - 1 for s in states_list], device=device)
                logits_last = logits[torch.arange(len(last_indices)), last_indices, :]
                values_last = values_new[torch.arange(len(last_indices)), last_indices].unsqueeze(1)

                dist = Categorical(F.softmax(logits_last, dim=-1, dtype=torch.float32)) # Ensure softmax in FP32
                new_log_probs = dist.log_prob(trajectory['actions'])        

                ratios = torch.exp(new_log_probs - trajectory['log_probs'])
                surr1 = ratios * trajectory['advantages']
                surr2 = torch.clamp(ratios, 1-clip_param, 1+clip_param) * trajectory['advantages']
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values_last, trajectory['returns'])
                entropy = dist.entropy().mean()
                moe_loss = sum(module.auxiliary_loss() for module in model.modules() if isinstance(module, DYNMOELayer))
                total_loss = (policy_loss
                            + value_loss_coef * value_loss
                            - entropy_coef * entropy
                            + moe_aux_loss_coef * moe_loss)

            # Scale loss and perform optimization step
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

    return avg_reward



def evaluate_model(model, difficulty_level, num_samples=100):
    model.eval()
    total_reward = 0.0
    eval_data = generate_expressions(num_samples, difficulty_level)
    
    for problem, target in eval_data:
        input_seq = problem + "="
        input_tokens = tokenizer.encode(input_seq)
        input_tensor = torch.tensor([input_tokens], dtype=torch.long).to(device)
        
        with torch.no_grad():
            generated = model.generate(input_tensor, max_length=20)
            
        generated_tokens = generated[0].tolist()
        generated_expr = tokenizer.decode(generated_tokens).split('EoS')[0]
        full_expr = problem + "=" + generated_expr
        
        try:
            result = eval(full_expr)
            reward = 1.0 if abs(result - target) < 1e-6 else 0.0
        except:
            reward = 0.0
            
        total_reward += reward
    
    avg_reward = total_reward / num_samples
    model.train()
    return avg_reward


# --- Main Training and Inference (in notebook) ---
# Hyperparameters
vocab_chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '*', '/', '(', ')', '=', 'EoS']
vocab = vocab_chars
vocab_size = len(vocab)
tokenizer = ExpressionTokenizer(vocab)

input_dim = 512
expert_dim = 192
max_experts = 6
max_iters = 6
num_layers = 4
num_heads = 8
learning_rate = 1e-4
ppo_epochs = 4
batch_size_ppo = 32
initial_difficulty = 1
difficulty_increment_interval = 50
difficulty_increment_amount = 0.5
num_pre_layers = 3
num_post_layers = 2
initial_difficulty = 1  # Start with 1 operation
difficulty_increment_amount = 0.5  # Increment by 1 operation
max_difficulty = 5
evaluation_interval = 5  # Evaluate every 10 iterations
success_threshold = 0.95  # 95% accuracy to advance

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = RecallTransformerWithMoE(
    num_layers=3,
    input_dim=input_dim,
    vocab_size=vocab_size,
    expert_dim=expert_dim,
    num_heads=num_heads,
    max_experts=max_experts,
    max_iters=max_iters,
    num_pre_layers=num_pre_layers,    # Pass new params
    num_post_layers=num_post_layers
).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scaler = torch.cuda.amp.GradScaler() # Initialize GradScaler here


num_iterations = 500
current_difficulty = initial_difficulty

for iteration in range(num_iterations):
    # Generate training data at current difficulty
    expressions_data = generate_expressions(batch_size_ppo, current_difficulty)
    dataset_rl = expressions_data
    
    # Run PPO iteration
    avg_reward = ppo_iteration(
        model, optimizer, scaler, ppo_epochs, batch_size_ppo, # Pass scaler here
        dataset_rl, tokenizer, vocab
    )
    
    # Adapt MoE experts
    for name, module in model.named_modules():
        if isinstance(module, DYNMOELayer):
            module.adapt_experts(iteration, adapt_interval=10)
    
    # Evaluate and adjust difficulty
    if (iteration + 1) % evaluation_interval == 0:
        eval_reward = evaluate_model(model, current_difficulty)
        print(f"Iteration {iteration+1} - Current Difficulty: {current_difficulty}")
        print(f"Evaluation Reward: {eval_reward:.4f}")
        
        if eval_reward >= success_threshold:
            if current_difficulty < max_difficulty:
                current_difficulty += difficulty_increment_amount
                print(f"Mastered! New Difficulty: {current_difficulty}")
            else:
                print("Reached maximum difficulty")
        else:
            print(f"Not mastered - Continuing at difficulty {current_difficulty}")

print("Training complete!")

# --- Inference Example (in notebook) ---
model.eval()
test_expression = "(8+2)*3="
test_input_ids = torch.tensor([tokenizer.encode(test_expression)], dtype=torch.long).to(device)

with torch.no_grad():
    generated_sequence = model.generate(test_input_ids, max_length=10, num_iters=max_iters)
    generated_expression_result = tokenizer.decode(generated_sequence[0].tolist())
    print(f"Input Expression: {test_expression}")
    print(f"Generated Result: {generated_expression_result}")
    try:
        evaluated_result = eval(test_expression[:-1] + generated_expression_result.replace("EoS", ""))
        print(f"Evaluated Result: {evaluated_result}")
    except Exception as e:
        print(f"Evaluation Error: {e}")