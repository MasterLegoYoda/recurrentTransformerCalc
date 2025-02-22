# ---------- Notebook Setup ----------
import random
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence

# ---------- Model Definition (Modified without MoE) ----------

class RotaryPositionalEmbeddings(nn.Module):
    """Rotational Positional Embeddings (RoPE)."""
    def __init__(self, dim: int, max_seq_len: int = 4096, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self._rope_init(max_seq_len, base)

    def _rope_init(self, max_seq_len, base):
        theta = 1.0 / (base ** (2 * (torch.arange(0, self.dim, 2).float() / self.dim)))
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
                raise ValueError(f"Sequence length {seq_len} exceeds precomputed max {self.max_seq_len}") # Raise error instead of resizing
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


class InternalLayerDense(nn.Module):
    def __init__(self, input_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        # Self-attention components
        self.q_proj = nn.Linear(input_dim, input_dim)
        self.k_proj = nn.Linear(input_dim, input_dim)
        self.v_proj = nn.Linear(input_dim, input_dim)
        self.rope = RotaryPositionalEmbeddings(self.head_dim)
        # Standard dense FFN (replacing MoE)
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Linear(4 * input_dim, input_dim)
        )
        # Normalization
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.out_proj = nn.Linear(input_dim, input_dim)

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
        attn_out = self.norm1(x + attn_output)
        # FFN instead of MoE
        ffn_output = self.ffn(attn_out)
        output = self.norm2(attn_out + ffn_output)
        return output


class RecurrentBlockWithDense(nn.Module):
    """Recurrent Block with configurable hidden dimension"""
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers=3):
        super().__init__()
        # Project combined input to hidden dimension
        self.recall_proj = nn.Linear(2 * input_dim, hidden_dim)
        # Recurrent layers in hidden dimension
        self.layers = nn.ModuleList([
            InternalLayerDense(hidden_dim, num_heads)
            for _ in range(num_layers)
        ])
        # Project back to original input dimension
        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, phi_prev, x_emb):
        # Combine inputs: [batch, seq_len, 2*input_dim]
        combined = torch.cat([phi_prev, x_emb], dim=-1)
        # Project to hidden dimension
        hidden = self.recall_proj(combined)
        # Process through hidden layers
        for layer in self.layers:
            hidden = layer(hidden)
        # Project back to input dimension
        return self.output_proj(hidden)


class RecallTransformerWithDense(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, vocab_size, num_heads=8,
                 max_iters=4, num_pre_layers=2, num_post_layers=2):
        super().__init__()
        self.encoder = nn.Embedding(vocab_size, input_dim)
        # Pre/Post layers remain in input dimension
        self.pre_layers = nn.ModuleList([
            DenseTransformerLayer(input_dim, num_heads)
            for _ in range(num_pre_layers)
        ])
        # Recurrent block uses separate hidden dimension
        self.recurrent_block = RecurrentBlockWithDense(
            input_dim, hidden_dim, num_heads, num_layers
        )
        self.post_layers = nn.ModuleList([
            DenseTransformerLayer(input_dim, num_heads)
            for _ in range(num_post_layers)
        ])
        # Heads remain in input dimension
        self.output_head = nn.Linear(input_dim, vocab_size)
        self.value_head = nn.Linear(input_dim, 1)
        self.max_iters = max_iters

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
            logits, _ = self.forward(generated_tokens, num_iters=num_iters)
            next_token_logits = logits[:, -1, :] / temperature  # Use logits of the last token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens = torch.cat([generated_tokens, next_token], dim=-1)
            # Assume the last token in vocab is the EOS token.
            if next_token.item() == self.encoder.num_embeddings - 1:
                break
        return generated_tokens


# ---------- Expression Generation Utilities ----------

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
        # Save each example as a pair: (problem string, result)
        expr_list.append((expr, result))
    return expr_list

# ---------- Tokenizer (adjusted for supervised pretraining) ----------

class ExpressionTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.token_to_id = {token: i for i, token in enumerate(vocab)}
        self.id_to_token = {i: token for i, token in enumerate(vocab)}

    def encode(self, text):
        # Convert each character to its corresponding id.
        # Note: since vocab only contains digits and EoS, all non-digit characters are ignored.
        return [self.token_to_id[char] for char in text if char in self.token_to_id]

    def decode(self, token_ids):
        # Ignore tokens not in vocabulary.
        return "".join([self.id_to_token[i] for i in token_ids if i in self.id_to_token])

# ---------- Reward Calculation (for RL phase) ----------

def calculate_reward(current_seq_tokens, last_action_token, target_result, tokenizer, vocab):
    # In this example the generated answer should be a sequence of digits terminated by EoS.
    generated_expression_tokens = current_seq_tokens[0, :]  # all tokens
    generated_expression_str = tokenizer.decode(generated_expression_tokens.tolist())
    generated_number_str = generated_expression_str.split('EoS')[0]
    if not generated_number_str.isdigit():
        return 0.0
    try:
        generated_number = int(generated_number_str)
    except:
        return 0.0
    return 1.0 if generated_number == target_result else 0.0

# ---------- PPO Iteration (RL phase) ----------

def ppo_iteration_batch(model, optimizer, ppo_epochs, batch_size, dataset, tokenizer, vocab,
                        gamma=0.99, gae_lambda=0.95, clip_param=0.2, value_loss_coef=0.5,
                        entropy_coef=0.01, max_len=20, temperature=1.0):
    device = next(model.parameters()).device
    # Prepare input sequences: each problem is the arithmetic expression with an "=" appended.
    input_seqs = []
    input_lens = []
    target_results = []
    for problem, target in dataset:
        input_seq_str = problem + "="
        input_seq_tokens = tokenizer.encode(input_seq_str)
        input_seqs.append(torch.tensor(input_seq_tokens, dtype=torch.long))
        input_lens.append(len(input_seq_tokens))
        target_results.append(target)
    # Pad the input batch
    batch_initial = pad_sequence(input_seqs, batch_first=True, padding_value=tokenizer.token_to_id['EoS']).to(device)
    batch_size = batch_initial.size(0)
    eos_token_id = vocab.index('EoS')
    # Rollout trajectories
    current_seq = batch_initial.clone()
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    actions_list = [[] for _ in range(batch_size)]
    log_probs_list = [[] for _ in range(batch_size)]
    values_list = [[] for _ in range(batch_size)]
    masks_list = [[] for _ in range(batch_size)]
    states_list = [[] for _ in range(batch_size)]
    rewards = torch.zeros(batch_size, device=device)

    for t in range(max_len):
        for i in range(batch_size):
            states_list[i].append(current_seq[i:i+1].clone().detach())
        logits, value = model(current_seq)
        last_idxs = torch.tensor([(s != tokenizer.token_to_id['EoS']).sum().item()-1 for s in current_seq],
                                  device=device)
        logits_last = logits[torch.arange(batch_size), last_idxs, :] / temperature
        probs = F.softmax(logits_last, dim=-1)
        dist = torch.distributions.Categorical(probs)
        sampled = dist.sample()
        log_prob = dist.log_prob(sampled)
        for i in range(batch_size):
            if not finished[i]:
                actions_list[i].append(sampled[i].detach())
                log_probs_list[i].append(log_prob[i].detach())
                values_list[i].append(value[i, last_idxs[i]].detach())
            masks_list[i].append(1.0 if not finished[i] else 0.0)
        new_finished = (sampled == eos_token_id)
        for i in range(batch_size):
            finished[i] = finished[i] or new_finished[i]
        sampled = sampled.unsqueeze(1)
        sampled = torch.where(finished.view(-1, 1), torch.full_like(sampled, eos_token_id), sampled)
        current_seq = torch.cat([current_seq, sampled], dim=1)
        if finished.all():
            break

    final_rewards = []
    for i in range(batch_size):
        input_prefix_len = input_lens[i]
        generated_tokens = current_seq[i, input_prefix_len:-1]
        generated_str = tokenizer.decode(generated_tokens.tolist())
        full_answer = generated_str.split('EoS')[0]
        try:
            eval_result = int(full_answer)  # since the answer is only composed of digits.
            reward_val = 1.0 if eval_result == target_results[i] else 0.0
        except Exception:
            reward_val = 0.0
        final_rewards.append(reward_val)
    final_rewards = torch.tensor(final_rewards, dtype=torch.float32, device=device)

    # Compute advantages with GAE for each trajectory.
    trajectories = []
    for i in range(batch_size):
        values_tensor = torch.stack(values_list[i])
        next_values = torch.zeros_like(values_tensor)
        next_values[:-1] = values_tensor[1:]
        steps_taken = len(values_list[i])
        rewards_tensor = torch.zeros(steps_taken, device=device)
        rewards_tensor[-1] = final_rewards[i]
        delta_t = rewards_tensor + gamma * next_values[:steps_taken] - values_tensor[:steps_taken]
        advantage_buffer = 0.0
        advantages = []
        for delta, mask in zip(reversed(delta_t.tolist()), reversed(masks_list[i][:steps_taken])):
            advantage_buffer = delta + gamma * gae_lambda * mask * advantage_buffer
            advantages.insert(0, advantage_buffer)
        advantages = torch.tensor(advantages, dtype=torch.float32, device=device).unsqueeze(1)
        trajectory = {
            'states': states_list[i],
            'input_sequence': input_seqs[i],
            'actions': torch.stack(actions_list[i]).to(device),
            'log_probs': torch.stack(log_probs_list[i]).to(device),
            'values': values_tensor.unsqueeze(1),
            'advantages': advantages,
            'returns': (advantages + values_tensor.unsqueeze(1)).detach()
        }
        trajectories.append(trajectory)

    avg_reward = final_rewards.mean().item()
    print(f"Average reward for this iteration: {avg_reward:.4f}")

    # PPO update loop
    for _ in range(ppo_epochs):
        for trajectory in trajectories:
            model.train()
            optimizer.zero_grad()
            num_steps = trajectory['actions'].shape[0]
            states_list_traj = [s.squeeze(0) for s in trajectory['states'][:num_steps]]
            states_batch = pad_sequence(
                states_list_traj,
                batch_first=True,
                padding_value=tokenizer.token_to_id['EoS']
            ).to(device)
            logits, values_new = model(states_batch)
            last_indices = torch.tensor(
                [(s != tokenizer.token_to_id['EoS']).sum().item()-1 for s in states_list_traj],
                device=device
            )
            logits_last = logits[torch.arange(len(last_indices)), last_indices, :]
            values_last = values_new[torch.arange(len(last_indices)), last_indices].unsqueeze(1)
            dist = torch.distributions.Categorical(F.softmax(logits_last, dim=-1))
            new_log_probs = dist.log_prob(trajectory['actions'])
            ratios = torch.exp(new_log_probs - trajectory['log_probs'])
            surr1 = ratios * trajectory['advantages']
            surr2 = torch.clamp(ratios, 1 - clip_param, 1 + clip_param) * trajectory['advantages']
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values_last, trajectory['returns'])
            entropy = dist.entropy().mean()
            total_loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy
            total_loss.backward()
            optimizer.step()
    return avg_reward

# ---------- Supervised Pretraining (new section) ----------

def supervised_pretraining(model, optimizer, dataset, tokenizer, vocab, num_epochs=5, batch_size=16):
    """
    This function pretrains the model in a supervised fashion.
    Each training example is a pair:
       input: problem + "="
       target: answer digits followed by "EoS"
    """
    device = next(model.parameters()).device
    inputs = []
    targets = []
    for problem, target in dataset:
        # Prepare input sequence: the problem followed by "="
        input_str = problem + "="
        input_tokens = tokenizer.encode(input_str)
        inputs.append(torch.tensor(input_tokens, dtype=torch.long))
        # Prepare target sequence: the numerical answer converted to string, then "EoS"
        target_str = str(int(target))  # convert result to string (assumes integer result)
        target_str += "EoS"
        target_tokens = tokenizer.encode(target_str)
        targets.append(torch.tensor(target_tokens, dtype=torch.long))

    # Create a DataLoader for the pretraining dataset.
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=tokenizer.token_to_id['EoS'])
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=tokenizer.token_to_id['EoS'])
    dataset_tensor = TensorDataset(inputs_padded, targets_padded)
    loader = DataLoader(dataset_tensor, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id['EoS'])
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_inputs, batch_targets in loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad()
            logits, _ = model(batch_inputs, num_iters=model.max_iters)
            # We only take the part of logits corresponding to the target length.
            # Flatten the sequences in the batch.
            logits = logits.view(-1, logits.size(-1))
            batch_targets = batch_targets.view(-1)
            loss = criterion(logits, batch_targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Supervised pretraining Epoch {epoch+1} loss: {epoch_loss/len(loader):.4f}")

# ---------- Evaluation Helper (RL phase) ----------

def evaluate_model(model, difficulty_level, num_samples=100, tokenizer=None, device=None):
    model.eval()
    total_reward = 0.0
    eval_data = generate_expressions(num_samples, int(difficulty_level))
    for problem, target in eval_data:
        input_seq = problem + "="
        input_tokens = tokenizer.encode(input_seq)
        input_tensor = torch.tensor([input_tokens], dtype=torch.long).to(device)
        with torch.no_grad():
            generated = model.generate(input_tensor, max_length=20)
            generated_tokens = generated[0].tolist()
            generated_expr = tokenizer.decode(generated_tokens).split('EoS')[0]
            try:
                result = int(generated_expr)  # since answer is a sequence of digits.
                reward = 1.0 if result == target else 0.0
            except:
                reward = 0.0
            total_reward += reward
    avg_reward = total_reward / num_samples
    model.train()
    return avg_reward

# ---------- Main Training and Inference ----------

# --- Vocabulary (Restricted) ---
# NOTE: Our vocabulary is now only digits 0-9 and "EoS"
vocab_chars = ['0','1','2','3','4','5','6','7','8','9','EoS']
vocab = vocab_chars  # Reduced vocabulary
vocab_size = len(vocab)
tokenizer = ExpressionTokenizer(vocab)

# --- Hyperparameters ---
input_dim = 128
recurrent_hidden_dim = 96    # hidden dimension for recurrent block
max_iters = 6
num_layers = 3
num_heads = 4
#learning_rate = 1e-4
learning_rate = 3e-4
ppo_epochs = 4
batch_size_ppo = 32
num_pre_layers = 3
num_post_layers = 2
sup_pretrain_epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model Initialization ---
model = RecallTransformerWithDense(
    num_layers=num_layers,
    input_dim=input_dim,
    hidden_dim=recurrent_hidden_dim,
    vocab_size=vocab_size,
    num_heads=num_heads,
    max_iters=max_iters,
    num_pre_layers=num_pre_layers,
    num_post_layers=num_post_layers
).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# --- Supervised Pretraining Phase ---
# Generate a relatively small pregenerated dataset.
# In our supervised pretraining the target answer is what the model needs to output (only digit tokens).
sup_dataset = generate_expressions(100*sup_pretrain_epochs, num_ops=1)  # for example, 200 problems with low complexity.
print("Starting supervised pretraining...")
supervised_pretraining(model, optimizer, sup_dataset, tokenizer, vocab, num_epochs=sup_pretrain_epochs, batch_size=16)
print("Supervised pretraining complete!\n")

# --- Reinforcement Learning (PPO) Phase ---
num_iterations = 500
current_difficulty = 1  # starting difficulty (number of ops)
difficulty_increment_amount = 0.5
max_difficulty = 5
evaluation_interval = 5
success_threshold = 0.95

for iteration in range(num_iterations):
    # Generate training data at current difficulty
    expressions_data = generate_expressions(batch_size_ppo, int(current_difficulty))
    dataset_rl = expressions_data
    avg_reward = ppo_iteration_batch(
                    model, optimizer, ppo_epochs, batch_size_ppo,
                    dataset_rl, tokenizer, vocab)
    print(f"\n--- Iteration {iteration + 1} ---")
    # For demonstration, we print the input of the first example.
    example_input = sup_dataset[0][0] + "="
    example_target = str(int(sup_dataset[0][1])) + "EoS"
    print(f"Example Input: {example_input}")
    print(f"Supervised Target Answer: {example_target}")
    # Evaluate and adjust difficulty
    if (iteration + 1) % evaluation_interval == 0:
        eval_reward = evaluate_model(model, current_difficulty, num_samples=50,
                                     tokenizer=tokenizer, device=device)
        print(f"Iteration {iteration+1} - Difficulty: {current_difficulty}")
        print(f"Evaluation Reward: {eval_reward:.4f}")
        if eval_reward >= success_threshold:
            if current_difficulty < max_difficulty:
                current_difficulty += difficulty_increment_amount
                print(f"Mastered current difficulty! New Difficulty: {current_difficulty}")
            else:
                print("Reached maximum difficulty.")
        else:
            print(f"Not mastered - Continuing at difficulty {current_difficulty}")

print("Training complete!")

# --- Inference Example ---
model.eval()
test_expression = "(8+2)*3="  # note: even though our model outputs only digits, the input
                              # is a full arithmetic expression. The model is only supervised to produce the numeric answer.
test_input_ids = torch.tensor([tokenizer.encode(test_expression)], dtype=torch.long).to(device)
with torch.no_grad():
    generated_sequence = model.generate(test_input_ids, max_length=10, num_iters=max_iters)
    generated_expression_result = tokenizer.decode(generated_sequence[0].tolist())
print(f"Input Expression: {test_expression}")
print(f"Generated Result: {generated_expression_result}")
try:
    evaluated_result = int(generated_expression_result.split('EoS')[0])
    print(f"Evaluated Result: {evaluated_result}")
except Exception as e:
    print(f"Evaluation Error: {e}")