#!/usr/bin/env python3
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
#############################################
# 1. ARITHMETIC EXPRESSION GENERATOR
#############################################
def generate_expression(num_ops):
    if num_ops == 0:
        return str(random.randint(0, 9))
    left_ops = random.randint(0, num_ops - 1)
    right_ops = num_ops - 1 - left_ops
    left_expr = generate_expression(left_ops)
    right_expr = generate_expression(right_ops)
    op = random.choice(['+', '-', '*', '/'])
    return f"({left_expr}{op}{right_expr})"
def generate_expressions(num_expressions, num_ops):
    expr_list = []
    for _ in range(num_expressions): #changed to for loop for cleaner generation
        while True: # loop until a valid expression is generated
            expr = generate_expression(num_ops)
            try:
                result = eval(expr)
                expr_list.append((expr, str(int(result))))
                break # valid expression found, exit inner loop
            except ZeroDivisionError:
                pass # silently skip and regenerate
            except (SyntaxError, TypeError, NameError, ValueError):
                pass # silently skip and regenerate
            except Exception:
                pass # silently skip and regenerate
    return expr_list
#############################################
# 2. VOCABULARY AND TEXT METHODS
#############################################
INPUT_TOKENS = ['<PAD>', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '(', ')', '+', '-', '/', '*', '=']
OUTPUT_TOKENS = ['<PAD>', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'EOS']
input_token2id = {t: i for i, t in enumerate(INPUT_TOKENS)}
output_token2id = {t: i for i, t in enumerate(OUTPUT_TOKENS)}
id2input_token = {i: t for t, i in input_token2id.items()}
id2output_token = {i: t for t, i in output_token2id.items()}
def encode_input(text):
    return [input_token2id[t] for t in text]
def encode_output(text):
    return [output_token2id[t] for t in text]
def decode_input(token_ids):
    return "".join([id2input_token[id] for id in token_ids if id != input_token2id['<PAD>']])
def decode_output(token_ids):
    decoded_tokens = []
    for id in token_ids:
        token = id2output_token[id]
        if token == 'EOS':
            break
        if token != '<PAD>': # Good practice to also skip PAD in output decoding if present (though less likely in this specific problem)
            decoded_tokens.append(token)
    return "".join(decoded_tokens)
#############################################
# 3. ENHANCED DATASET CLASS
#############################################
class ArithmeticDataset(Dataset):
    def __init__(self, num_samples, current_difficulty, max_difficulty):
        self.samples = []
        difficulties = [max(1, current_difficulty-1), current_difficulty] if current_difficulty > 1 else [current_difficulty] # Ensure difficulty 1 is always included if current_difficulty is 1
        for diff in difficulties:
            num = num_samples // len(difficulties)
            problems = generate_expressions(num, diff)
            for expr, result_str in problems:
                cleaned_expr = expr + "="
                target_seq = list(result_str) + ['EOS']
                current = cleaned_expr
                for token in target_seq:
                    self.samples.append((encode_input(current), output_token2id[token]))
                    current = current + token
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        input_ids, target_id = self.samples[idx]
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(target_id, dtype=torch.long)
    def collate_fn(batch):
        inputs, targets = zip(*batch)
        max_len = max(len(seq) for seq in inputs)
        padded_inputs = [F.pad(seq, (0, max_len - len(seq)), value=input_token2id['<PAD>']) for seq in inputs]
        return torch.stack(padded_inputs), torch.stack(targets)
#############################################
# 4. ENHANCED MODEL ARCHITECTURE
#############################################
def rotary_position_embedding(x, seq_dim=2):
    batch, num_heads, seq_len, d = x.shape
    if d % 2 != 0: raise ValueError("RoPE requires even hidden dimension")
    pos = torch.arange(seq_len, dtype=x.dtype, device=x.device).unsqueeze(1)
    freq_seq = torch.arange(0, d//2, dtype=x.dtype, device=x.device)
    inv_freq = 1.0 / (10000 ** (freq_seq / (d//2)))
    sinusoid_inp = pos * inv_freq.unsqueeze(0)
    sin_emb = torch.sin(sinusoid_inp).unsqueeze(0).unsqueeze(0)
    cos_emb = torch.cos(sinusoid_inp).unsqueeze(0).unsqueeze(0)
    x1 = x[..., :d//2]
    x2 = x[..., d//2:]
    x_rot = torch.cat((x1 * cos_emb - x2 * sin_emb, x1 * sin_emb + x2 * cos_emb), dim=-1)
    return x_rot
class MultiHeadSelfAttentionRoPE(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
    def forward(self, x):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        q = rotary_position_embedding(q)
        k = rotary_position_embedding(k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(context)
class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.mhsa = MultiHeadSelfAttentionRoPE(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        attn_out = self.mhsa(x)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        ffn_out = self.ffn(x)
        x = x + self.dropout(ffn_out)
        x = self.norm2(x)
        return x
class RCTNetIPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config['vocab_size'], config['D_initial'])
        self.embed_dropout = nn.Dropout(config['dropout'])
        self.initial_transformer = nn.ModuleList(
            [TransformerEncoderBlock(config['D_initial'], config['num_heads'], config['d_ff'], config['dropout'])
             for _ in range(config['N_initial'])]
        )
        self.proj_initial_to_recurrent = nn.Sequential(
            nn.Linear(config['D_initial'], config['D_recurrent']),
            nn.LayerNorm(config['D_recurrent']),
            nn.GELU()
        )
        self.recurrent_block = nn.ModuleList(
            [TransformerEncoderBlock(config['D_recurrent'], config['num_heads'], config['d_ff'], config['dropout'])
             for _ in range(config['L_recurrent'])]
        )
        self.halting_unit = nn.Sequential(
            nn.Linear(2 * config['D_recurrent'], config['D_halt']),
            nn.GELU(),
            nn.LayerNorm(config['D_halt']),
            nn.Linear(config['D_halt'], 1)
        )
        self.final_transformer = nn.ModuleList(
            [TransformerEncoderBlock(config['D_final'], config['num_heads'], config['d_ff'], config['dropout'])
             for _ in range(config['N_final'])]
        )
        self.proj_recurrent_to_final = nn.Sequential(
            nn.Linear(config['D_recurrent'], config['D_final']),
            nn.LayerNorm(config['D_final']),
            nn.GELU()
        )
        self.output_head = nn.Linear(config['D_final'], config['output_dim'])
    def forward(self, input_tokens, m_iterations):
        x = self.embedding(input_tokens)
        x = self.embed_dropout(x)
        for layer in self.initial_transformer:
            x = layer(x)
        recurrent_init = self.proj_initial_to_recurrent(x)
        recurrent_state = recurrent_init
        for _ in range(m_iterations):
            for layer in self.recurrent_block:
                recurrent_state = layer(recurrent_state)
        final_in = self.proj_recurrent_to_final(recurrent_state)
        for layer in self.final_transformer:
            final_in = layer(final_in)
        return self.output_head(final_in[:, -1, :])
#############################################
# 5. ENHANCED TRAINING LOOP
#############################################
def evaluate_model(model, dataloader, m_iterations, device, print_examples=False, num_examples_to_print=2): # added print_examples and num_examples_to_print
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    example_count = 0 # Counter for printed examples
    max_examples = num_examples_to_print # Maximum number of examples to print, now configurable
    with torch.no_grad():
        for input_ids, targets in dataloader:
            input_ids, targets = input_ids.to(device), targets.to(device)
            logits = model(input_ids, m_iterations)
            predictions = torch.argmax(F.softmax(logits, dim=-1), dim=-1)
            correct_predictions += (predictions == targets).sum().item()
            total_predictions += targets.size(0)
            if print_examples and example_count < max_examples: # Print example predictions only if print_examples is True
                for i in range(input_ids.size(0)): # Iterate through batch examples
                    input_seq = input_ids[i].tolist()
                    target_token = targets[i].item()
                    predicted_token = predictions[i].item()
                    input_expr = decode_input(input_seq) # Decode input, remove padding in decode_input
                    predicted_output = decode_output([predicted_token])
                    target_output = decode_output([target_token])
                    print(f"  Input: '{input_expr}'") # Indented example prints
                    print(f"  Predicted: '{predicted_output}', Target: '{target_output}'") # Indented example prints
                    example_count += 1
                    if example_count >= max_examples: # Stop printing after max examples
                        break
                if example_count >= max_examples:
                    break # Break outer loop as well if max examples reached
    return correct_predictions / total_predictions
def train_curriculum(model, device, base_num_epochs=3, base_num_samples=1000,
                    m_iterations=8, alpha=0.5, max_difficulty=4, batch_size=32, print_batch_loss_every=100, stabilization_patience=2): # added print_batch_loss_every and stabilization_patience
    def get_dynamic_threshold(current_diff):
        return min(0.85 + 0.05 * current_diff, 0.95)
    optimizer = AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    current_difficulty = 1
    total_steps = base_num_epochs * (max_difficulty * base_num_samples // batch_size)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=total_steps)
    success_count = 0 # Counter for consecutive successful validations
    while current_difficulty <= max_difficulty:
        threshold = get_dynamic_threshold(current_difficulty)
        print(f"\n--- Difficulty {current_difficulty}/{max_difficulty} (threshold: {threshold:.2f}) ---") # More concise difficulty print
        dataset = ArithmeticDataset(base_num_samples, current_difficulty, max_difficulty)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=ArithmeticDataset.collate_fn)
        for epoch in range(base_num_epochs):
            model.train()
            epoch_loss = 0.0
            for batch_idx, (input_ids, targets) in enumerate(dataloader):
                input_ids, targets = input_ids.to(device), targets.to(device)
                optimizer.zero_grad()
                logits = model(input_ids, m_iterations)
                loss = loss_fn(logits, targets)
                loss.backward()
                #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                epoch_loss += loss.item()
                if batch_idx % print_batch_loss_every == 0: # Reduced batch loss printing frequency
                    print(f"  Epoch {epoch+1}/{base_num_epochs} Batch {batch_idx} Loss: {loss.item():.4f}") # Indented batch loss print
            avg_loss = epoch_loss / len(dataloader)
            print(f"  Epoch {epoch+1}/{base_num_epochs} Avg Loss: {avg_loss:.4f}") # Indented epoch avg loss print
            # Evaluation
            val_dataset = ArithmeticDataset(800, current_difficulty, max_difficulty)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=ArithmeticDataset.collate_fn)
            acc = evaluate_model(model, val_loader, m_iterations, device, print_examples=True, num_examples_to_print=1) # Reduced example prints in validation
            print(f"  Validation accuracy: {acc*100:.2f}%") # Indented validation accuracy print
            if acc >= threshold:
                success_count += 1
                print(f"  Threshold reached, success count: {success_count}/{stabilization_patience}") # Indicate success count
                if success_count >= stabilization_patience:
                    current_difficulty += 1
                    success_count = 0 # reset success count after difficulty increase
                    print(f"  Difficulty increased to {current_difficulty}") # Indented difficulty increase print
                else:
                    print(f"  Continuing at difficulty {current_difficulty} to stabilize") # Indicate continuing for stabilization
            else:
                success_count = 0 # reset success count if threshold not reached
                print(f"  Validation accuracy below threshold. Repeating difficulty {current_difficulty}") # Indented repeating difficulty print
        print(f"Finished difficulty {current_difficulty-1}, moving to next.") # Clarify difficulty completion before moving to next
    print("Curriculum training complete.") # Keep curriculum completion message
#############################################
# 6. MAIN EXECUTION
#############################################
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = {
        "N_initial": 3,
        "D_initial": 512,
        "L_recurrent": 4,
        "D_recurrent": 768,
        "H_layers": 2, # H_layers is not used, removed from prints.
        "D_halt": 256,
        "N_final": 2,
        "D_final": 256,
        "num_heads": 8,
        "d_ff": 2048,
        "dropout": 0.1,
        "vocab_size": len(INPUT_TOKENS),
        "output_dim": len(OUTPUT_TOKENS),
    }
    model = RCTNetIPT(config).to(device)
    train_curriculum(model, device, print_batch_loss_every=200, stabilization_patience=3) # Example of changing batch loss print frequency and setting stabilization patience