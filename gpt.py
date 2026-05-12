import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from datasets import load_dataset
import numpy as np
torch.set_float32_matmul_precision("high")

def scaled_dot_product_attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)

    # 1. calculate dot products QK
    attention_scores = torch.matmul(query, key.transpose(-2, -1))

    # 2. scale the scores
    attention_scores = attention_scores / math.sqrt(d_k)
    
    # 3. apply mask if given
    if mask is not None:
        attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
    
    # 4. apply softmax
    attention_weights = F.softmax(attention_scores, dim=-1)

    # 5. apply dropout
    if dropout is not None:
        attention_weights = dropout(attention_weights)

    # 6.  multiply weights by values
    output = torch.matmul(attention_weights, value)

    return output, attention_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.W_qkv = nn.Linear(d_model, 3 * d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
    
    def split_heads(self, x):
        B, L, D = x.size()
        x = x.view(B, L, self.num_heads, self.head_dim)
        x = x.transpose(1, 2)

        return x
    
    def combine_heads(self, x):
        B, H, L, Hd = x.size()
        x = x.transpose(1,2).contiguous()
        x = x.view(B, L, H * Hd)
        return x
    
    def forward(self, x, mask=None):
        qkv = self.W_qkv(x)
        Q, K, V = qkv.chunk(3, dim=-1)

        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        if mask is not None and mask.dim() == 3:
            mask = mask.unsqueeze(1)
        
        attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask=mask)
        attn_output = self.combine_heads(attn_output)
        attn_output = self.W_o(attn_output)
        attn_output = self.dropout(attn_output)

        return attn_output, attn_weights
    
# position-wise feed forward network
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, tgt_mask=None):
        self_attn_input = self.norm1(x)
        self_attn_output, _ = self.self_attn(
            x=self_attn_input,
            mask=tgt_mask
        )
        x = x + self.dropout1(self_attn_output)

        ff_output = self.feed_forward(self.norm3(x))
        x = x + self.dropout3(ff_output)

        return x
    
class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, n_layer, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(n_layer)])
        self.norm = nn.LayerNorm(d_model)

    def _build_causal_mask(self, x):
        B, T, _ = x.size()
        m = torch.tril(torch.ones((T, T), device=x.device, dtype=torch.bool))
        m = m.unsqueeze(0).unsqueeze(1).expand(B, 1, T, T)
        return m

    def forward(self, x):
        tgt_mask = self._build_causal_mask(x)
        for layer in self.layers:
            x = layer(x, tgt_mask=tgt_mask)
        x = self.norm(x)
        return x
    
class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, n_layer, dropout=0.1, max_seq_len=512):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, d_model)
        self.wpe = nn.Embedding(max_seq_len, d_model)
        self.decoder = Decoder(d_model, num_heads, d_ff, n_layer, dropout)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight

        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("W_o.weight") or name.endswith("linear2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * n_layer))
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        
    def forward(self, idx, targets=None):
        B, T = idx.size()
        x = self.wte(idx)
        pos = torch.arange(T, device=idx.device).unsqueeze(0).expand(B, T)
        x = x + self.wpe(pos)
        x = self.decoder(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss
    
    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        rng = None
        device = next(self.parameters()).device
        if temperature > 0:
            rng = torch.Generator(device="cpu")
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # add batch dim
        for _ in range(max_tokens):
            max_seq_len = self.wpe.num_embeddings
            ids_cond = ids[:, - max_seq_len :]
            logits, _ = self.forward(ids_cond)
            logits = logits[:, -1, :]
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1).cpu()
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng).to(device)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token

def get_device():
    if torch.xpu.is_available():
        return "xpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

device = get_device()
batch_size = 32
learning_rate = 1e-3

warmup_iters = 50
max_iters = 2000
eval_interval = 200
eval_iters = 10
block_size = 128

def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1)/ warmup_iters
    progress = (it - warmup_iters)/max(1, max_iters - warmup_iters)
    return learning_rate * 0.5 * (1.0 + math.cos(math.pi*progress)) * 0.9 + learning_rate * 0.1

dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")

from transformers import GPT2TokenizerFast
enc = GPT2TokenizerFast.from_pretrained("gpt2")

def tokenize(example):
    return {"input_ids": enc.encode(example["text"])}

tokenized = dataset.map(tokenize, remove_columns=["text"])

def create_chunks(dataset_split, chunk_size=256):
    all_tokens = []
    for example in dataset_split:
        all_tokens.extend(example["input_ids"])

    all_tokens = np.array(all_tokens)

    n_chunks = len(all_tokens) // chunk_size
    all_tokens = all_tokens[:n_chunks * chunk_size]
    chunks = all_tokens.reshape(-1, chunk_size)

    return chunks

train_data = create_chunks(tokenized["train"], chunk_size=block_size)
test_data = create_chunks(tokenized["test"], chunk_size=block_size)
val_data = create_chunks(tokenized["validation"], chunk_size=block_size)

train_data = torch.tensor(train_data, dtype=torch.long, device=device)
test_data = torch.tensor(test_data, dtype=torch.long, device=device)
val_data = torch.tensor(val_data, dtype=torch.long, device=device)

print(f"Train: {train_data.shape}")  # (num_sequences, 256)
print(f"Val: {val_data.shape}")

model = GPT(
    vocab_size=enc.vocab_size,
    d_model=128,
    num_heads=4,
    n_layer=4,
    d_ff=512,
    dropout=0.1
    )

print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

model = model.to(device)

decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2]
nondecay_params = [p for n, p in model.named_parameters() if p.dim() < 2]

optimizer = torch.optim.AdamW([
    {'params': decay_params, 'weight_decay': 0.1},
    {'params': nondecay_params, 'weight_decay': 0.0}
], lr=learning_rate, betas=(0.9, 0.95))
losses = []
avg_val_losses = []

scaler = torch.amp.GradScaler(device=device)

for iter_num in range(max_iters):
    lr = get_lr(iter_num)
    for pg in optimizer.param_groups:
        pg["lr"] = lr

    ix = torch.randint(len(train_data), (batch_size,))
    x = train_data[ix, :-1]
    y = train_data[ix, 1:]

    with torch.autocast(device_type=device, dtype=torch.float16):
        _, loss = model(x, y)

    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()

    with torch.no_grad():
        losses.append(loss.item())
    
    if iter_num % eval_interval == 0 or iter_num == max_iters - 1:
        model.eval()
        with torch.no_grad():
            val_losses = []
            for _ in range(eval_iters):
                ix = torch.randint(len(val_data), (batch_size,))
                x = val_data[ix, :-1]
                y = val_data[ix, 1:]
                with torch.autocast(device_type=device, dtype=torch.float16):
                    _, val_loss = model(x, y)
                val_losses.append(val_loss.item())
            avg_val_loss = sum(val_losses) / len(val_losses)
            avg_val_losses.append(avg_val_loss)
        print(f"Iter {iter_num}: Train Loss {loss.item():.4f}, Val Loss {avg_val_loss:.4f}")
        model.train()

# plot
import matplotlib.pyplot as plt
plt.plot(losses)
plt.plot(np.linspace(0, len(losses), len(avg_val_losses)), avg_val_losses)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend(["Train Loss", "Val Loss"])
plt.show()
# Save the model
torch.save(model.state_dict(), "gpt_model.pth")

# Quick generation smoke test.
model.eval()
prompt = "The meaning of life is"
prompt_ids = enc.encode(prompt)
generated_ids = prompt_ids + list(
    model.generate(prompt_ids, max_tokens=80, temperature=0.8, top_k=50)
)
print("\nSample generation:")
print(enc.decode(generated_ids))
