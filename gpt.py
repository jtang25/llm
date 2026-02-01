import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pandas as pd
from datasets import load_dataset
import tiktoken
import numpy as np
import torch._inductor.config
torch._inductor.config.triton.cudagraphs = False
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.allow_tf32 = True

def scaled_dot_product_attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)

    # 1. calculate dot products QK
    attention_scores = torch.matmul(query, key.transpose(-2, -1))

    # 2. scale the scores
    attention_scores = attention_scores / math.sqrt(d_k)
    
    # 3. apply mask if given
    if mask is not None:
        attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
    
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

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
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
    
    def forward(self, q, k, v, mask=None):
        Q = self.W_q(q)
        K = self.W_k(k)
        V = self.W_v(v)

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
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output, _ = self.self_attn(q=self.norm1(x), k=self.norm1(x), v=self.norm1(x), mask=mask)
        x = x + self.dropout1(attn_output)
        
        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout2(ff_output)

        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_output=None, tgt_mask=None, memory_mask=None):
        self_attn_input = self.norm1(x)
        self_attn_output, _ = self.self_attn(
            q=self_attn_input,
            k=self_attn_input,
            v=self_attn_input,
            mask=tgt_mask
        )
        x = x + self.dropout1(self_attn_output)

        if enc_output is not None:
            cross_attn_input = self.norm2(x)
            cross_attn_output, _ = self.cross_attn(
                q=cross_attn_input,
                k=enc_output,
                v=enc_output,
                mask=memory_mask
            )
            x = x + self.dropout2(cross_attn_output)

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
        device = self.get_device()
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # add batch dim
        for _ in range(max_tokens):
            logits, _ = self.forward(ids)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 512
learning_rate = 1e-3
max_iters = 8000
eval_interval = 200
eval_iters = 100
block_size = 256

dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")

enc = tiktoken.get_encoding("gpt2")

def tokenize(example):
    return {"input_ids": enc.encode(example["text"], allowed_special=set())}

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

train_data = create_chunks(tokenized["train"], chunk_size=256)
test_data = create_chunks(tokenized["test"], chunk_size=256)
val_data = create_chunks(tokenized["validation"], chunk_size=256)

train_data = torch.tensor(train_data, dtype=torch.long, device=device)
test_data = torch.tensor(test_data, dtype=torch.long, device=device)
val_data = torch.tensor(val_data, dtype=torch.long, device=device)

print(f"Train: {train_data.shape}")  # (num_sequences, 256)
print(f"Val: {val_data.shape}")

model = GPT(
    vocab_size=enc.n_vocab,
    d_model=512,
    num_heads=8,
    n_layer=8,
    d_ff=2048,
    dropout=0.1
    )
model = torch.compile(model)

print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)
losses = []
avg_val_losses = []

for iter_num in range(max_iters):
    ix = torch.randint(len(train_data), (batch_size, ))
    x = train_data[ix, :-1]
    y = train_data[ix, 1:]

    with torch.amp.autocast(device, dtype=torch.bfloat16):
        _, loss = model(x,y)
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        losses.append(loss.item())
    
    if iter_num % eval_interval == 0 or iter_num == max_iters - 1:
        model.eval()
        with torch.no_grad():
            val_losses = []
            for _ in range(eval_iters):
                ix = torch.randint(len(val_data), (batch_size, ))
                x = val_data[ix, :-1]
                y = val_data[ix, 1:]
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
