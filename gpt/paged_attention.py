"""Paged KV cache + paged decoding.

Instead of one contiguous (past_k, past_v) tensor per layer that we torch.cat
onto every step, the KV cache is stored in fixed-size *blocks* drawn from a
pool. A *block table* maps logical token positions to physical block ids, so
blocks can live anywhere in the pool in any order (like OS virtual memory).

Growing the sequence = pop a free block; no copy of existing K/V, no need to
pre-reserve a worst-case contiguous slab. Attention "gathers" K/V by following
the block table. This single-sequence implementation keeps the mechanics
visible; a batched engine (vLLM) uses one block table per sequence and a fused
gather+attention kernel, but the memory model is exactly this.
"""
import math
import torch
import torch.nn.functional as F


class PagedKVCache:
    def __init__(self, n_layers, num_kv_heads, head_dim, max_seq_len,
                 block_size=16, device="cpu", dtype=torch.float32):
        self.block_size = block_size
        self.n_layers = n_layers
        num_blocks = math.ceil(max_seq_len / block_size) + 1

        # Physical pools. Shape: (layer, block, slot_in_block, kv_head, head_dim).
        shape = (n_layers, num_blocks, block_size, num_kv_heads, head_dim)
        self.k_pool = torch.zeros(shape, device=device, dtype=dtype)
        self.v_pool = torch.zeros(shape, device=device, dtype=dtype)

        self.free_blocks = list(range(num_blocks))   # pool of unused physical ids
        self.block_table = []                         # logical block -> physical id

    def _ensure_capacity(self, total_len):
        """Allocate physical blocks until block_table covers total_len tokens.
        Idempotent: safe to call once per layer with the same total_len."""
        needed = math.ceil(total_len / self.block_size)
        while len(self.block_table) < needed:
            self.block_table.append(self.free_blocks.pop(0))

    def write(self, layer, start_pos, k, v):
        """Store new K/V. k, v: (1, num_kv_heads, L, head_dim) for this layer."""
        L = k.size(2)
        self._ensure_capacity(start_pos + L)
        for t in range(L):
            pos = start_pos + t
            blk = self.block_table[pos // self.block_size]
            off = pos % self.block_size
            self.k_pool[layer, blk, off] = k[0, :, t, :]
            self.v_pool[layer, blk, off] = v[0, :, t, :]

    def read(self, layer, total_len):
        """Gather K/V for positions [0, total_len) by following the block table.
        Returns (K, V) each (1, num_kv_heads, total_len, head_dim)."""
        n = math.ceil(total_len / self.block_size)
        blocks = [self.block_table[b] for b in range(n)]
        K = self.k_pool[layer, blocks].reshape(-1, *self.k_pool.shape[3:])[:total_len]
        V = self.v_pool[layer, blocks].reshape(-1, *self.v_pool.shape[3:])[:total_len]
        # (total_len, kv_head, head_dim) -> (1, kv_head, total_len, head_dim)
        return K.permute(1, 0, 2).unsqueeze(0), V.permute(1, 0, 2).unsqueeze(0)


def _paged_attention(attn, x, cache, layer_idx, start_pos, mask):
    """One attention block using the paged cache. Mirrors GQA.forward but
    reads/writes K/V through the block table instead of torch.cat."""
    Q, K, V = attn.project(x, start_pos)          # K,V are the NEW tokens only
    cache.write(layer_idx, start_pos, K, V)
    K, V = cache.read(layer_idx, start_pos + x.size(1))
    K, V = attn.repeat_kv(K), attn.repeat_kv(V)   # GQA: expand kv heads to q heads
    out = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask)
    return attn.W_o(attn.combine_heads(out))


def paged_forward(model, idx, cache, start_pos):
    """Full forward pass routing attention through the paged cache. Mirrors
    Decoder/DecoderLayer.forward; dropout is identity in eval, MoE aux ignored."""
    dec = model.decoder
    L = idx.size(1)
    # Query rows are positions [start_pos, start_pos+L); keys span [0, start_pos+L).
    # Row i may attend key j iff j <= start_pos+i  (causal).
    mask = dec.causal_mask[start_pos:start_pos + L, :start_pos + L]
    mask = mask.unsqueeze(0).unsqueeze(0)         # (1,1,L,total_len)

    x = model.wte(idx)
    for i, layer in enumerate(dec.layers):
        attn_out = _paged_attention(layer.self_attn, layer.norm1(x), cache, i, start_pos, mask)
        x = x + attn_out
        if layer.is_moe:
            ff, _ = layer.feed_forward(layer.norm3(x))
        else:
            ff = layer.feed_forward(layer.norm3(x))
        x = x + ff
    return model.lm_head(dec.norm(x))


@torch.inference_mode()
def generate_paged(model, tokens, max_tokens, temperature=1.0, top_k=None,
                   seed=42, block_size=16):
    """Same interface and output distribution as GPT.generate, but backed by a
    PagedKVCache. Yields tokens one at a time."""
    from .sampling import sample_token

    device = next(model.parameters()).device
    rng = torch.Generator(device="cpu").manual_seed(seed) if temperature > 0 else None
    cache = PagedKVCache(
        n_layers=len(model.decoder.layers),
        num_kv_heads=model.decoder.layers[0].self_attn.num_kv_heads,
        head_dim=model.decoder.layers[0].self_attn.head_dim,
        max_seq_len=model.max_seq_len,
        block_size=block_size,
        device=device,
        dtype=next(model.parameters()).dtype,
    )

    ids = torch.tensor([tokens], dtype=torch.long, device=device)
    logits = paged_forward(model, ids, cache, start_pos=0)[:, -1, :]  # prefill
    cur_pos = ids.size(1)

    for _ in range(max_tokens):
        nxt = sample_token(logits, temperature, top_k, rng, device)
        yield nxt.item()
        if cur_pos >= model.max_seq_len:
            break
        logits = paged_forward(model, nxt, cache, start_pos=cur_pos)[:, -1, :]
        cur_pos += 1
