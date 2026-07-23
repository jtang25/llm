import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import GroupedQueryAttention
from .layers import RMSNorm, SwiGLUFeedForward, MoEFeedForward
from .sampling import sample_token
from .config import GPTConfig


class DecoderLayer(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.self_attn = GroupedQueryAttention(
            cfg.d_model, cfg.num_heads, cfg.num_kv_heads, cfg.max_seq_len, cfg.dropout
        )
        self.is_moe = cfg.num_experts is not None and cfg.num_experts > 1
        self.feed_forward = (
            MoEFeedForward(cfg.d_model, cfg.d_ff, cfg.num_experts,
                           cfg.num_experts_per_tok, cfg.dropout)
            if self.is_moe else
            SwiGLUFeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)
        )
        self.norm1 = RMSNorm(cfg.d_model)
        self.norm3 = RMSNorm(cfg.d_model)
        self.dropout1 = nn.Dropout(cfg.dropout)
        self.dropout3 = nn.Dropout(cfg.dropout)

    def forward(self, x, tgt_mask=None, kv_cache=None, start_pos=0):
        attn_out, new_cache = self.self_attn(
            self.norm1(x), mask=tgt_mask, kv_cache=kv_cache, start_pos=start_pos
        )
        x = x + self.dropout1(attn_out)

        if self.is_moe:
            ff, aux = self.feed_forward(self.norm3(x))
        else:
            ff, aux = self.feed_forward(self.norm3(x)), None
        x = x + self.dropout3(ff)
        return x, new_cache, aux


class Decoder(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.n_layer)])
        self.norm = RMSNorm(cfg.d_model)
        mask = torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", mask)

    def forward(self, x, kv_caches, start_pos=0):
        L = x.size(1)
        # Only need a mask when processing >1 token at once (prefill/training).
        # For a single decode step the one query attends the whole cache.
        tgt_mask = self.causal_mask[:L, :L].unsqueeze(0).unsqueeze(0) if L > 1 else None

        new_caches, aux_total, n_moe = [], 0.0, 0
        for i, layer in enumerate(self.layers):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            x, new_cache, aux = layer(x, tgt_mask=tgt_mask,
                                      kv_cache=layer_cache, start_pos=start_pos)
            new_caches.append(new_cache)
            if aux is not None:
                aux_total, n_moe = aux_total + aux, n_moe + 1
        return self.norm(x), new_caches, (aux_total / n_moe if n_moe else None)


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.max_seq_len = cfg.max_seq_len
        self.aux_coef = cfg.aux_coef
        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.decoder = Decoder(cfg)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight   # weight tying

        self.apply(self._init_weights)
        # Scale residual-projection init by depth (GPT-2 trick).
        for name, p in self.named_parameters():
            if name.endswith("W_o.weight") or name.endswith("w_down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def forward(self, idx, targets=None, kv_caches=None, start_pos=0):
        x = self.wte(idx)
        x, new_caches, aux = self.decoder(x, kv_caches=kv_caches, start_pos=start_pos)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            if aux is not None:
                loss = loss + self.aux_coef * aux
        return logits, loss, new_caches

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None,
                 seed=42, use_kv_cache=True):
        """Yields tokens one at a time. use_kv_cache=False recomputes the whole
        sequence every step (slow reference path)."""
        device = next(self.parameters()).device
        rng = torch.Generator(device="cpu").manual_seed(seed) if temperature > 0 else None
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        n_layer = len(self.decoder.layers)

        def _sample(logits):
            return sample_token(logits, temperature, top_k, rng, device)

        if use_kv_cache:
            # Prefill: one pass over the prompt fills the cache.
            empty = [(None, None) for _ in range(n_layer)]
            logits, _, kv_caches = self.forward(ids, kv_caches=empty, start_pos=0)
            logits = logits[:, -1, :]
            cur_pos = ids.size(1)
            for _ in range(max_tokens):
                nxt = _sample(logits)
                yield nxt.item()
                if cur_pos >= self.max_seq_len:
                    break
                # Decode: feed only the new token; cache supplies the past.
                logits, _, kv_caches = self.forward(nxt, kv_caches=kv_caches, start_pos=cur_pos)
                logits = logits[:, -1, :]
                cur_pos += 1
        else:
            seq = ids
            for _ in range(max_tokens):
                empty = [(None, None) for _ in range(n_layer)]
                cond = seq[:, -self.max_seq_len:]
                logits, _, _ = self.forward(cond, kv_caches=empty, start_pos=0)
                nxt = _sample(logits[:, -1, :])
                yield nxt.item()
                seq = torch.cat([seq, nxt], dim=1)


def build_model(cfg: GPTConfig) -> GPT:
    return GPT(cfg)
