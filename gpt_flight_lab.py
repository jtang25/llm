r"""
Offline GPT flight lab for TinyStories on PyTorch XPU.

This file intentionally does not touch gpt.ipynb. It is a reference
implementation you can import into the notebook or run from PowerShell.

Useful commands:

    .\.venv\Scripts\python.exe gpt_flight_lab.py self-test
    .\.venv\Scripts\python.exe gpt_flight_lab.py train --preset flight --iters 2000
    .\.venv\Scripts\python.exe gpt_flight_lab.py kv-bench --preset flight
    .\.venv\Scripts\python.exe gpt_flight_lab.py moe-viz --steps 200
    .\.venv\Scripts\python.exe gpt_flight_lab.py paged-demo

What is included:
  A3: contiguous pre-allocated KV cache with prefill and decode paths
  A4: MHA/GQA/MQA via configurable n_kv_heads
  A5: native bf16 training. torch.compile is skipped on XPU because this
      local Intel Arc stack can fail inside Triton/Inductor.
  A7: top-2 MoE layer with router stats and visualizations
  C1: first-principles paged KV cache and paged attention equivalence demo
"""

from __future__ import annotations

import argparse
import html
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer


# -----------------------------
# Device and data
# -----------------------------


def get_device(name: str = "auto") -> str:
    if name != "auto":
        return name
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def sync_device(device: str) -> None:
    if device == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


class TinyStoriesData:
    def __init__(self, data_dir: str | Path = "data/tinystories"):
        self.data_dir = Path(data_dir)
        self.tokenizer = Tokenizer.from_file(str(self.data_dir / "tokenizer.json"))
        self.vocab_size = self.tokenizer.get_vocab_size()
        self.train = np.memmap(self.data_dir / "train.bin", dtype=np.uint16, mode="r")
        self.val = np.memmap(self.data_dir / "val.bin", dtype=np.uint16, mode="r")

    def get_batch(self, split: str, batch_size: int, block_size: int, device: str):
        data = self.train if split == "train" else self.val
        ix = torch.randint(len(data) - block_size - 1, (batch_size,))
        x = torch.stack(
            [torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix]
        )
        y = torch.stack(
            [torch.from_numpy(data[i + 1 : i + block_size + 1].astype(np.int64)) for i in ix]
        )
        return x.to(device), y.to(device)

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, ids: Iterable[int]) -> str:
        return self.tokenizer.decode(list(ids))


# -----------------------------
# Model config
# -----------------------------


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int = 256
    n_layer: int = 6
    n_heads: int = 8
    n_kv_heads: int = 2
    d_ff: int = 1024
    block_size: int = 256
    dropout: float = 0.0
    moe_layer: int | None = None
    n_experts: int = 4
    moe_aux_weight: float = 0.01


def preset_config(name: str, vocab_size: int, block_size: int | None = None) -> ModelConfig:
    if name == "quick":
        cfg = ModelConfig(
            vocab_size=vocab_size,
            d_model=128,
            n_layer=2,
            n_heads=4,
            n_kv_heads=2,
            d_ff=512,
            block_size=128,
        )
    elif name == "flight":
        cfg = ModelConfig(
            vocab_size=vocab_size,
            d_model=256,
            n_layer=6,
            n_heads=8,
            n_kv_heads=2,
            d_ff=1024,
            block_size=256,
        )
    elif name == "timing":
        cfg = ModelConfig(
            vocab_size=vocab_size,
            d_model=384,
            n_layer=8,
            n_heads=8,
            n_kv_heads=2,
            d_ff=1536,
            block_size=256,
        )
    else:
        raise ValueError(f"unknown preset: {name}")
    if block_size is not None:
        cfg.block_size = block_size
    return cfg


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# -----------------------------
# KV cache and attention
# -----------------------------


class KVCache:
    """Contiguous pre-allocated per-layer KV cache.

    K/V tensors are shaped:
        (batch, n_kv_heads, max_seq, head_dim)
    """

    def __init__(
        self,
        n_layers: int,
        batch: int,
        n_kv_heads: int,
        max_seq: int,
        head_dim: int,
        device: str,
        dtype: torch.dtype,
    ):
        self.k = [
            torch.empty(batch, n_kv_heads, max_seq, head_dim, device=device, dtype=dtype)
            for _ in range(n_layers)
        ]
        self.v = [
            torch.empty(batch, n_kv_heads, max_seq, head_dim, device=device, dtype=dtype)
            for _ in range(n_layers)
        ]
        self.max_seq = max_seq

    def write(self, layer_idx: int, pos: int, k_new: torch.Tensor, v_new: torch.Tensor) -> None:
        t_new = k_new.size(2)
        if pos + t_new > self.max_seq:
            raise ValueError(f"cache write {pos}+{t_new} exceeds max_seq={self.max_seq}")
        self.k[layer_idx][:, :, pos : pos + t_new, :] = k_new
        self.v[layer_idx][:, :, pos : pos + t_new, :] = v_new

    def read(self, layer_idx: int, end_pos: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.k[layer_idx][:, :, :end_pos, :], self.v[layer_idx][:, :, :end_pos, :]

    def bytes(self) -> int:
        tensors = [*self.k, *self.v]
        return sum(t.numel() * t.element_size() for t in tensors)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads for GQA.

    Input:
        x: (B, n_kv_heads, T, head_dim)
    Output:
        x: (B, n_heads, T, head_dim)
    """

    if n_rep == 1:
        return x
    b, h_kv, t, hd = x.shape
    x = x[:, :, None, :, :].expand(b, h_kv, n_rep, t, hd)
    return x.reshape(b, h_kv * n_rep, t, hd)


def cache_position_mask(
    cache_pos: int,
    query_len: int,
    key_len: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Mask from absolute cache positions, not a recomputed square causal mask.

    For decode with one token, there is no need for a mask because the cache
    only exposes positions up through the current token.
    """

    if query_len == 1:
        return None
    q_pos = torch.arange(cache_pos, cache_pos + query_len, device=device)
    k_pos = torch.arange(key_len, device=device)
    return (k_pos[None, :] <= q_pos[:, None])[None, None, :, :]


class CausalSelfAttentionGQA(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        assert cfg.n_heads % cfg.n_kv_heads == 0
        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.n_rep = cfg.n_heads // cfg.n_kv_heads

        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def split_q(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

    def split_kv(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        layer_idx: int | None = None,
        cache: KVCache | None = None,
        cache_pos: int | None = None,
    ) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.split_q(self.q_proj(x))
        k = self.split_kv(self.k_proj(x))
        v = self.split_kv(self.v_proj(x))

        if cache is None:
            k_full, v_full = k, v
            attn_mask = None
            is_causal = True
        else:
            if layer_idx is None or cache_pos is None:
                raise ValueError("layer_idx and cache_pos are required when cache is used")
            cache.write(layer_idx, cache_pos, k, v)
            end_pos = cache_pos + t
            k_full, v_full = cache.read(layer_idx, end_pos)
            attn_mask = cache_position_mask(cache_pos, t, end_pos, x.device)
            is_causal = False

        k_full = repeat_kv(k_full, self.n_rep)
        v_full = repeat_kv(v_full, self.n_rep)
        y = F.scaled_dot_product_attention(
            q,
            k_full,
            v_full,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        return y.transpose(1, 2).contiguous().view(b, t, self.d_model)


# -----------------------------
# MLP and MoE
# -----------------------------


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Top2MoE(nn.Module):
    """Simple Mixtral-style top-2 MoE for learning and visualization."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.router = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)
        self.experts = nn.ModuleList([MLP(cfg) for _ in range(cfg.n_experts)])
        self.n_experts = cfg.n_experts
        self.last_stats: dict[str, torch.Tensor | int] | None = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, d = x.shape
        flat = x.reshape(b * t, d)
        logits = self.router(flat)
        probs = F.softmax(logits, dim=-1)
        top_p, top_i = torch.topk(probs, k=2, dim=-1)
        top_p = top_p / top_p.sum(dim=-1, keepdim=True)

        out = torch.zeros_like(flat)
        for slot in range(2):
            expert_ids = top_i[:, slot]
            weights = top_p[:, slot, None]
            for expert_idx, expert in enumerate(self.experts):
                mask = expert_ids == expert_idx
                if mask.any():
                    out[mask] += weights[mask] * expert(flat[mask])

        top1 = top_i[:, 0]
        tokens_per_expert = F.one_hot(top1, self.n_experts).float().mean(dim=0)
        prob_per_expert = probs.mean(dim=0)
        aux_loss = self.n_experts * torch.sum(tokens_per_expert * prob_per_expert)
        entropy = -(probs * torch.log(probs.clamp_min(1e-9))).sum(dim=-1)

        self.last_stats = {
            "B": b,
            "T": t,
            "top_i": top_i.detach().cpu(),
            "top_p": top_p.detach().cpu(),
            "top1": top1.detach().cpu(),
            "tokens_per_expert": tokens_per_expert.detach().cpu(),
            "prob_per_expert": prob_per_expert.detach().cpu(),
            "entropy": entropy.detach().cpu(),
            "aux_loss": aux_loss.detach().cpu(),
        }
        return out.view(b, t, d), aux_loss


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttentionGQA(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = Top2MoE(cfg) if cfg.moe_layer == layer_idx else MLP(cfg)

    def forward(
        self,
        x: torch.Tensor,
        layer_idx: int,
        cache: KVCache | None = None,
        cache_pos: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = x + self.attn(self.ln1(x), layer_idx=layer_idx, cache=cache, cache_pos=cache_pos)
        y = self.mlp(self.ln2(x))
        aux_loss = None
        if isinstance(y, tuple):
            y, aux_loss = y
        x = x + y
        return x, aux_loss


class TinyGPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.wpe = nn.Embedding(cfg.block_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self.apply(self._init_weights)
        self.lm_head.weight = self.wte.weight

        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("net.2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def new_cache(
        self,
        batch: int,
        max_seq: int,
        dtype: torch.dtype | None = None,
    ) -> KVCache:
        p = next(self.parameters())
        return KVCache(
            n_layers=self.cfg.n_layer,
            batch=batch,
            n_kv_heads=self.cfg.n_kv_heads,
            max_seq=max_seq,
            head_dim=self.cfg.d_model // self.cfg.n_heads,
            device=str(p.device),
            dtype=dtype or p.dtype,
        )

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        cache: KVCache | None = None,
        cache_pos: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        b, t = idx.shape
        if cache_pos is None:
            pos = torch.arange(t, device=idx.device)
        else:
            pos = torch.arange(cache_pos, cache_pos + t, device=idx.device)
        if int(pos[-1]) >= self.cfg.block_size:
            raise ValueError(
                f"position {int(pos[-1])} exceeds block_size={self.cfg.block_size}; "
                "increase block_size or generate fewer tokens"
            )

        x = self.wte(idx) + self.wpe(pos)[None, :, :]
        aux_losses = []
        for i, block in enumerate(self.blocks):
            x, aux_loss = block(x, layer_idx=i, cache=cache, cache_pos=cache_pos)
            if aux_loss is not None:
                aux_losses.append(aux_loss)
        logits = self.lm_head(self.ln_f(x))

        aux = torch.stack(aux_losses).mean() if aux_losses else None
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            if aux is not None:
                loss = loss + self.cfg.moe_aux_weight * aux
        return logits, loss, aux


# -----------------------------
# Generation and benchmarks
# -----------------------------


@torch.inference_mode()
def generate_no_cache(model: TinyGPT, prompt: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    model.eval()
    ids = prompt.clone()
    if ids.size(1) + max_new_tokens > model.cfg.block_size:
        raise ValueError("prompt + max_new_tokens exceeds model.cfg.block_size")
    for _ in range(max_new_tokens):
        logits, _, _ = model(ids)
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, next_id], dim=1)
    return ids


@torch.inference_mode()
def generate_with_cache(model: TinyGPT, prompt: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    model.eval()
    batch, prompt_len = prompt.shape
    total_len = prompt_len + max_new_tokens
    if total_len > model.cfg.block_size:
        raise ValueError("prompt + max_new_tokens exceeds model.cfg.block_size")

    cache = model.new_cache(batch=batch, max_seq=total_len)
    logits, _, _ = model(prompt, cache=cache, cache_pos=0)
    next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    ids = torch.cat([prompt, next_id], dim=1)

    cache_pos = prompt_len
    for _ in range(max_new_tokens - 1):
        logits, _, _ = model(next_id, cache=cache, cache_pos=cache_pos)
        cache_pos += 1
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, next_id], dim=1)
    return ids


def benchmark_decode(
    model: TinyGPT,
    prompt: torch.Tensor,
    max_new_tokens: int,
    repeats: int,
    device: str,
) -> tuple[float, float]:
    model.eval()
    generate_no_cache(model, prompt, 2)
    generate_with_cache(model, prompt, 2)
    sync_device(device)

    t0 = time.perf_counter()
    for _ in range(repeats):
        generate_no_cache(model, prompt, max_new_tokens)
    sync_device(device)
    no_cache_tps = repeats * max_new_tokens / (time.perf_counter() - t0)

    t0 = time.perf_counter()
    for _ in range(repeats):
        generate_with_cache(model, prompt, max_new_tokens)
    sync_device(device)
    cache_tps = repeats * max_new_tokens / (time.perf_counter() - t0)
    return no_cache_tps, cache_tps


@torch.inference_mode()
def greedy_text(model: TinyGPT, data: TinyStoriesData, prompt: str, max_new_tokens: int, device: str) -> str:
    ids = torch.tensor([data.encode(prompt)], dtype=torch.long, device=device)
    out = generate_with_cache(model, ids, max_new_tokens=max_new_tokens)
    return data.decode(out[0].tolist())


# -----------------------------
# Training and eval
# -----------------------------


@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    data: TinyStoriesData,
    batch_size: int,
    block_size: int,
    eval_iters: int,
    device: str,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = []
        for _ in range(eval_iters):
            x, y = data.get_batch(split, batch_size, block_size, device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device != "cpu"):
                _, loss, _ = model(x, y)
            losses.append(float(loss))
        out[split] = sum(losses) / len(losses)
    if was_training:
        model.train()
    return out


def make_optimizer(model: nn.Module, lr: float, weight_decay: float = 0.1):
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.95),
    )


def cosine_lr(it: int, max_iters: int, lr: float, warmup_iters: int) -> float:
    if it < warmup_iters:
        return lr * (it + 1) / max(1, warmup_iters)
    progress = (it - warmup_iters) / max(1, max_iters - warmup_iters)
    return lr * (0.1 + 0.45 * (1.0 + math.cos(math.pi * progress)))


def save_training_plot(history: list[dict[str, float]], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [h["iter"] for h in history if "val_loss" in h]
    train = [h["train_loss"] for h in history if "val_loss" in h]
    val = [h["val_loss"] for h in history if "val_loss" in h]
    plt.figure(figsize=(7, 4))
    plt.plot(steps, train, label="train")
    plt.plot(steps, val, label="val")
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.title("TinyStories loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def run_train(args: argparse.Namespace) -> None:
    device = get_device(args.device)
    data = TinyStoriesData(args.data_dir)
    cfg = preset_config(args.preset, data.vocab_size, args.block_size)
    cfg.dropout = args.dropout
    cfg.n_kv_heads = args.n_kv_heads
    cfg.moe_layer = args.moe_layer
    cfg.n_experts = args.n_experts

    model = TinyGPT(cfg).to(device)
    raw_model = model
    print(f"device: {device}")
    print(f"vocab_size: {data.vocab_size}")
    print(f"model: {count_parameters(raw_model) / 1e6:.2f}M params")
    print(f"config: {cfg}")

    optimizer = make_optimizer(raw_model, args.lr)
    if args.compile and device == "xpu":
        print("torch.compile requested, but skipping on XPU for this lab.")
        print("Use eager bf16 on the laptop; test compile later on CUDA/H200.")
        args.compile = False

    if args.compile:
        print(f"compiling with mode={args.compile_mode}")
        model = torch.compile(model, mode=args.compile_mode)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float]] = []

    for it in range(args.iters):
        lr = cosine_lr(it, args.iters, args.lr, args.warmup_iters)
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = data.get_batch("train", args.batch_size, cfg.block_size, device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device != "cpu"):
            _, loss, aux = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
        optimizer.step()

        if it % args.log_interval == 0:
            aux_text = "" if aux is None else f", moe_aux {float(aux.detach()):.3f}"
            print(f"iter {it:5d}: loss {float(loss.detach()):.4f}, lr {lr:.2e}{aux_text}")

        if it % args.eval_interval == 0 or it == args.iters - 1:
            losses = estimate_loss(model, data, args.batch_size, cfg.block_size, args.eval_iters, device)
            print(
                f"eval {it:5d}: train {losses['train']:.4f}, "
                f"val {losses['val']:.4f}"
            )
            history.append(
                {"iter": it, "train_loss": losses["train"], "val_loss": losses["val"]}
            )

    ckpt_path = out_dir / f"tinygpt_{args.preset}_last.pt"
    torch.save({"model": raw_model.state_dict(), "config": cfg.__dict__}, ckpt_path)
    print(f"saved checkpoint: {ckpt_path}")

    if history:
        plot_path = out_dir / "training_loss.png"
        save_training_plot(history, plot_path)
        print(f"saved plot: {plot_path}")

    try:
        print("sample:")
        print(greedy_text(raw_model, data, args.prompt, args.generate_tokens, device))
    except Exception as exc:
        print(f"generation skipped: {type(exc).__name__}: {exc}")


# -----------------------------
# MoE visualizations
# -----------------------------


def find_moe_modules(model: nn.Module) -> list[Top2MoE]:
    return [m for m in model.modules() if isinstance(m, Top2MoE)]


def save_moe_plots(moe: Top2MoE, out_dir: Path) -> None:
    if moe.last_stats is None:
        raise ValueError("run a forward pass before plotting MoE stats")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stats = moe.last_stats
    b = int(stats["B"])
    t = int(stats["T"])
    top_i = stats["top_i"].numpy()
    top1 = top_i[:, 0].reshape(b, t)
    top2 = top_i[:, 1].reshape(b, t)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    im0 = axes[0].imshow(top1, aspect="auto", interpolation="nearest")
    axes[0].set_title("Top-1 expert per token")
    axes[0].set_xlabel("token position")
    axes[0].set_ylabel("batch item")
    fig.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(top2, aspect="auto", interpolation="nearest")
    axes[1].set_title("Top-2 expert per token")
    axes[1].set_xlabel("token position")
    fig.colorbar(im1, ax=axes[1])
    fig.tight_layout()
    path = out_dir / "moe_routing.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)

    token_frac = stats["tokens_per_expert"].numpy()
    prob_mass = stats["prob_per_expert"].numpy()
    x = np.arange(len(token_frac))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 0.18, token_frac, width=0.36, label="top-1 token fraction")
    ax.bar(x + 0.18, prob_mass, width=0.36, label="router probability mass")
    ax.axhline(1 / len(token_frac), linestyle="--", color="black", linewidth=1, label="perfect")
    ax.set_xlabel("expert")
    ax.set_ylabel("fraction")
    ax.set_title("MoE expert balance")
    ax.legend()
    fig.tight_layout()
    path = out_dir / "moe_balance.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)

    top_p = stats["top_p"].numpy()
    entropy = stats["entropy"].numpy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(top_p[:, 0], bins=30)
    axes[0].set_title("Router top-1 confidence")
    axes[0].set_xlabel("normalized top-1 prob")
    axes[0].set_ylabel("tokens")
    axes[1].hist(entropy, bins=30)
    axes[1].set_title("Router entropy")
    axes[1].set_xlabel("entropy")
    fig.tight_layout()
    path = out_dir / "moe_confidence.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)

    print(f"saved MoE plots in {out_dir}")
    print(f"aux loss: {float(stats['aux_loss']):.4f}")
    print("top-1 token fraction:", np.round(token_frac, 3))
    print("router probability mass:", np.round(prob_mass, 3))


def save_token_expert_html(
    input_ids: torch.Tensor,
    moe: Top2MoE,
    tokenizer: Tokenizer,
    out_path: Path,
    batch_idx: int = 0,
) -> None:
    if moe.last_stats is None:
        raise ValueError("run a forward pass before writing token HTML")
    stats = moe.last_stats
    b = int(stats["B"])
    t = int(stats["T"])
    top1 = stats["top_i"][:, 0].reshape(b, t)[batch_idx].numpy()
    ids = input_ids[batch_idx].detach().cpu().tolist()
    colors = [
        "#e41a1c",
        "#377eb8",
        "#4daf4a",
        "#984ea3",
        "#ff7f00",
        "#a65628",
        "#f781bf",
        "#666666",
    ]
    spans = []
    for tok_id, expert in zip(ids, top1):
        text = tokenizer.decode([int(tok_id)]).replace("\n", "\\n")
        safe = html.escape(text)
        color = colors[int(expert) % len(colors)]
        spans.append(
            f"<span title='expert {int(expert)}' "
            f"style='background:{color}; color:white; padding:2px 4px; "
            f"margin:1px; border-radius:3px; display:inline-block'>{safe}</span>"
        )
    doc = (
        "<!doctype html><meta charset='utf-8'>"
        "<body style='font-family: system-ui, sans-serif; line-height:2.0'>"
        "<h3>Tokens colored by top-1 expert</h3>"
        + "".join(spans)
        + "</body>"
    )
    out_path.write_text(doc, encoding="utf-8")
    print(f"saved token expert HTML: {out_path}")


def run_moe_viz(args: argparse.Namespace) -> None:
    device = get_device(args.device)
    data = TinyStoriesData(args.data_dir)
    cfg = preset_config("quick", data.vocab_size, args.block_size)
    cfg.moe_layer = 1
    cfg.n_experts = args.n_experts
    cfg.dropout = 0.0
    model = TinyGPT(cfg).to(device)
    optimizer = make_optimizer(model, args.lr)

    print(f"device: {device}")
    print(f"MoE model: {count_parameters(model) / 1e6:.2f}M params")
    model.train()
    for step in range(args.steps):
        x, y = data.get_batch("train", args.batch_size, cfg.block_size, device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device != "cpu"):
            _, loss, aux = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % max(1, args.steps // 5) == 0:
            print(f"step {step:4d}: loss {float(loss):.4f}, aux {float(aux):.4f}")

    model.eval()
    x, y = data.get_batch("val", args.batch_size, cfg.block_size, device)
    with torch.no_grad():
        model(x, y)
    moes = find_moe_modules(model)
    if not moes:
        raise RuntimeError("no MoE modules found")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_moe_plots(moes[0], out_dir)
    save_token_expert_html(x, moes[0], data.tokenizer, out_dir / "moe_tokens.html")


# -----------------------------
# Paged attention demo
# -----------------------------


class PagedKVCache:
    """Fixed-size block KV cache with per-sequence block tables."""

    def __init__(
        self,
        n_layers: int,
        n_blocks: int,
        block_size: int,
        n_kv_heads: int,
        head_dim: int,
        device: str,
        dtype: torch.dtype,
    ):
        self.k = torch.empty(
            n_layers, n_blocks, n_kv_heads, block_size, head_dim, device=device, dtype=dtype
        )
        self.v = torch.empty_like(self.k)
        self.block_size = block_size
        self.free = list(range(n_blocks))
        self.block_tables: dict[str, list[int]] = {}
        self.lengths: dict[str, int] = {}

    def start_sequence(self, seq_id: str) -> None:
        self.block_tables[seq_id] = []
        self.lengths[seq_id] = 0

    def _ensure_block_for_pos(self, seq_id: str, pos: int) -> int:
        table = self.block_tables[seq_id]
        block_index = pos // self.block_size
        while len(table) <= block_index:
            if not self.free:
                raise RuntimeError("out of paged KV blocks")
            table.append(self.free.pop())
        return table[block_index]

    def write_token(
        self,
        layer_idx: int,
        seq_id: str,
        pos: int,
        k_token: torch.Tensor,
        v_token: torch.Tensor,
    ) -> None:
        block_id = self._ensure_block_for_pos(seq_id, pos)
        offset = pos % self.block_size
        self.k[layer_idx, block_id, :, offset, :] = k_token[0, :, 0, :]
        self.v[layer_idx, block_id, :, offset, :] = v_token[0, :, 0, :]
        self.lengths[seq_id] = max(self.lengths[seq_id], pos + 1)

    def write_many(
        self,
        layer_idx: int,
        seq_id: str,
        start_pos: int,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
    ) -> None:
        for i in range(k_new.size(2)):
            self.write_token(
                layer_idx,
                seq_id,
                start_pos + i,
                k_new[:, :, i : i + 1, :],
                v_new[:, :, i : i + 1, :],
            )

    def gather(self, layer_idx: int, seq_id: str, length: int | None = None):
        length = self.lengths[seq_id] if length is None else length
        parts_k = []
        parts_v = []
        remaining = length
        for block_id in self.block_tables[seq_id]:
            take = min(self.block_size, remaining)
            parts_k.append(self.k[layer_idx, block_id, :, :take, :])
            parts_v.append(self.v[layer_idx, block_id, :, :take, :])
            remaining -= take
            if remaining <= 0:
                break
        return torch.cat(parts_k, dim=1)[None, :, :, :], torch.cat(parts_v, dim=1)[None, :, :, :]

    def free_sequence(self, seq_id: str) -> None:
        self.free.extend(self.block_tables.pop(seq_id))
        self.lengths.pop(seq_id)

    def waste_fraction(self) -> float:
        allocated_slots = sum(len(t) * self.block_size for t in self.block_tables.values())
        used_slots = sum(self.lengths.values())
        if allocated_slots == 0:
            return 0.0
        return (allocated_slots - used_slots) / allocated_slots


def paged_gqa_decode_attention(
    q: torch.Tensor,
    cache: PagedKVCache,
    layer_idx: int,
    seq_ids: list[str],
    n_rep: int,
) -> torch.Tensor:
    outs = []
    for batch_idx, seq_id in enumerate(seq_ids):
        k, v = cache.gather(layer_idx, seq_id)
        k = repeat_kv(k, n_rep)
        v = repeat_kv(v, n_rep)
        y = F.scaled_dot_product_attention(
            q[batch_idx : batch_idx + 1],
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )
        outs.append(y)
    return torch.cat(outs, dim=0)


def save_paged_plot(cache: PagedKVCache, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seq_ids = list(cache.block_tables.keys())
    max_blocks = max(len(cache.block_tables[s]) for s in seq_ids)
    table = np.full((len(seq_ids), max_blocks), -1, dtype=np.int32)
    for row, seq_id in enumerate(seq_ids):
        blocks = cache.block_tables[seq_id]
        table[row, : len(blocks)] = blocks
    fig, ax = plt.subplots(figsize=(8, 3))
    im = ax.imshow(table, aspect="auto", interpolation="nearest")
    ax.set_yticks(range(len(seq_ids)), seq_ids)
    ax.set_xlabel("logical block index")
    ax.set_title("Block table: logical blocks -> physical KV blocks")
    for row in range(table.shape[0]):
        for col in range(table.shape[1]):
            val = table[row, col]
            if val >= 0:
                ax.text(col, row, str(val), ha="center", va="center", color="white")
    fig.colorbar(im, ax=ax, label="physical block id")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_paged_demo(args: argparse.Namespace) -> None:
    device = get_device(args.device)
    torch.manual_seed(args.seed)
    n_layers = 1
    n_heads = 8
    n_kv_heads = 2
    head_dim = 32
    n_rep = n_heads // n_kv_heads
    seq_ids = ["seqA", "seqB", "seqC"]
    lengths = [47, 63, 79]

    cache = PagedKVCache(
        n_layers=n_layers,
        n_blocks=32,
        block_size=args.block_size,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        device=device,
        dtype=torch.float32,
    )
    refs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for seq_id, length in zip(seq_ids, lengths):
        cache.start_sequence(seq_id)
        k = torch.randn(1, n_kv_heads, length, head_dim, device=device)
        v = torch.randn(1, n_kv_heads, length, head_dim, device=device)
        cache.write_many(0, seq_id, 0, k, v)
        refs[seq_id] = (k, v)

    q = torch.randn(len(seq_ids), n_heads, 1, head_dim, device=device)
    paged = paged_gqa_decode_attention(q, cache, 0, seq_ids, n_rep)

    contiguous_outs = []
    for i, seq_id in enumerate(seq_ids):
        k, v = refs[seq_id]
        k = repeat_kv(k, n_rep)
        v = repeat_kv(v, n_rep)
        contiguous_outs.append(
            F.scaled_dot_product_attention(
                q[i : i + 1], k, v, dropout_p=0.0, is_causal=False
            )
        )
    contiguous = torch.cat(contiguous_outs, dim=0)
    max_diff = (paged - contiguous).abs().max().item()
    print(f"paged attention max diff vs contiguous: {max_diff:.3e}")
    print(f"waste fraction before free: {100 * cache.waste_fraction():.2f}%")
    print("block tables:", cache.block_tables)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = out_dir / "paged_block_table.png"
    save_paged_plot(cache, plot_path)
    print(f"saved paged block table plot: {plot_path}")

    cache.free_sequence("seqB")
    print(f"free blocks after freeing seqB: {len(cache.free)}")


# -----------------------------
# Self-test and CLI commands
# -----------------------------


def run_self_test(args: argparse.Namespace) -> None:
    device = get_device(args.device)
    torch.manual_seed(args.seed)
    vocab_size = 512
    cfg = ModelConfig(
        vocab_size=vocab_size,
        d_model=64,
        n_layer=2,
        n_heads=4,
        n_kv_heads=2,
        d_ff=128,
        block_size=96,
        dropout=0.0,
    )
    model = TinyGPT(cfg).to(device).eval()
    prompt = torch.randint(0, vocab_size, (1, 24), device=device)
    no_cache = generate_no_cache(model, prompt, 24)
    cached = generate_with_cache(model, prompt, 24)
    if not torch.equal(no_cache, cached):
        mismatch = (no_cache != cached).nonzero()[0].tolist()
        raise AssertionError(f"KV cache generation mismatch at {mismatch}")
    print("PASS: KV cache greedy tokens match no-cache greedy tokens")

    sizes = []
    for n_kv in [4, 2, 1]:
        c = ModelConfig(
            vocab_size=vocab_size,
            d_model=64,
            n_layer=2,
            n_heads=4,
            n_kv_heads=n_kv,
            d_ff=128,
            block_size=96,
        )
        m = TinyGPT(c).to(device)
        sizes.append(m.new_cache(batch=1, max_seq=96).bytes())
    if not (sizes[0] > sizes[1] > sizes[2]):
        raise AssertionError(f"cache sizes did not decrease with n_kv_heads: {sizes}")
    print("PASS: GQA/MQA cache memory scales with n_kv_heads", sizes)

    train_cfg = ModelConfig(
        vocab_size=vocab_size,
        d_model=64,
        n_layer=2,
        n_heads=4,
        n_kv_heads=2,
        d_ff=128,
        block_size=64,
        moe_layer=1,
        n_experts=4,
    )
    train_model = TinyGPT(train_cfg).to(device)
    opt = make_optimizer(train_model, lr=1e-3)
    x = torch.randint(0, vocab_size, (4, 32), device=device)
    y = torch.randint(0, vocab_size, (4, 32), device=device)
    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device != "cpu"):
        _, loss, aux = train_model(x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    if aux is None:
        raise AssertionError("MoE aux loss was not produced")
    print(
        "PASS: bf16/no-scaler train step and MoE backward, "
        f"loss={float(loss.detach()):.4f}"
    )

    paged_args = argparse.Namespace(
        device=device,
        seed=args.seed,
        block_size=16,
        out_dir=args.out_dir,
    )
    run_paged_demo(paged_args)

    if args.compile_smoke and device == "xpu":
        print("SKIP: torch.compile smoke on XPU")
    elif args.compile_smoke:
        compile_model = TinyGPT(cfg).to(device).eval()
        sample = torch.randint(0, vocab_size, (2, 32), device=device)
        with torch.no_grad():
            eager_logits, eager_loss, _ = compile_model(sample, sample)
        compiled = torch.compile(compile_model, mode="reduce-overhead")
        with torch.no_grad():
            compiled_logits, compiled_loss, _ = compiled(sample, sample)
        diff = (eager_logits - compiled_logits).abs().max().item()
        print(f"PASS: torch.compile smoke, loss={float(compiled_loss):.4f}, max_diff={diff:.3e}")


def run_kv_bench(args: argparse.Namespace) -> None:
    device = get_device(args.device)
    data = TinyStoriesData(args.data_dir)
    cfg = preset_config(args.preset, data.vocab_size, args.block_size)
    cfg.n_kv_heads = args.n_kv_heads
    cfg.dropout = 0.0
    model = TinyGPT(cfg).to(device).eval()
    prompt = torch.randint(0, data.vocab_size, (1, args.prompt_len), device=device)
    no_cache, cache = benchmark_decode(
        model,
        prompt,
        max_new_tokens=args.new_tokens,
        repeats=args.repeats,
        device=device,
    )
    print(f"model: {count_parameters(model) / 1e6:.2f}M params")
    print(f"prompt_len: {args.prompt_len}, new_tokens: {args.new_tokens}, repeats: {args.repeats}")
    print(f"no-cache decode tok/s: {no_cache:.1f}")
    print(f"cache decode tok/s:    {cache:.1f}")
    print(f"speedup:               {cache / no_cache:.2f}x")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline TinyStories GPT lab")
    sub = parser.add_subparsers(dest="command", required=True)

    common_data = argparse.ArgumentParser(add_help=False)
    common_data.add_argument("--data-dir", default="data/tinystories")
    common_data.add_argument("--device", default="auto")

    p = sub.add_parser("self-test")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--out-dir", default="outputs/flight_lab")
    p.add_argument("--compile-smoke", action="store_true", help="skipped on XPU")
    p.set_defaults(func=run_self_test)

    p = sub.add_parser("train", parents=[common_data])
    p.add_argument("--preset", choices=["quick", "flight", "timing"], default="flight")
    p.add_argument("--block-size", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument("--eval-interval", type=int, default=200)
    p.add_argument("--eval-iters", type=int, default=20)
    p.add_argument("--log-interval", type=int, default=20)
    p.add_argument("--warmup-iters", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--n-kv-heads", type=int, default=2)
    p.add_argument("--moe-layer", type=int, default=None)
    p.add_argument("--n-experts", type=int, default=4)
    p.add_argument("--compile", action="store_true", help="CUDA/H200 only in this lab; skipped on XPU")
    p.add_argument("--compile-mode", default="reduce-overhead")
    p.add_argument("--out-dir", default="outputs/flight_lab")
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--generate-tokens", type=int, default=80)
    p.set_defaults(func=run_train)

    p = sub.add_parser("kv-bench", parents=[common_data])
    p.add_argument("--preset", choices=["quick", "flight", "timing"], default="flight")
    p.add_argument("--block-size", type=int, default=None)
    p.add_argument("--n-kv-heads", type=int, default=2)
    p.add_argument("--prompt-len", type=int, default=128)
    p.add_argument("--new-tokens", type=int, default=96)
    p.add_argument("--repeats", type=int, default=2)
    p.set_defaults(func=run_kv_bench)

    p = sub.add_parser("moe-viz", parents=[common_data])
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--n-experts", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--out-dir", default="outputs/flight_lab")
    p.set_defaults(func=run_moe_viz)

    p = sub.add_parser("paged-demo")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--out-dir", default="outputs/flight_lab")
    p.set_defaults(func=run_paged_demo)

    return parser


def main() -> None:
    torch.set_float32_matmul_precision("high")
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
