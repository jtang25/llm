from pathlib import Path

import torch

from .config import checkpoint_config, GPTConfig
from .data import load_tokenizer, ROOT
from .device import get_device
from .model import GPT

CHECKPOINT = ROOT / "gpt_model.pth"


def load_model(device=None):
    device = device or get_device()
    tok = load_tokenizer()
    cfg = checkpoint_config(tok.get_vocab_size())
    model = GPT(cfg).to(device)
    state = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, tok, cfg


def make_draft(target: GPT, cfg: GPTConfig, n_layer=1, device=None):
    """A cheaper draft that reuses the target's first n_layer layers. Its output
    distribution differs from the target, so it exercises accept/reject, yet
    greedy speculative decoding still matches the target exactly."""
    device = device or next(target.parameters()).device
    dcfg = GPTConfig(**{**cfg.__dict__, "n_layer": n_layer})
    draft = GPT(dcfg).to(device)

    src = target.state_dict()
    dst = draft.state_dict()
    for k in dst:                     # copy every param the draft also has
        if k in src and src[k].shape == dst[k].shape:
            dst[k] = src[k]
    draft.load_state_dict(dst)
    draft.eval()
    return draft
