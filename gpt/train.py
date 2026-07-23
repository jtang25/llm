import math

import torch
import torch.nn.functional as F

from .config import GPTConfig
from .device import get_device
from .data import Dataset, load_tokenizer, prepare_data
from .layers import MoEFeedForward
from .model import GPT


def get_lr(it, lr, warmup, max_iters):
    if it < warmup:
        return lr * (it + 1) / warmup
    progress = (it - warmup) / max(1, max_iters - warmup)
    return lr * 0.5 * (1.0 + math.cos(math.pi * progress)) * 0.9 + lr * 0.1


def main():
    device = get_device()
    batch_size, accumulation_steps = 8, 4
    learning_rate, warmup_iters, max_iters = 1e-3, 300, 15000
    eval_interval, eval_iters, block_size = 10, 20, 128

    prepare_data()
    tok = load_tokenizer()
    data = Dataset()

    cfg = GPTConfig(vocab_size=tok.get_vocab_size(), max_seq_len=256, dropout=0.0)
    model = GPT(cfg).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    nodecay = [p for n, p in model.named_parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1},
         {"params": nodecay, "weight_decay": 0.0}],
        lr=learning_rate, betas=(0.9, 0.95),
    )
    scaler = torch.amp.GradScaler(device=device)

    for it in range(max_iters):
        for pg in optimizer.param_groups:
            pg["lr"] = get_lr(it, learning_rate, warmup_iters, max_iters)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        accum = 0.0
        for _ in range(accumulation_steps):
            x, y = data.get_batch("train", batch_size, block_size, device)
            with torch.autocast(device_type=device, dtype=torch.float16):
                _, loss, _ = model(x, y)
            loss = loss / accumulation_steps
            scaler.scale(loss).backward()
            accum += loss.item()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if it % eval_interval == 0 or it == max_iters - 1:
            model.eval()
            with torch.no_grad():
                vlosses = []
                for _ in range(eval_iters):
                    x, y = data.get_batch("val", batch_size, block_size, device)
                    with torch.autocast(device_type=device, dtype=torch.float16):
                        _, vl, _ = model(x, y)
                    vlosses.append(vl.item())
            print(f"Iter {it}: train {accum:.4f}, val {sum(vlosses)/len(vlosses):.4f}", flush=True)

    torch.save(model.state_dict(), "gpt_model.pth")


if __name__ == "__main__":
    main()
