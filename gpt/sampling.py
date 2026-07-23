import torch
import torch.nn.functional as F


def _top_k_filter(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    # Keep only the top_k logits; set the rest to -inf so softmax zeros them.
    k = min(top_k, logits.size(-1))
    kth = torch.topk(logits, k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < kth, float("-inf"))


def sample_token(logits, temperature=1.0, top_k=None, generator=None, device="cpu"):
    """logits: (1, vocab). Returns a (1, 1) long tensor with the next token."""
    if temperature == 0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    if top_k is not None:
        logits = _top_k_filter(logits, top_k)
    probs = F.softmax(logits / temperature, dim=-1).cpu()
    nxt = torch.multinomial(probs, num_samples=1, generator=generator)
    return nxt.to(device)


def dist_from_logits(logits, temperature=1.0, top_k=None) -> torch.Tensor:
    """logits: (vocab,). Returns the full probability distribution (vocab,) on CPU.

    Speculative decoding needs the whole distribution (not just a sample) to
    run its accept/reject rule, so this is separate from sample_token.
    """
    logits = logits.float().cpu()
    if top_k is not None:
        logits = _top_k_filter(logits, top_k)
    return F.softmax(logits / max(temperature, 1e-8), dim=-1)
