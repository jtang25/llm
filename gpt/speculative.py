"""Speculative decoding (Leviathan et al. 2023 / Chen et al. 2023).

A cheap `draft` model proposes gamma tokens autoregressively. The expensive
`target` model then scores all gamma proposals in ONE forward pass (a pass over
gamma+1 positions costs ~the same as over 1, since weights are loaded once).
A rejection rule accepts the longest correct prefix and resamples the first
mismatch, so the emitted tokens are distributed EXACTLY as target-only sampling.

This reference version is cache-free: each model call recomputes the whole
sequence. That keeps the accept/reject logic obvious; a production engine keeps
KV caches and rolls them back on rejection.

Pass a `stats` dict to collect efficiency numbers:
  stats["rounds"]   -> number of target forward passes
  stats["accepted"] -> draft tokens accepted (excludes rejection/bonus tokens)
  stats["emitted"]  -> total tokens produced
"""
import torch

from .sampling import dist_from_logits


def _logits_full(model, seq, device):
    """Logits for every position of `seq` (list[int]). Returns (len, vocab)."""
    ids = torch.tensor([seq], dtype=torch.long, device=device)
    logits, _, _ = model(ids, kv_caches=None, start_pos=0)
    return logits[0]


@torch.inference_mode()
def speculative_generate(target, draft, tokens, max_new, gamma=4,
                         temperature=1.0, top_k=None, seed=42, stats=None):
    """Yields target-distributed tokens one at a time."""
    device = next(target.parameters()).device
    greedy = temperature == 0
    rng = None if greedy else torch.Generator(device="cpu").manual_seed(seed)
    max_len = target.max_seq_len
    if stats is not None:
        stats.update(rounds=0, accepted=0, emitted=0)

    def _emit(x, accepted):
        if stats is not None:
            stats["emitted"] += 1
            stats["accepted"] += accepted
        return x

    seq = list(tokens)
    produced = 0
    while produced < max_new and len(seq) < max_len:
        if stats is not None:
            stats["rounds"] += 1

        # 1. Draft proposes gamma tokens, recording its distribution q_i each step.
        draft_seq = list(seq)
        proposals, q_dists = [], []
        for _ in range(gamma):
            q = dist_from_logits(_logits_full(draft, draft_seq, device)[-1], temperature, top_k)
            x = int(torch.argmax(q)) if greedy else int(torch.multinomial(q, 1, generator=rng))
            proposals.append(x)
            q_dists.append(q)
            draft_seq.append(x)

        # 2. Target scores the whole [seq + proposals] in one pass. The dist that
        #    should predict proposals[i] is the logits row at base_len-1+i.
        base_len = len(seq)
        target_logits = _logits_full(target, seq + proposals, device)

        # 3. Walk proposals, accepting under the rejection rule.
        rejected = False
        for i in range(gamma):
            p = dist_from_logits(target_logits[base_len - 1 + i], temperature, top_k)
            x = proposals[i]
            if greedy:
                accept = (x == int(torch.argmax(p)))
            else:
                r = torch.rand(1, generator=rng).item()
                accept = r < min(1.0, (p[x] / q_dists[i][x]).item())

            if accept:
                seq.append(x); produced += 1
                yield _emit(x, accepted=True)
                if produced >= max_new or len(seq) >= max_len:
                    return
            else:
                # Reject: greedy -> target argmax; sampling -> renormalized max(0, p-q).
                if greedy:
                    x = int(torch.argmax(p))
                else:
                    resid = torch.clamp(p - q_dists[i], min=0)
                    resid = resid / resid.sum() if resid.sum() > 0 else p
                    x = int(torch.multinomial(resid, 1, generator=rng))
                seq.append(x); produced += 1
                yield _emit(x, accepted=False)
                rejected = True
                break

        # 4. All gamma accepted -> a free "bonus" token from target's last row.
        if not rejected:
            p = dist_from_logits(target_logits[base_len - 1 + gamma], temperature, top_k)
            x = int(torch.argmax(p)) if greedy else int(torch.multinomial(p, 1, generator=rng))
            seq.append(x); produced += 1
            yield _emit(x, accepted=False)
