from gpt.pretrained import load_model, make_draft
from gpt.speculative import speculative_generate


def spec(target, draft, ids, max_new, gamma):
    stats = {}
    toks = list(speculative_generate(target, draft, ids, max_new=max_new,
                                     gamma=gamma, temperature=0.0, stats=stats))
    return toks, stats


def main():
    target, tok, cfg = load_model()
    prompt = "The mitochondria is"
    ids = tok.encode(prompt).ids
    max_new, gamma = 80, 4

    baseline = list(target.generate(ids, max_tokens=max_new, temperature=0.0, use_kv_cache=True))

    # Draft 1: the target itself -> never rejects (pure correctness check).
    self_toks, self_stats = spec(target, target, ids, max_new, gamma)
    assert self_toks == baseline, "self-draft speculative diverged!"
    print(f"self-draft   : matches greedy target exactly. "
          f"{self_stats['emitted']} tokens in {self_stats['rounds']} target passes.")

    # Draft 2: a weaker 1-layer draft -> real rejections, still exact output.
    draft = make_draft(target, cfg, n_layer=1)
    weak_toks, weak_stats = spec(target, draft, ids, max_new, gamma)
    assert weak_toks == baseline, "weak-draft speculative diverged!"
    eff = weak_stats["emitted"] / weak_stats["rounds"]
    print(f"weak 1-layer : matches greedy target exactly. "
          f"{weak_stats['emitted']} tokens in {weak_stats['rounds']} target passes "
          f"({eff:.2f} tokens/pass, {weak_stats['accepted']} draft tokens accepted).")

    print(f"\nAll decoders agree. Output:\n{prompt}{tok.decode(baseline)}")


if __name__ == "__main__":
    main()
