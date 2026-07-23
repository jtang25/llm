import time

from gpt.pretrained import load_model


def collect(model, prompt_ids, **kw):
    return list(model.generate(prompt_ids, **kw))


def main():
    model, tok, _ = load_model()
    prompt = "The mitochondria is"
    ids = tok.encode(prompt).ids
    kw = dict(max_tokens=80, temperature=0.0, top_k=None)  # greedy -> deterministic

    t0 = time.perf_counter()
    no_cache = collect(model, ids, use_kv_cache=False, **kw)
    t1 = time.perf_counter()
    cached = collect(model, ids, use_kv_cache=True, **kw)
    t2 = time.perf_counter()

    assert no_cache == cached, "KV cache changed the output!"
    print("KV cache matches no-cache output exactly (greedy).")
    print(f"no-cache: {len(no_cache)/(t1-t0):.1f} tok/s   cache: {len(cached)/(t2-t1):.1f} tok/s")
    print(f"\n{prompt}{tok.decode(cached)}")


if __name__ == "__main__":
    main()
