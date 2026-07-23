import time

from gpt.pretrained import load_model
from gpt.paged_attention import generate_paged


def main():
    model, tok, _ = load_model()
    prompt = "The mitochondria is"
    ids = tok.encode(prompt).ids

    # Greedy so both paths are deterministic and must agree token-for-token.
    baseline = list(model.generate(ids, max_tokens=80, temperature=0.0, use_kv_cache=True))

    t0 = time.perf_counter()
    paged = list(generate_paged(model, ids, max_tokens=80, temperature=0.0, block_size=16))
    t1 = time.perf_counter()

    assert baseline == paged, "Paged output diverged from the contiguous cache!"
    print("Paged KV cache matches the contiguous cache exactly (greedy).")
    print(f"paged: {len(paged)/(t1-t0):.1f} tok/s, block_size=16")
    print(f"\n{prompt}{tok.decode(paged)}")


if __name__ == "__main__":
    main()
