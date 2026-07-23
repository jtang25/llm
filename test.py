"""
Verifies whether F.scaled_dot_product_attention is dispatching to a fused
kernel on XPU, or falling back to the naive/math decomposition.

How to read the output:
- FUSED path: profiler trace shows a small number of large ops, typically
  something with "sdpa" / "efficient_attention" / "flash" / "onednn" in
  the name, and the op count stays roughly constant regardless of shape.
- NAIVE/MATH fallback: trace shows attention decomposed into many separate
  ops (matmul/bmm, softmax, dropout, add, mul) with the number of ops
  scaling with number of layers, since each layer's attention becomes
  ~4-5 separate kernel launches instead of one fused call.

Run: python verify_sdpa_backend.py
"""

import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity

print(f"[info] torch version: {torch.__version__}")

if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
    raise RuntimeError("XPU not available in this environment.")

device = torch.device("xpu")

# oneDNN's fused SDPA path on XPU landed as default in PyTorch 2.7.
# On anything older, expect a naive fallback regardless of what you do.
major, minor = (int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
if (major, minor) < (2, 7):
    print(f"[warn] torch {torch.__version__} predates PyTorch 2.7, where "
          f"oneDNN became the default fused XPU SDPA backend. You are very "
          f"likely on the naive fallback. Consider upgrading torch.")

# Representative shapes matching the benchmark's default config.
B, H, T, D = 4, 8, 512, 64  # batch, heads, seq_len, head_dim
q = torch.randn(B, H, T, D, device=device, dtype=torch.bfloat16)
k = torch.randn(B, H, T, D, device=device, dtype=torch.bfloat16)
v = torch.randn(B, H, T, D, device=device, dtype=torch.bfloat16)

# Warmup (first call often includes JIT/kernel-selection overhead)
for _ in range(3):
    _ = F.scaled_dot_product_attention(q, k, v, is_causal=True)
torch.xpu.synchronize()

# Try explicitly requesting each backend PyTorch knows about, where the
# API is available. Older torch versions may not expose sdpa_kernel with
# XPU backend enums, in which case this section is skipped gracefully.
try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
    print("\n[info] torch.nn.attention.sdpa_kernel is available, will try "
          "forcing each backend explicitly.\n")
    for backend in [SDPBackend.MATH, SDPBackend.FLASH_ATTENTION,
                    SDPBackend.EFFICIENT_ATTENTION]:
        try:
            with sdpa_kernel(backend):
                out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                torch.xpu.synchronize()
            print(f"  {backend} : supported on this build")
        except Exception as e:
            print(f"  {backend} : NOT usable ({type(e).__name__}: {e})")
except ImportError:
    print("\n[warn] torch.nn.attention.sdpa_kernel not available in this "
          "torch version, skipping explicit backend probing.")

# The actual diagnostic: profile a real call with no backend forced (i.e.
# whatever the default dispatcher picks) and inspect the op-level trace.
print("\n[info] profiling default (auto-dispatched) SDPA call...\n")
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU]) as prof:
    for _ in range(5):
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.xpu.synchronize()

print(prof.key_averages().table(sort_by="xpu_time_total", row_limit=20))

print(
    "\n[how to interpret] If the top rows above are dominated by a single "
    "op with a name containing 'scaled_dot_product', 'efficient_attention', "
    "'flash', or 'onednn', SDPA is fused. If instead you see a long list "
    "of separate 'aten::matmul' / 'aten::bmm' / 'aten::softmax' entries "
    "each taking meaningful time, that's the naive math fallback."
)