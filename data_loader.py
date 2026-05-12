import sys, torch
print("python:", sys.executable)
print("torch:", torch.__version__)
print("xpu available:", torch.xpu.is_available() if hasattr(torch, "xpu") else False)
if hasattr(torch, "xpu") and torch.xpu.is_available():
    print("device:", torch.xpu.get_device_name(0))
    p = torch.xpu.get_device_properties(0)
    print("vram_gb:", round(p.total_memory / 1e9, 2))
    print("props:", p)
    # bf16/fp16 sanity
    print("bf16 ok:", torch.randn(2,2, dtype=torch.bfloat16, device="xpu").sum().item())
    print("fp16 ok:", torch.randn(2,2, dtype=torch.float16,  device="xpu").sum().item())