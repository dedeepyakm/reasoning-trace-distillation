#!/usr/bin/env python3
"""
Run this FIRST. It checks your setup and tells you in plain language
whether you're ready to generate traces.

    python check_setup.py
"""
import sys, platform

print("=" * 62)
print("SETUP CHECK")
print("=" * 62)

print(f"\nPython version : {sys.version.split()[0]}")
print(f"Platform       : {platform.system()} {platform.release()}")
if sys.version_info < (3, 9):
    print("  ! Python 3.9+ recommended. Install a newer version.")

# ---- torch -----------------------------------------------------------------
try:
    import torch
    print(f"\nPyTorch        : {torch.__version__}")
except ImportError:
    print("\n[X] PyTorch NOT installed.")
    print("    Fix: pip install torch --index-url https://download.pytorch.org/whl/cu128")
    sys.exit(1)

# ---- GPU -------------------------------------------------------------------
if not torch.cuda.is_available():
    print("[X] No GPU detected by PyTorch.")
    print("    Most likely cause: PyTorch was installed WITHOUT CUDA support.")
    print("    Fix: pip uninstall torch -y")
    print("         pip install torch --index-url https://download.pytorch.org/whl/cu128")
    print("    You can still run tiny models on CPU, but it will be slow.")
else:
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    cap = torch.cuda.get_device_capability(0)
    print(f"[ok] GPU        : {name}")
    print(f"     VRAM       : {vram:.1f} GB")
    print(f"     Compute cap: sm_{cap[0]}{cap[1]}")

    # Actually try a GPU operation -- detection alone can lie on new hardware.
    try:
        x = torch.randn(1000, 1000, device="cuda")
        _ = (x @ x).sum().item()
        print("[ok] GPU math works.")
    except Exception as e:
        print(f"[X] GPU detected but computation FAILED: {type(e).__name__}")
        print("    On RTX 50-series (Blackwell) this means your PyTorch build")
        print("    is too old. Fix:")
        print("      pip uninstall torch -y")
        print("      pip install torch --index-url https://download.pytorch.org/whl/cu128")

    print(f"\n     What fits in {vram:.0f} GB:")
    for label, need in [("Qwen2.5-1.5B (fp16)", 3.5),
                        ("Qwen2.5-7B  (4-bit)", 6.0),
                        ("Qwen2.5-7B  (fp16)", 16.0),
                        ("gpt-oss-20b (MXFP4)", 14.0)]:
        print(f"       {'yes' if vram >= need else 'no ':>4}  {label}  (~{need:.0f} GB)")

# ---- other packages --------------------------------------------------------
print()
for mod, pipname in [("transformers", "transformers"), ("datasets", "datasets"),
                     ("accelerate", "accelerate"), ("scipy", "scipy"),
                     ("pandas", "pandas"), ("bitsandbytes", "bitsandbytes")]:
    try:
        m = __import__(mod)
        v = getattr(m, "__version__", "?")
        print(f"[ok] {mod:<14} {v}")
    except ImportError:
        print(f"[X] {mod:<14} missing   ->  pip install {pipname}")

print("\n" + "=" * 62)
print("If everything above says [ok], run:")
print("  python generate_traces.py --model Qwen/Qwen2.5-1.5B-Instruct \\")
print("         --dataset math500 --n 3 --k 2 --max-new-tokens 400")
print("=" * 62)
