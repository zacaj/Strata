#!/bin/sh
# Compile the engine without a GPU present: no hardware checks, no model choice, no model download.
# Runs setup.py's own build_engine() / build_engine_hip() with the card named here instead of a detected one, so the
# result (engine/ + its BUILD.json stamp) is exactly what setup would build, and a later ./setup.sh reuses it.
#
# usage: ./build.sh ARCH[,ARCH...] [--vision none|cpu|gpu] [--cuda 12|13]
#   AMD:    gfx1100 (RX 7900), gfx1201 (RX 9070), ...  ROCm: /opt/rocm 7+ if present, else AMD's wheels into .venv
#   NVIDIA: 75 (RTX 20), 86 (RTX 30), 89 (RTX 40), 120 (RTX 50), ...  needs nvcc (PATH, /usr/local/cuda*, STRATA_NVCC)
cd "$(dirname "$0")" || exit 1
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv || { echo "could not create .venv"; exit 1; }
fi
if ! .venv/bin/python -c 'import cmake, ninja' 2>/dev/null; then
  .venv/bin/python -m pip install -q -r requirements.txt || exit 1
fi
exec .venv/bin/python - "$@" <<'EOF'
import argparse
import setup

ap = argparse.ArgumentParser(prog="build.sh", description="compile the Strata engine for the given GPU arch(s)")
ap.add_argument("arch", help="gfx1100 (AMD) or 89 (NVIDIA compute capability, no dot); several: comma-separated")
ap.add_argument("--vision", choices=["none", "cpu", "gpu"], default="none",
                help="also build the image encoder (AMD: cpu only)")
ap.add_argument("--cuda", choices=["12", "13"], help="NVIDIA toolkit (default: 12 below sm_75, else 13, as setup picks)")
a = ap.parse_args()

archs = sorted({x.strip() for x in a.arch.replace(";", ",").split(",") if x.strip()})
hip = any(x.startswith("gfx") for x in archs)
if hip and not all(x.startswith("gfx") for x in archs):
    ap.error("mixing AMD (gfx...) and NVIDIA archs in one engine is not possible")
llama = setup.get_llama_cpp()
if hip:
    if a.vision == "gpu":
        ap.error("the AMD backend has no GPU image encoder: use --vision cpu")
    eng = setup.build_engine_hip({"arch": archs[0], "archs": archs}, llama, a.vision)
else:
    nums = sorted(int(x) for x in archs)
    eng = setup.build_engine({"arch": nums[0], "archs": nums}, a.vision, True, llama,
                             toolkit=int(a.cuda) if a.cuda else None)
print(f"engine: {eng}")
EOF
