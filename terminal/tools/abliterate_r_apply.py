"""tools/abliterate_r_apply.py - apply an abliteration direction r (from abliterate_r_extract.py) to a GGUF.

    python tools/abliterate_r_apply.py MODEL.gguf r.json --out-dir DIR [--alpha X | --per-tensor-alpha] [--strength 1.0]
                                       [--experts] [--include REGEX] [--dry-run]

W' = W - alpha * strength * r (r^T W) on every matching residual-writer tensor, re-encoded in its own type, so every
offset and size stays the same. Only types the tools can quantize are written (F32, F16, BF16, Q8_0, Q4_0/1, Q5_0/1);
other types (TQ2_T experts, K-quants, IQ types) are reported and left unchanged.

Output: DIR gets every part of the split set under the same file name. A part with no changed tensor is hard-linked
(a plain copy if the filesystem refuses), so an untouched 50 GB expert part costs no disk. The source is never written.
Routed experts stored Hadamard-rotated (prism.hadamard, input axis) need no change to r: the projection is on the output axis.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import time

import numpy as np

import strata_quants
from abliterate_common import RESIDUAL_WRITERS, select, split_parts, tensors

WRITABLE = {"F32", "F16", "BF16", "Q8_0", "Q4_0", "Q4_1", "Q5_0", "Q5_1"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("rjson")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--alpha", type=float, help="override alpha (default: the median in r.json)")
    ap.add_argument("--per-tensor-alpha", action="store_true", help="use each tensor's own alpha from r.json where present")
    ap.add_argument("--strength", type=float, default=1.0, help="multiplier on alpha")
    ap.add_argument("--include", default=RESIDUAL_WRITERS)
    ap.add_argument("--experts", action="store_true", help="also edit ffn_down_exps (only if its type is writable)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    R = json.loads(pathlib.Path(a.rjson).read_text())
    r = np.asarray(R["r"], np.float32)
    r /= np.linalg.norm(r)
    base = a.alpha if a.alpha is not None else R["alpha_median"]
    per = R.get("tensors", {})

    src_parts = split_parts(a.model)
    out_dir = pathlib.Path(a.out_dir)
    if out_dir.resolve() == src_parts[0].parent.resolve():
        raise SystemExit("--out-dir must differ from the model's folder")
    T = tensors(a.model)
    plan, skipped = [], []
    for n in select(T, a.include, a.experts):
        t = T[n]
        if t.rows != r.size:
            skipped.append((n, f"output dim {t.rows} != r {r.size}"))
        elif t.type not in WRITABLE:
            skipped.append((n, f"type {t.type} not writable"))
        else:
            al = per[n]["alpha"] if a.per_tensor_alpha and n in per else base
            plan.append((n, al * a.strength))
    print(f"{len(plan)} tensors to edit (alpha {base:.4f} x strength {a.strength}"
          f"{', per-tensor where known' if a.per_tensor_alpha else ''}); {len(skipped)} skipped")
    for n, why in skipped:
        print(f"  skip {n}: {why}")
    if a.dry_run or not plan:
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    touched = {T[n].part for n, _ in plan}
    dest = {}
    for p in src_parts:
        d = out_dir / p.name
        if d.exists():
            raise SystemExit(f"{d} exists; remove it or pick another --out-dir")
        if p in touched:
            print(f"copy {p.name}")
            shutil.copyfile(p, d)
        else:
            try:
                os.link(p, d); print(f"link {p.name}")
            except OSError:
                print(f"copy {p.name} (hard link refused)"); shutil.copyfile(p, d)
        dest[p] = d

    t0 = time.time()
    rr = r.astype(np.float64)
    for i, (n, al) in enumerate(plan):
        t = T[n]
        out = bytearray()
        rel = 0.0
        for e in range(t.n_mat):                     # one matrix at a time keeps expert tensors in memory bounds
            W = t.read(e, 1)[0].astype(np.float64)
            dW = -al * np.outer(rr, rr @ W)
            Wn = (W + dW).astype(np.float32)
            rel = max(rel, float(np.linalg.norm(dW) / (np.linalg.norm(W) or 1)))
            out += strata_quants.quantize(Wn, t.type).tobytes()
        if len(out) != t.mat_bytes * t.n_mat:
            raise SystemExit(f"{n}: re-encoded size {len(out)} != {t.mat_bytes * t.n_mat}")
        with open(dest[t.part], "r+b") as f:
            f.seek(t.offset); f.write(out)
        print(f"  [{i + 1}/{len(plan)}] {n:36} {t.type:5} alpha {al:.4f} |dW|/|W| {rel:.4f}")
    print(f"done in {time.time() - t0:.0f}s -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
