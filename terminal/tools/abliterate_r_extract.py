"""tools/abliterate_r_extract.py - recover the abliteration direction r from a stock / abliterated GGUF pair.

    python tools/abliterate_r_extract.py STOCK.gguf ABLITERATED.gguf -o swift-v2-r.json [--experts 4] [--include REGEX]

Either file may be any part of a split set, local or an https URL (e.g. a Hugging Face .../resolve/main/... link):
a URL is read with HTTP range requests, only the header and the tensors used, cached under %TEMP%/strata-abliterate-cache. The two sets must have the same tensor names and shapes; the quant types
may differ (each side is dequantized). Per tensor, D = abliterated - stock. Every changed residual-writer tensor adds
D D^T / |D|^2 to a covariance; r is its top eigenvector, so noise from 4-bit rounding averages out across tensors.
Then for each tensor, alpha is the least-squares fit of D ~ -alpha * r (r^T W).

The JSON written holds r, the median alpha and per-tensor stats:
  rel   |D| / |W|                         size of the edit
  cos   |cos| of that tensor's own top direction with r   (~1 = the same direction everywhere)
  fit   share of |D|^2 that alpha * r r^T W explains     (low on 4-bit inputs: the rest is rounding noise)
--experts K also samples the first K experts of each ffn_down_exps (v1-style abliteration touches them).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import pathlib
import time

import numpy as np

from abliterate_common import RESIDUAL_WRITERS, is_url, select, tensors


def top_vec(G: np.ndarray, iters: int = 30) -> np.ndarray:
    v = np.random.default_rng(0).standard_normal(G.shape[0]).astype(np.float64)
    for _ in range(iters):
        v = G @ v
        v /= np.linalg.norm(v)
    return v


def deltas(A_t, B_t, k: int):
    n = A_t.n_mat if A_t.n_mat == 1 else min(k, A_t.n_mat)
    A, B = A_t.read(0, n).astype(np.float64), B_t.read(0, n).astype(np.float64)
    return A, B - A


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stock")
    ap.add_argument("abliterated")
    ap.add_argument("-o", "--out", required=True, help="output .json")
    ap.add_argument("--include", default=RESIDUAL_WRITERS, help="tensor name regex (default: residual writers)")
    ap.add_argument("--experts", type=int, default=0, metavar="K", help="also use the first K experts of ffn_down_exps")
    a = ap.parse_args()

    S, T = tensors(a.stock), tensors(a.abliterated)
    names = [n for n in select(S, a.include, a.experts > 0) if n in T]
    print(f"{len(names)} candidate tensors")
    t0 = time.time()
    remote = [(X[n], 1 if X[n].n_mat == 1 else min(a.experts, X[n].n_mat)) for X in (S, T) for n in names
              if is_url(X[n].part) and X[n].readable]
    if remote:
        mb = sum(t.mat_bytes * k for t, k in remote) / 1e6
        print(f"fetching {len(remote)} remote tensors ({mb:,.0f} MB) ...")
        with cf.ThreadPoolExecutor(8) as ex:
            list(ex.map(lambda tk: tk[0].read_raw(0, tk[1]), remote))
        print(f"  fetched in {time.time() - t0:.0f}s")
    C, used, unchanged, skipped, dim = None, [], [], [], None
    for n in names:
        s, t = S[n], T[n]
        if s.shape != t.shape or not (s.readable and t.readable):
            skipped.append(n); continue
        _, D = deltas(s, t, a.experts)
        if not D.any():
            unchanged.append(n); continue
        dim = D.shape[1]
        if C is None:
            C = np.zeros((dim, dim))
        for d in D:
            nn = (d * d).sum()
            if nn:
                C += (d @ d.T) / nn
        used.append(n)
    if not used:
        raise SystemExit("no changed tensors: is this really a stock / abliterated pair?")
    w, V = np.linalg.eigh(C)
    r = V[:, -1]
    print(f"pass 1: {len(used)} changed, {len(unchanged)} unchanged, {len(skipped)} skipped "
          f"(top eigen share {w[-1] / w.sum():.3f}, 2nd/1st {w[-2] / w[-1]:.3f}) {time.time() - t0:.0f}s")

    stats = {}
    for n in used:
        A, D = deltas(S[n], T[n], a.experts)
        num = den = dd = ww = 0.0
        for Ai, Di in zip(A, D):
            pa, pd = r @ Ai, r @ Di                     # r^T W, r^T D
            num += -(pd @ pa); den += pa @ pa
            dd += (Di * Di).sum(); ww += (Ai * Ai).sum()
        alpha = num / den if den else 0.0
        # |D - (-alpha r r^T W)|^2 = |D|^2 - 2 alpha * num + alpha^2 den  ->  explained = alpha^2 den / |D|^2 (at the LS optimum)
        fit = alpha * alpha * den / dd if dd else 0.0
        u = top_vec(sum(Di @ Di.T for Di in D))
        stats[n] = {"type": S[n].type, "alpha": round(alpha, 5), "rel": round(float(np.sqrt(dd / ww)), 5),
                    "cos": round(float(abs(u @ r)), 4), "fit": round(fit, 4)}
        print(f"  {n:36} {S[n].type:7}->{T[n].type:7} alpha {alpha:7.4f}  rel {stats[n]['rel']:.4f}  "
              f"cos {stats[n]['cos']:.3f}  fit {stats[n]['fit']:.3f}")
    alphas = np.array([v["alpha"] for v in stats.values()])
    out = {"dim": int(dim), "alpha_median": float(np.median(alphas)), "alpha_mean": float(alphas.mean()),
           "alpha_std": float(alphas.std()), "stock": a.stock if is_url(a.stock) else str(pathlib.Path(a.stock).resolve()),
           "abliterated": a.abliterated if is_url(a.abliterated) else str(pathlib.Path(a.abliterated).resolve()), "include": a.include, "experts": a.experts,
           "unchanged": unchanged, "skipped": skipped, "tensors": stats, "r": [float(x) for x in r.astype(np.float32)]}
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"alpha median {out['alpha_median']:.4f} (mean {out['alpha_mean']:.4f}, std {out['alpha_std']:.4f}); "
          f"wrote {a.out} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
