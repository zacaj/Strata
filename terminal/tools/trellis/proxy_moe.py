"""proxy_moe.py - the routed-expert output error of a layer, rebuilt from real tokens: no Hessian approximation.

proxy_ldl.py scores each matrix with an input second moment H, as if gate, up and down were independent linear maps.
They are not: the expert computes down . (silu(gate x) * up x), and the layer adds the experts weighted by the router.
This tool takes the tokens themselves (STRATA_ACT_DUMP_RAW: the routed ids and the input x of each token), recomputes the
router weights (softmax over all experts, top k, renormalised - src/core/layer.cpp; the recomputed ids are checked
against the dumped ones), and for every routed (token, expert) pair compares the candidate expert's output with the
original BF16 expert's:

    err = sum over pairs  w^2 |out_cand(x) - out_orig(x)|^2,   energy = sum  w^2 |out_orig(x)|^2

(the cross terms between two experts' errors on one token are left out: independent quantization errors).  Reported in
dB (energy / err), for the whole expert and with only gate, only up or only down quantized.  Candidates: the shipped
file's blocks and any swap.bin (build_swap.py: Q8_0 or trellis), so a swap whose KL was measured can be scored as is.

  python tools/trellis/proxy_moe.py --gguf GYRO.gguf --orig ORIG_DIR --raw DUMP/act_L24_raw.bin --layer 24 \
      [--swap ldl=VAR/swap.bin ...] [--experts all|0,5,...] [--max-tokens N]

--encode also encodes each scored expert on the fly (ldl, damping --damp, GPU with --gpu) with an H per matrix built
from calibration tokens (--calib: raw dumps of other text; --pool-dump: the layer's pooled H, STRATA_ACT_DUMP_COV):
  ldl          the pooled H (build_swap.py --mode ldl)
  ldl-r        this expert's tokens, each weighted by its router weight squared
  ldl-s        the same, and SwiGLU's sensitivity: gate rows see a token by mean_r (u_r silu'(g_r))^2, up rows by
               mean_r silu(g_r)^2; down sees the expert's own SwiGLU output
A per-expert H rests on a few hundred tokens, so it is shrunk to the pooled one (normalized to its trace):
(n_eff H_e + kappa H_pool) / (n_eff + kappa), n_eff the effective number of tokens.
"""
from __future__ import annotations

import argparse
import mmap
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import gguf_reader  # noqa: E402
import tqenc  # noqa: E402
from build_swap import bf16  # noqa: E402

N_EXP, N_EMBD, N_FF = 512, 2560, 640


def read_raw(path, max_tokens):
    raw = np.fromfile(path, dtype=np.uint8)
    n_embd, k = (int(v) for v in raw[:8].view(np.int32))
    rec = 4 * k + 4 * n_embd
    nt = (raw.size - 8) // rec
    if max_tokens:
        nt = min(nt, max_tokens)
    body = raw[8:8 + nt * rec].reshape(nt, rec)
    ids = body[:, :4 * k].copy().view(np.int32)
    x = body[:, 4 * k:].copy().view(np.float32)
    return ids, x, k


def route(x, wr, k):
    """softmax over all experts, the k largest (ties by index), renormalised - the engine's rule."""
    lg = x.astype(np.float64) @ wr.T.astype(np.float64)
    lg -= lg.max(axis=1, keepdims=True)
    p = np.exp(lg)
    p /= p.sum(axis=1, keepdims=True)
    order = np.argsort(-p, axis=1, kind="stable")[:, :k]
    w = np.take_along_axis(p, order, axis=1)
    return order.astype(np.int32), (w / w.sum(axis=1, keepdims=True)).astype(np.float64)


def q8_0_decode(raw: np.ndarray) -> np.ndarray:
    b = raw.reshape(-1, 34)
    d = b[:, :2].copy().view(np.float16).astype(np.float32)
    return (b[:, 2:].view(np.int8).astype(np.float32) * d).reshape(-1)


class Source:
    """Decoded expert matrices of one file (rotated basis), by (role, expert)."""

    def __init__(self, path, layer):
        self.g = gguf_reader.GGUFFile(path)
        self.fh = open(path, "rb")
        self.mm = mmap.mmap(self.fh.fileno(), 0, access=mmap.ACCESS_READ)
        self.by = {t.name: t for t in self.g.tensors}
        self.layer = layer

    def get(self, role, e):
        ti = self.by["blk.%d.ffn_%s_exps.weight" % (self.layer, role)]
        n_in, n_out = ti.shape[0], ti.shape[1]
        if ti.type_name == "Q8_0":
            rb = n_in // 32 * 34
            raw = np.frombuffer(self.mm, dtype=np.uint8, count=n_out * rb, offset=self.g.data_start + ti.offset + e * n_out * rb)
            return q8_0_decode(raw).reshape(n_out, n_in)
        t = tqenc.TYPE_BY_NAME[ti.type_name]
        rb = n_in // 128 * tqenc.block_bytes(t)
        raw = np.frombuffer(self.mm, dtype=np.uint8, count=n_out * rb, offset=self.g.data_start + ti.offset + e * n_out * rb)
        return tqenc.decode(raw, t).reshape(n_out, n_in)


def silu(v):
    return v / (1.0 + np.exp(-v))


def dsilu(v):
    s = 1.0 / (1.0 + np.exp(-v))
    return s * (1.0 + v * (1.0 - s))


def shrunk(Xt, a, Hp, kappa):
    """sum a x x^T / sum a, shrunk to Hp (scaled to the same trace) with kappa pseudo-tokens."""
    if Xt.shape[0] == 0 or a.sum() <= 0:
        return Hp
    # fp32 for the matmul (about 2x numpy's fp64 speed): a sum over a few hundred tokens, then mixed in fp64
    X32 = np.asarray(Xt, dtype=np.float32)
    He = ((X32 * a.astype(np.float32)[:, None]).T @ X32).astype(np.float64) / a.sum()
    neff = a.sum() ** 2 / max(float((a ** 2).sum()), 1e-300)
    Hp_s = Hp * (np.trace(He) / max(np.trace(Hp), 1e-300))
    return (neff * He + kappa * Hp_s) / (neff + kappa)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True, help="the shipped file: rotation signs, router, the 'shipped' candidate")
    ap.add_argument("--orig", required=True, help="fetch_orig.py's BF16 experts (the reference)")
    ap.add_argument("--raw", required=True, help="a STRATA_ACT_DUMP_RAW file of this layer")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--swap", action="append", default=[], help="NAME=swap.bin (build_swap.py) - another candidate")
    ap.add_argument("--experts", default="all")
    ap.add_argument("--max-tokens", type=int, default=0)
    ap.add_argument("--parts", default="all,gate only,up only,down only")
    ap.add_argument("--encode", default="", help="ldl,ldl-r,ldl-s (see above)")
    ap.add_argument("--calib", action="append", default=[], help="raw dumps of calibration text (--encode ldl-r / ldl-s)")
    ap.add_argument("--pool-dump", default=None, help="a STRATA_ACT_DUMP_COV directory: the pooled H")
    ap.add_argument("--damp", type=float, default=0.5)
    ap.add_argument("--kappa", type=float, default=256.0)
    ap.add_argument("--gpu", action="store_true")
    a = ap.parse_args()
    l = a.layer
    ship = Source(a.gguf, l)
    g = ship.g
    sw = [int(x) for x in g.metadata["prism.hadamard.sign_widths"]]
    sv = np.array(g.metadata["prism.hadamard.sign_values"], dtype=np.float32)
    signs, at = {}, 0
    for w in sw:
        signs[w] = sv[at:at + w]
        at += w
    ti = ship.by["blk.%d.ffn_gate_inp.weight" % l]
    wr = np.frombuffer(ship.mm, dtype=np.float32, count=N_EXP * N_EMBD, offset=g.data_start + ti.offset).reshape(N_EXP, N_EMBD)

    ids_d, x, k = read_raw(a.raw, a.max_tokens)
    ids, w = route(x, wr, k)
    same = np.mean([set(ids[t]) == set(ids_d[t]) for t in range(len(ids))])
    print("tokens %d, k %d; recomputed routing = dumped routing for %.2f%% of tokens" % (len(ids), k, 100 * same))
    if same < 0.99:
        print("  (the dumped ids are used; the weights come from the recomputed softmax over the dumped set)")
    # the weights of the dumped ids (identical sets in practice; this keeps the engine's choice if they differ)
    lg = x.astype(np.float64) @ wr.T.astype(np.float64)
    lg -= lg.max(axis=1, keepdims=True)
    p = np.exp(lg)
    p /= p.sum(axis=1, keepdims=True)
    wd = np.take_along_axis(p, ids_d, axis=1)
    wd /= wd.sum(axis=1, keepdims=True)
    xr = tqenc.rotate(np.ascontiguousarray(x), signs[N_EMBD]).astype(np.float64)

    cands = {"shipped": ship}
    for s in a.swap:
        name, path = s.split("=", 1)
        cands[name] = Source(path, l)
    encs = [v for v in a.encode.split(",") if v]
    for v in encs:
        cands[v] = None   # encoded per expert below
    if encs:
        import blockldl
        from build_swap import load_hessian
        Hp = {kd: load_hessian(a.pool_dump, l, kd) for kd in ("gu", "dn")}
        cx, cids = [], []
        for path in a.calib:
            i_, x_, _ = read_raw(path, 0)
            cx.append(x_)
            cids.append(i_)
        cx = np.concatenate(cx) if cx else np.zeros((0, N_EMBD), np.float32)
        cids = np.concatenate(cids) if cids else np.zeros((0, k), np.int32)
        if len(cx):
            lgc = cx.astype(np.float64) @ wr.T.astype(np.float64)
            lgc -= lgc.max(axis=1, keepdims=True)
            pc = np.exp(lgc)
            pc /= pc.sum(axis=1, keepdims=True)
            cw = np.take_along_axis(pc, cids, axis=1)
            cw /= cw.sum(axis=1, keepdims=True)
            cxr = tqenc.rotate(np.ascontiguousarray(cx), signs[N_EMBD]).astype(np.float64)
        print("calibration tokens %d" % len(cx))
        t_gu = tqenc.TYPE_BY_NAME[ship.by["blk.%d.ffn_gate_exps.weight" % l].type_name]
        t_dn = tqenc.TYPE_BY_NAME[ship.by["blk.%d.ffn_down_exps.weight" % l].type_name]

        def ldl_dec(Wm, t, H):
            fb = blockldl.feedback(H, a.damp, gpu=a.gpu)
            _, dec = blockldl.encode_ldl(Wm.astype(np.float32), t, fb, gpu=a.gpu)
            return dec.astype(np.float64)

        def encode_expert(v, e, G, U, D):
            if v == "ldl":
                return ldl_dec(G, t_gu, Hp["gu"]), ldl_dec(U, t_gu, Hp["gu"]), ldl_dec(D, t_dn, Hp["dn"])
            tok, slot = np.nonzero(cids == e)
            X = cxr[tok]
            w2 = cw[tok, slot] ** 2
            g, u = X @ G.T, X @ U.T
            ff = silu(g) * u
            ffr = tqenc.rotate(np.ascontiguousarray(ff.astype(np.float32)), signs[N_FF]).astype(np.float64)
            if v == "ldl-r":
                ag = au = w2
            else:   # ldl-s
                ag = w2 * np.mean((u * dsilu(g)) ** 2, axis=1)
                au = w2 * np.mean(silu(g) ** 2, axis=1)
            Hg = shrunk(X, ag, Hp["gu"], a.kappa)
            Hu = shrunk(X, au, Hp["gu"], a.kappa)
            Hd = shrunk(ffr, w2, Hp["dn"], a.kappa)
            return ldl_dec(G, t_gu, Hg), ldl_dec(U, t_gu, Hu), ldl_dec(D, t_dn, Hd)
    gu_raw = np.memmap(pathlib.Path(a.orig) / ("L%d_gate_up.bf16" % l), dtype=np.uint16, mode="r").reshape(N_EXP, 2 * N_FF, N_EMBD)
    dn_raw = np.memmap(pathlib.Path(a.orig) / ("L%d_down.bf16" % l), dtype=np.uint16, mode="r").reshape(N_EXP, N_EMBD, N_FF)
    experts = range(N_EXP) if a.experts == "all" else [int(v) for v in a.experts.split(",")]
    parts = a.parts.split(",")
    err = {(c, p): 0.0 for c in cands for p in parts}
    energy = 0.0
    used = 0
    for e in experts:
        tok, slot = np.nonzero(ids_d == e)
        if tok.size == 0:
            continue
        used += 1
        xt = xr[tok]
        w2 = wd[tok, slot] ** 2
        G = tqenc.rotate(bf16(np.asarray(gu_raw[e][:N_FF])), signs[N_EMBD]).astype(np.float64)
        U = tqenc.rotate(bf16(np.asarray(gu_raw[e][N_FF:])), signs[N_EMBD]).astype(np.float64)
        D = tqenc.rotate(bf16(np.asarray(dn_raw[e])), signs[N_FF]).astype(np.float64)

        def out(Gm, Um, Dm):
            ff = silu(xt @ Gm.T) * (xt @ Um.T)
            return tqenc.rotate(np.ascontiguousarray(ff.astype(np.float32)), signs[N_FF]).astype(np.float64) @ Dm.T

        ref = out(G, U, D)
        energy += float(np.sum(w2 * np.sum(ref ** 2, axis=1)))
        for c, src in cands.items():
            if src is None:
                Gq, Uq, Dq = encode_expert(c, e, G, U, D)
            else:
                Gq, Uq, Dq = (src.get(r, e).astype(np.float64) for r in ("gate", "up", "down"))
            combos = {"all": (Gq, Uq, Dq), "gate only": (Gq, U, D), "up only": (G, Uq, D), "down only": (G, U, Dq)}
            for p in parts:
                d = out(*combos[p]) - ref
                err[(c, p)] += float(np.sum(w2 * np.sum(d ** 2, axis=1)))
    print("experts scored %d; MoE-output SQNR in dB (higher is better), router-weighted over every routed pair:" % used)
    print("  %-12s" % "" + "".join("%12s" % p for p in parts))
    for c in cands:
        print("  %-12s" % c + "".join("%12.3f" % (10 * np.log10(energy / err[(c, p)])) for p in parts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
