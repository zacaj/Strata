"""build_swap.py - Gyro-S with a few layers' experts swapped, without copying the 58 GB file.

Writes <var>/swap.bin (the new gate / up / down tensors of the chosen layers) and a pack <var>/pack whose native_experts.txt
points those layers at swap.bin (the shard column), everything else being the shipped Gyro-S pack and file: <var>/ holds a
symlink to the shipped GGUF, so `--native <var>/<gguf name>` finds swap.bin beside it.

Modes (all from the ORIGINAL Qwen weights, rotated into the file's basis with the file's own sign vectors):
  q8     Q8_0 (types 8/8): the near-lossless reference for the swapped layers
  plain  TQK6 gate/up + TQK7 down, plain-MSE trellis encode (tools/tqenc.py)
  diag   the same with per-input-column importance weights h = E[x^2]^power from the calibration dump (STRATA_ACT_DUMP)
  ldl    the same with block feedback from the full second-moment matrix of the dump (STRATA_ACT_DUMP_COV=1), blockldl.py

  python3 tools/trellis/build_swap.py --gguf SHIPPED --pack PACK --orig DIR --layers 4,16,28,40 --mode plain --var DIR [--gpu]
"""
from __future__ import annotations

import argparse
import pathlib
import re
import shutil
import struct
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import gguf_reader  # noqa: E402
import blockldl  # noqa: E402
import tqenc  # noqa: E402

N_EXP, N_EMBD, N_FF = 512, 2560, 640
ALIGN = 4096


def q8_0_quantize(w: np.ndarray) -> np.ndarray:
    """float32 [..., n] (n % 32 == 0) -> uint8 [..., n/32 * 34] block_q8_0 (fp16 scale, 32 int8)."""
    shp = w.shape
    b = w.reshape(-1, 32).astype(np.float32)
    amax = np.abs(b).max(axis=1)
    d = amax / 127.0
    d16 = d.astype(np.float16)
    dq = d16.astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(dq[:, None] > 0, b / dq[:, None], 0.0)
    q = np.clip(np.sign(q) * np.floor(np.abs(q) + 0.5), -127, 127).astype(np.int8)
    out = np.empty((b.shape[0], 34), dtype=np.uint8)
    out[:, :2] = d16.view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q.view(np.uint8)
    return out.reshape(*shp[:-1], shp[-1] // 32 * 34)


def link_to(dst: pathlib.Path, src: pathlib.Path) -> None:
    """A symlink; on Windows without symlink rights, a hard link for a file (same volume) or a copy of a directory."""
    try:
        dst.symlink_to(src, target_is_directory=src.is_dir())
    except OSError:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            os.link(src, dst)


def bf16(raw: np.ndarray) -> np.ndarray:
    return (raw.astype(np.uint32) << 16).view(np.float32)


def load_h(dump_dir, layer, kind, power):
    p = pathlib.Path(dump_dir) / ("act_L%d_%s.bin" % (layer, kind))
    raw = p.read_bytes()
    n, count = np.frombuffer(raw[:16], dtype=np.int64)
    sq = np.frombuffer(raw[16:16 + 8 * n], dtype=np.float64)
    h = (sq / max(count, 1)) ** power
    h = np.maximum(h, h.mean() * 1e-4)    # no weight below 1e-4 of the mean
    return (h / h.mean()).astype(np.float32), int(count)


def load_hessian(dump_dir, layer, kind):
    p = pathlib.Path(dump_dir) / ("act_L%d_%s.bin" % (layer, kind))
    raw = p.read_bytes()
    n, count = np.frombuffer(raw[:16], dtype=np.int64)
    n = int(n)
    if len(raw) < 16 + 8 * n + 8 * n * n:
        raise SystemExit("%s has no covariance (run the calibration with STRATA_ACT_DUMP_COV=1)" % p)
    cov = np.frombuffer(raw[16 + 8 * n:16 + 8 * n + 8 * n * n], dtype=np.float64).reshape(n, n)
    return cov / max(int(count), 1)


_expert_cache = {}


def load_expert_hessian(dump_dir, layer, kind, expert, kappa=64.0):
    """H of one expert from STRATA_ACT_DUMP_EXPERTS: the layer's pooled correlation R scaled by the expert's own channel
    energies, H_e = D_e^1/2 R D_e^1/2.  D_e is shrunk toward the pooled diagonal with kappa pseudo-tokens:
    (sum x^2 + kappa d_pool) / (count + kappa), so a rarely routed expert falls back to the layer's H."""
    key = (str(dump_dir), layer, kind)
    if key not in _expert_cache:
        Hp = load_hessian(dump_dir, layer, kind)
        dp = np.clip(np.diag(Hp).copy(), 1e-30, None)
        R = Hp / np.sqrt(np.outer(dp, dp))
        p = pathlib.Path(dump_dir) / ("act_L%d_%s_experts.bin" % (layer, kind))
        raw = p.read_bytes()
        n, ne = (int(v) for v in np.frombuffer(raw[:16], dtype=np.int64))
        cnt = np.frombuffer(raw[16:16 + 8 * ne], dtype=np.int64).astype(np.float64)
        sq = np.frombuffer(raw[16 + 8 * ne:16 + 8 * ne + 8 * ne * n], dtype=np.float64).reshape(ne, n)
        _expert_cache[key] = (dp, R, cnt, sq)
    dp, R, cnt, sq = _expert_cache[key]
    de = (sq[expert] + kappa * dp) / (cnt[expert] + kappa)
    s = np.sqrt(np.clip(de, 1e-30, None))
    return R * np.outer(s, s)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--orig", required=True)
    ap.add_argument("--layers", required=True)
    ap.add_argument("--mode", required=True, choices=["q8", "ext", "raw", "plain", "diag", "ldl", "ldle", "ldlr"],
                    help="ldle: ldl with each expert's own H (a STRATA_ACT_DUMP_EXPERTS dump, load_expert_hessian); "
                         "ldlr: each expert's calibration tokens weighted by router weight^2 (--calib-dir raw dumps), "
                         "shrunk to the pooled H of --dump-dir with --kappa")
    ap.add_argument("--calib-dir", action="append", default=[], help="ldlr: STRATA_ACT_DUMP_RAW directories")
    ap.add_argument("--var", required=True)
    ap.add_argument("--dump-dir", default=None)
    ap.add_argument("--ext-dir", default=None, help="ext: L<l>_gate.bf16, L<l>_up.bf16, L<l>_down.bf16 (another quantizer's "
                    "decoded experts, original basis), stored as Q8_0 in the rotated basis")
    ap.add_argument("--raw-dir", default=None, help="raw: L<l>_gate.bin, L<l>_up.bin, L<l>_down.bin, tensors already encoded "
                    "as --raw-type (another file's experts, copied as they are)")
    ap.add_argument("--raw-type", type=int, default=0, help="raw: the ggml type of the raw tensors (148 = TQP6)")
    ap.add_argument("--type", type=int, default=0, help="encoding modes: this ggml type for every expert tensor (148 = "
                    "TQP6) instead of the shipped file's types")
    ap.add_argument("--power", type=float, default=1.0)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--ncand", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--damp", type=float, default=0.3, help="ldl: damping, a fraction of the mean diagonal of H")
    ap.add_argument("--kappa", type=float, default=64.0, help="ldle: pseudo-tokens of the pooled diagonal per expert")
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",")]
    var = pathlib.Path(a.var)
    var.mkdir(parents=True, exist_ok=True)
    gg = pathlib.Path(a.gguf)
    link = var / gg.name
    if not link.exists():
        link_to(link, gg)
    m = re.match(r"(.*)-00001-of-(\d{5})\.gguf$", gg.name)
    if m:   # a split file: the pack names the other shards, beside the first
        for i in range(2, int(m.group(2)) + 1):
            sh = gg.with_name("%s-%05d-of-%s.gguf" % (m.group(1), i, m.group(2)))
            if not (var / sh.name).exists():
                link_to(var / sh.name, sh)
    g = gguf_reader.GGUFFile(a.gguf)
    sw = [int(x) for x in g.metadata["prism.hadamard.sign_widths"]]
    sv = np.array(g.metadata["prism.hadamard.sign_values"], dtype=np.float32)
    signs, at = {}, 0
    for w in sw:
        signs[w] = sv[at:at + w]
        at += w
    if a.mode in ("q8", "ext"):
        gu_t = d_t = 8
        row_gu, row_d = N_EMBD // 32 * 34, N_FF // 32 * 34
    elif a.mode == "raw":   # TQP5-7: 128 weights = fp16 + 8K bytes
        gu_t = d_t = a.raw_type
        bb = 2 + 8 * (a.raw_type - 142)
        row_gu, row_d = N_EMBD // 128 * bb, N_FF // 128 * bb
    else:
        # the shipped file's own types: TQK6 gate/up + TQK7 down for Gyro-S, TQ2_T for both in Gyro-M
        first = (pathlib.Path(a.pack) / "native_experts.txt").read_text().splitlines()[1].split()
        gu_t, d_t = int(first[1]), int(first[2])
        if a.type:
            gu_t = d_t = a.type
        row_gu, row_d = N_EMBD // 128 * tqenc.block_bytes(gu_t), N_FF // 128 * tqenc.block_bytes(d_t)
    t_gate = N_EXP * N_FF * row_gu          # bytes of the gate (and up) tensor of a layer
    t_down = N_EXP * N_EMBD * row_d
    offs = {}
    pos = 0
    for l in layers:
        offs[l] = {}
        for role, nbytes in (("gate", t_gate), ("up", t_gate), ("down", t_down)):
            offs[l][role] = pos
            pos += (nbytes + ALIGN - 1) // ALIGN * ALIGN
    # swap.bin is a GGUF of its own: the engine checks every shard native_experts.txt names (names, types, shapes,
    # offsets) before it maps the experts.  Offsets in offs are relative to its data start, as in a GGUF.
    hdr = [b"GGUF", struct.pack("<IQQ", 3, 3 * len(layers), 0)]
    for l in layers:
        for role, (cols, rows, ty) in (("gate", (N_EMBD, N_FF, gu_t)), ("up", (N_EMBD, N_FF, gu_t)),
                                       ("down", (N_FF, N_EMBD, d_t))):
            name = ("blk.%d.ffn_%s_exps.weight" % (l, role)).encode()
            hdr.append(struct.pack("<Q", len(name)) + name + struct.pack("<I3QIQ", 3, cols, rows, N_EXP, ty, offs[l][role]))
    hdr = b"".join(hdr)
    data_start = (len(hdr) + 31) // 32 * 32
    swap = open(var / "swap.bin", "wb")
    swap.write(hdr)
    swap.truncate(data_start + pos)

    def enc(w2d, t, h):
        kw = dict(iters=a.iters, ncand=a.ncand, gpu=a.gpu)
        if not a.gpu:
            kw["threads"] = a.threads
        hh = None
        if h is not None:
            hh = np.broadcast_to(h, w2d.shape).astype(np.float32)
        packed, err, tot, st = tqenc.encode(w2d, t, h=hh, **kw)
        return packed, tot

    stats = {}
    t_start = time.time()
    if a.mode == "raw":
        for l in layers:
            for role, nbytes in (("gate", t_gate), ("up", t_gate), ("down", t_down)):
                data = np.fromfile(pathlib.Path(a.raw_dir) / ("L%d_%s.bin" % (l, role)), dtype=np.uint8)
                assert data.size == nbytes, (role, data.size, nbytes)
                swap.seek(data_start + offs[l][role])
                swap.write(data.tobytes())
        layers_enc = []
    else:
        layers_enc = layers
    for l in layers_enc:
        gu_raw = np.memmap(pathlib.Path(a.orig) / ("L%d_gate_up.bf16" % l), dtype=np.uint16, mode="r").reshape(N_EXP, 2 * N_FF, N_EMBD)
        dn_raw = np.memmap(pathlib.Path(a.orig) / ("L%d_down.bf16" % l), dtype=np.uint16, mode="r").reshape(N_EXP, N_EMBD, N_FF)
        h_gu = h_dn = None
        fb_gu = fb_dn = None
        if a.mode == "ldlr":
            import proxy_moe   # here: it imports this module
            Hp_gu, Hp_dn = load_hessian(a.dump_dir, l, "gu"), load_hessian(a.dump_dir, l, "dn")
            ti = {t.name: t for t in g.tensors}["blk.%d.ffn_gate_inp.weight" % l]
            wr = np.fromfile(a.gguf, dtype=np.float32, count=N_EXP * N_EMBD,
                             offset=g.data_start + ti.offset).reshape(N_EXP, N_EMBD)
            xs, ids_ = [], []
            for d in a.calib_dir:
                i_, x_, k_ = proxy_moe.read_raw(pathlib.Path(d) / ("act_L%d_raw.bin" % l), 0)
                xs.append(x_)
                ids_.append(i_)
            x_ = np.concatenate(xs)
            rc = {"ids": np.concatenate(ids_)}
            lg = x_.astype(np.float64) @ wr.T.astype(np.float64)
            lg -= lg.max(axis=1, keepdims=True)
            p = np.exp(lg)
            p /= p.sum(axis=1, keepdims=True)
            w_ = np.take_along_axis(p, rc["ids"], axis=1)
            rc["w"] = w_ / w_.sum(axis=1, keepdims=True)
            rc["xr"] = tqenc.rotate(np.ascontiguousarray(x_), signs[N_EMBD])   # float32: the per-expert H in fp32
            print("layer %d: %d calibration tokens" % (l, len(x_)), flush=True)
        if a.mode == "ldl":
            fb_gu = blockldl.feedback(load_hessian(a.dump_dir, l, "gu"), a.damp, gpu=a.gpu)
            fb_dn = blockldl.feedback(load_hessian(a.dump_dir, l, "dn"), a.damp, gpu=a.gpu)
        if a.mode == "diag":
            h_gu, c1 = load_h(a.dump_dir, l, "gu", a.power)
            h_dn, c2 = load_h(a.dump_dir, l, "dn", a.power)
            print("layer %d: calibration counts gu %d dn %d; h cv gu %.2f dn %.2f" % (l, c1, c2, h_gu.std() / h_gu.mean(), h_dn.std() / h_dn.mean()))
        err_sum = {"gate": 0.0, "up": 0.0, "down": 0.0}
        en_sum = {"gate": 0.0, "up": 0.0, "down": 0.0}
        for e0 in range(0, N_EXP, a.chunk):
            e1 = min(N_EXP, e0 + a.chunk)
            if a.mode == "ext":
                ext = {r: np.memmap(pathlib.Path(a.ext_dir) / ("L%d_%s.bf16" % (l, r)), dtype=np.uint16, mode="r")
                       for r in ("gate", "up", "down")}
                gu = np.concatenate([bf16(np.asarray(ext["gate"].reshape(N_EXP, N_FF, N_EMBD)[e0:e1])),
                                     bf16(np.asarray(ext["up"].reshape(N_EXP, N_FF, N_EMBD)[e0:e1]))], axis=1)
                dn = bf16(np.asarray(ext["down"].reshape(N_EXP, N_EMBD, N_FF)[e0:e1]))
            else:
                gu = bf16(np.asarray(gu_raw[e0:e1]))
                dn = bf16(np.asarray(dn_raw[e0:e1]))
            mats = {"gate": (gu[:, :N_FF], signs[N_EMBD], h_gu, row_gu, N_FF, t_gate, gu_t),
                    "up": (gu[:, N_FF:], signs[N_EMBD], h_gu, row_gu, N_FF, t_gate, gu_t),
                    "down": (dn, signs[N_FF], h_dn, row_d, N_EMBD, t_down, d_t)}
            ldl_done = {}
            if a.mode == "ldlr":
                # each expert's own calibration tokens, weighted by their router weight squared, shrunk to the pooled H
                # (proxy_moe.py --encode ldl-r): gate and up share it; down gets the expert's own SwiGLU outputs
                parts = {"gate": [], "up": [], "down": []}
                jobs_gu, jobs_dn = [], []
                for i in range(e1 - e0):
                    e = e0 + i
                    tok, slot = np.nonzero(rc["ids"] == e)
                    X = rc["xr"][tok]
                    w2 = rc["w"][tok, slot] ** 2
                    Gr = tqenc.rotate(np.ascontiguousarray(gu[i, :N_FF]), signs[N_EMBD])
                    Ur = tqenc.rotate(np.ascontiguousarray(gu[i, N_FF:]), signs[N_EMBD])
                    ff = proxy_moe.silu(X @ Gr.T) * (X @ Ur.T)
                    ffr = tqenc.rotate(np.ascontiguousarray(ff, dtype=np.float32), signs[N_FF])
                    jobs_gu.append((np.concatenate([Gr, Ur]),
                                    blockldl.feedback(proxy_moe.shrunk(X, w2, Hp_gu, a.kappa), a.damp, gpu=a.gpu)))
                    jobs_dn.append((tqenc.rotate(np.ascontiguousarray(dn[i]), signs[N_FF]),
                                    blockldl.feedback(proxy_moe.shrunk(ffr, w2, Hp_dn, a.kappa), a.damp, gpu=a.gpu)))
                # the chunk's experts in one trellis call per block column, each with its own feedback
                kw = dict(gpu=a.gpu, iters=a.iters, ncand=a.ncand, threads=a.threads)
                for pk, dec in blockldl.encode_ldl_multi(jobs_gu, gu_t, **kw):
                    parts["gate"].append((pk[:N_FF].reshape(N_FF, -1), dec[:N_FF]))
                    parts["up"].append((pk[N_FF:].reshape(N_FF, -1), dec[N_FF:]))
                for pk, dec in blockldl.encode_ldl_multi(jobs_dn, d_t, **kw):
                    parts["down"].append((pk.reshape(pk.shape[0], -1), dec))
                for r, ps in parts.items():
                    ldl_done[r] = (np.concatenate([p for p, _ in ps]), np.concatenate([d for _, d in ps]))
            if a.mode == "ldle":
                # each expert with its own H (load_expert_hessian): gate + up share it, one batch of their rows
                parts = {"gate": [], "up": [], "down": []}
                for i in range(e1 - e0):
                    fbg = blockldl.feedback(load_expert_hessian(a.dump_dir, l, "gu", e0 + i, a.kappa), a.damp, gpu=a.gpu)
                    fbd = blockldl.feedback(load_expert_hessian(a.dump_dir, l, "dn", e0 + i, a.kappa), a.damp, gpu=a.gpu)
                    both = np.concatenate([tqenc.rotate(np.ascontiguousarray(gu[i, :N_FF]), signs[N_EMBD]),
                                           tqenc.rotate(np.ascontiguousarray(gu[i, N_FF:]), signs[N_EMBD])])
                    pk, dec = blockldl.encode_ldl(both, gu_t, fbg, gpu=a.gpu, iters=a.iters, ncand=a.ncand, threads=a.threads)
                    parts["gate"].append((pk[:N_FF].reshape(N_FF, -1), dec[:N_FF]))
                    parts["up"].append((pk[N_FF:].reshape(N_FF, -1), dec[N_FF:]))
                    wrd = tqenc.rotate(np.ascontiguousarray(dn[i]), signs[N_FF])
                    pk, dec = blockldl.encode_ldl(wrd, d_t, fbd, gpu=a.gpu, iters=a.iters, ncand=a.ncand, threads=a.threads)
                    parts["down"].append((pk.reshape(pk.shape[0], -1), dec))
                for r, ps in parts.items():
                    ldl_done[r] = (np.concatenate([p for p, _ in ps]), np.concatenate([d for _, d in ps]))
            if a.mode == "ldl":
                # gate and up share H: one batch of rows
                both = np.concatenate([tqenc.rotate(np.ascontiguousarray(gu[:, :N_FF]), signs[N_EMBD]).reshape(-1, N_EMBD),
                                       tqenc.rotate(np.ascontiguousarray(gu[:, N_FF:]), signs[N_EMBD]).reshape(-1, N_EMBD)])
                pk, dec = blockldl.encode_ldl(both, gu_t, fb_gu, gpu=a.gpu, iters=a.iters, ncand=a.ncand, threads=a.threads)
                half = pk.shape[0] // 2
                ldl_done["gate"] = (pk[:half].reshape(half, -1), dec[:half])
                ldl_done["up"] = (pk[half:].reshape(half, -1), dec[half:])
                wrd = tqenc.rotate(np.ascontiguousarray(dn), signs[N_FF]).reshape(-1, N_FF)
                pk, dec = blockldl.encode_ldl(wrd, d_t, fb_dn, gpu=a.gpu, iters=a.iters, ncand=a.ncand, threads=a.threads)
                ldl_done["down"] = (pk.reshape(pk.shape[0], -1), dec)
            for role, (w, sg, h, row, n_out, tb, ty) in mats.items():
                wr = tqenc.rotate(np.ascontiguousarray(w), sg)           # [chunk, n_out, n_in]
                n_in = wr.shape[-1]
                if a.mode in ("q8", "ext"):
                    data = q8_0_quantize(wr)
                    dec = None
                elif a.mode in ("ldl", "ldle", "ldlr"):
                    data, dec = ldl_done[role]
                    err_sum[role] += float(np.sum((wr.reshape(-1, n_in).astype(np.float64) - dec) ** 2))
                else:
                    packed, tot = enc(wr.reshape(-1, n_in), ty, h if a.mode == "diag" else None)
                    data = packed
                    err_sum[role] += tot
                en_sum[role] += float(np.sum(wr.astype(np.float64) ** 2))
                swap.seek(data_start + offs[l][role] + e0 * n_out * row)
                swap.write(np.ascontiguousarray(data).tobytes())
            if e0 % 128 == 0:
                print("layer %d experts %d-%d done  (%.0f s)" % (l, e0, e1, time.time() - t_start), flush=True)
        stats[l] = {r: (10 * np.log10(en_sum[r] / err_sum[r]) if err_sum[r] > 0 else None) for r in err_sum}
        print("layer %d weight SQNR (dB, unweighted): %s" % (l, {k: (round(v, 3) if v else v) for k, v in stats[l].items()}), flush=True)
    swap.close()
    # the pack: symlinks to the shipped pack's files, a new native_experts.txt
    pk = var / "pack"
    pk.mkdir(exist_ok=True)
    src = pathlib.Path(a.pack)
    for f in src.iterdir():
        if f.name == "native_experts.txt":
            continue
        d = pk / f.name
        if not d.exists():
            link_to(d, f.resolve())
    lines = (src / "native_experts.txt").read_text().splitlines()
    head, body = lines[0], lines[1:]
    out_lines, arena = [], 0
    for ln in body:
        f = ln.split()
        l = int(f[0])
        if l in offs:
            blob = row_gu * N_FF * 2 + row_d * N_EMBD
            f = [str(l), str(gu_t), str(d_t), str(arena), str(blob), str(data_start + offs[l]["gate"]),
                 str(data_start + offs[l]["up"]), str(data_start + offs[l]["down"]), "swap.bin"]
        else:
            f[3] = str(arena)
        out_lines.append(" ".join(f))
        arena += N_EXP * int(f[4])
    head = re.sub(r"total \d+", "total %d" % arena, head)
    (pk / "native_experts.txt").write_text(head + "\n" + "\n".join(out_lines) + "\n")
    print("wrote", var, "swap.bin %.2f GB, pack %s" % (pos / 1e9, pk))
    return 0


if __name__ == "__main__":
    sys.exit(main())
