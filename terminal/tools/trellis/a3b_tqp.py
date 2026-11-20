"""a3b_tqp.py - Qwen3.6-35B-A3B with trellis (TQP) experts, measured by KL: a GGUF whose 40 main layers' routed experts are
replaced and everything else (Q6_K attention / SSM / shared expert, the MTP layer) is a base file's, e.g. Unsloth's
UD-IQ3_XXS.  The A3B engine has no Hadamard step, so the experts are stored DECODED as Q8_0 in the original basis: the
file measures the TQP experts' quality, it is not the small file itself (the packed TQP tensors are saved beside it).

  ref    experts = Q8_0 of the BF16 originals: the KL reference (only the experts differ from the base file)
  copy   the base file's experts as they are
  deq    the base file's experts decoded (ik_llama's ggml: ikdeq, --ikdeq) and stored as Q8_0 like ours: Unsloth's own
         quant on the same footing (the A3B engine's native expert kernels take neither IQ2_S nor IQ3_S)
  tqp    experts = TQP5 / 6 / 7 per --plan, ldlr-encoded (build_swap.py's ldlr: each expert's calibration tokens weighted
         by router weight^2, shrunk to the layer's pooled H) in a Hadamard basis of our own signs, decoded, unrotated

  python tools/trellis/a3b_tqp.py --base UD-IQ3_XXS-MTP.gguf --bf16 BF16-00001-of-00002.gguf --mode ref --out REF.gguf
  python tools/trellis/a3b_tqp.py ... --mode tqp --dump-dir POOLED --calib-dir RAW1 --calib-dir RAW2 --gpu --out OUT.gguf

Every mode writes the MTP block's experts (Unsloth's Q2_K / Q3_K, which the native expert kernels do not take) as zero Q8_0
tensors, so the engine loads the file; the KL runs use --mtp 0, and the main model's logits never read that block.

The calibration dumps come from the ref file (STRATA_ACT_DUMP / _COV / _RAW, quant_quality.py run --env); A3B's dumps are in
the original basis, so the pooled H is rotated here.
"""
from __future__ import annotations

import argparse
import pathlib
import re
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
from build_swap import bf16, load_hessian, q8_0_quantize  # noqa: E402

N_EXP, N_EMBD, N_FF, N_LAYER = 256, 2048, 512, 40
TQP5, TQP6, TQP7 = 147, 148, 149
TQP8 = tqenc.TQP8   # experimental (4.125 bpw): packs, but no engine here decodes it yet
Q8_0 = 8
SEED_GU, SEED_DN = 3601, 3602   # our Hadamard signs (any fixed vectors; the decoded weights are stored unrotated)

# the UD-IQ3_XXS budget (2.889 bpw on the main experts): gate/up TQP5 + down TQP6 everywhere, the layers Unsloth bumps
# (34, 38 down; 39 all) plus the first two layers one tier up, and a TQP7 down on twelve more layers
PLANS = {
    "iq3xxs": dict(base=(TQP5, TQP6),
                   layers={**{l: (TQP5, TQP7) for l in (2, 3, 4, 5, 6, 7, 31, 32, 33, 34, 35, 36, 37, 38)},
                           **{l: (TQP6, TQP7) for l in (0, 1, 39)}}),
    # a 4-bit tier (4.125 bpw) against Unsloth's UD-Q4_K_XL (4.877 bpw: Q4_K gate/up, Q5_K / Q6_K down)
    "tq4": dict(base=(TQP8, TQP8), layers={}),
}


def plan_types(name):
    p = PLANS[name]
    return [p["layers"].get(l, p["base"]) for l in range(N_LAYER)]


def plan_bpw(types):
    bits = sum(2 * tqenc.bpw(gu) + tqenc.bpw(d) for gu, d in types)
    return bits / (3 * len(types))


# ------------------------------------------------------------------------------------------------ GGUF in / out

def header_end(g: gguf_reader.GGUFFile) -> int:
    """Byte offset where the tensor infos start (the end of the metadata)."""
    info = sum(8 + len(t.name.encode()) + 4 + 8 * len(t.shape) + 4 + 8 for t in g.tensors)
    return g._pos - info


def tensor_bytes(g, ts, i, file_size):
    """Byte count of tensor i of ts (sorted by offset): up to the next tensor or the end of the file."""
    end = ts[i + 1].offset if i + 1 < len(ts) else file_size - g.data_start
    return end - ts[i].offset


def ggml_name(type_name):
    """ggml's own spelling of a type name (q3_K, iq2_s): what ikdeq / ikquant look up."""
    n = type_name.lower()
    return n[:-2] + "_K" if n.endswith("_k") else n


def kv_bytes(key, value):
    """One GGUF metadata entry: a uint32 (int), a string, an int32 array (numpy) or a string array (list of str)."""
    k = key.encode()
    b = struct.pack("<Q", len(k)) + k
    if isinstance(value, np.ndarray):
        return b + struct.pack("<IIQ", 9, 5, value.size) + value.astype(np.int32).tobytes()
    if isinstance(value, list):
        return b + struct.pack("<IIQ", 9, 8, len(value)) + b"".join(
            struct.pack("<Q", len(x.encode())) + x.encode() for x in value)
    if isinstance(value, str):
        return b + struct.pack("<IQ", 8, len(value.encode())) + value.encode()
    return b + struct.pack("<II", 4, value)


def write_gguf(base_path, out_path, replace, extra_kv=None):
    """Copy base_path to out_path with the tensors in `replace` (name -> (type_id, callable returning bytes)) swapped.
    A Q8_0 replacement has the Q8_0 size; any other type is the size of what the callable returns (called once here).
    extra_kv: {key: value} appended to the metadata (kv_bytes)."""
    g = gguf_reader.GGUFFile(base_path)
    size = pathlib.Path(base_path).stat().st_size
    ts = sorted(g.tensors, key=lambda t: t.offset)
    nbytes = {t.name: tensor_bytes(g, ts, i, size) for i, t in enumerate(ts)}
    al = g.alignment
    new_type, new_size = {}, {}
    for t in g.tensors:
        if t.name in replace:
            ty = replace[t.name][0]
            new_type[t.name] = ty
            if ty == Q8_0:
                new_size[t.name] = t.elements // 32 * 34
            else:
                b = replace[t.name][1]()
                replace[t.name] = (ty, lambda b=b: b)
                new_size[t.name] = len(b)
        else:
            new_type[t.name] = t.type_id
            new_size[t.name] = nbytes[t.name]
    offs, pos = {}, 0
    for t in ts:   # the base file's data order
        offs[t.name] = pos
        pos += (new_size[t.name] + al - 1) // al * al
    with open(base_path, "rb") as fi, open(out_path, "wb") as fo:
        head = bytearray(fi.read(header_end(g)))
        if extra_kv:   # the metadata count is the uint64 after the magic, the version and the tensor count
            n_kv = struct.unpack_from("<Q", head, 16)[0]
            struct.pack_into("<Q", head, 16, n_kv + len(extra_kv))
            head += b"".join(kv_bytes(k, v) for k, v in extra_kv.items())
        fo.write(head)
        for t in g.tensors:   # the infos in the base file's info order
            nm = t.name.encode()
            fo.write(struct.pack("<Q", len(nm)) + nm + struct.pack("<I", len(t.shape)))
            fo.write(struct.pack("<%dQ" % len(t.shape), *t.shape))
            fo.write(struct.pack("<IQ", new_type[t.name], offs[t.name]))
        fo.write(b"\0" * (-fo.tell() % al))
        data0 = fo.tell()
        for t in ts:
            assert fo.tell() == data0 + offs[t.name]
            if t.name in replace:
                b = replace[t.name][1]()
                assert len(b) == new_size[t.name], (t.name, len(b), new_size[t.name])
            else:
                fi.seek(g.data_start + t.offset)
                b = fi.read(new_size[t.name])
            fo.write(b)
            fo.write(b"\0" * (-fo.tell() % al))
    return pos


class BF16Model:
    """The BF16 shards: expert tensors as numpy [n_exp, rows, cols] float32."""

    def __init__(self, first: str):
        m = re.match(r"(.*)-00001-of-(\d{5})\.gguf$", first)
        paths = [first] if not m else ["%s-%05d-of-%s.gguf" % (m.group(1), i, m.group(2)) for i in range(1, int(m.group(2)) + 1)]
        self.where = {}
        for p in paths:
            g = gguf_reader.GGUFFile(p)
            for t in g.tensors:
                self.where[t.name] = (p, g.data_start + t.offset, t)

    def experts(self, name):
        p, off, t = self.where[name]
        assert t.type_name == "BF16", (name, t.type_name)
        n_exp, rows, cols = t.shape[2], t.shape[1], t.shape[0]
        raw = np.memmap(p, dtype=np.uint16, mode="r", offset=off, shape=(n_exp, rows, cols))
        return raw


def q8_bytes(w):
    return q8_0_quantize(np.ascontiguousarray(w, dtype=np.float32)).tobytes()


# ------------------------------------------------------------------------------------------------ the ldlr encode

def ldlr_layer(l, a, gu, dn, gu_t, d_t, signs, router):
    """gu [N_EXP, 2 N_FF, N_EMBD] (gate rows then up rows), dn [N_EXP, N_EMBD, N_FF], float32, original basis.
    Returns the decoded (gate+up, down) in the original basis and the packed tensors."""
    import proxy_moe
    sg, sd = signs
    Hp_gu = load_hessian(a.dump_dir, l, "gu")
    Hp_dn = load_hessian(a.dump_dir, l, "dn")
    # A3B's dumps are in the original basis: H -> U H U^T
    Hp_gu = tqenc.rotate(tqenc.rotate(Hp_gu, sg).T.copy(), sg).astype(np.float64)
    Hp_dn = tqenc.rotate(tqenc.rotate(Hp_dn, sd).T.copy(), sd).astype(np.float64)
    xs, ids_ = [], []
    for d in a.calib_dir:
        i_, x_, k_ = proxy_moe.read_raw(pathlib.Path(d) / ("act_L%d_raw.bin" % l), 0)
        xs.append(x_)
        ids_.append(i_)
    x_ = np.concatenate(xs)
    ids = np.concatenate(ids_)
    lg = x_.astype(np.float64) @ router.T.astype(np.float64)
    lg -= lg.max(axis=1, keepdims=True)
    p = np.exp(lg)
    p /= p.sum(axis=1, keepdims=True)
    w_ = np.take_along_axis(p, ids, axis=1)
    w_ = w_ / w_.sum(axis=1, keepdims=True)
    xr = tqenc.rotate(np.ascontiguousarray(x_), sg)
    dec_gu = np.empty_like(gu)
    dec_dn = np.empty_like(dn)
    pk_gu = np.empty((N_EXP, 2 * N_FF, N_EMBD // 128, tqenc.block_bytes(gu_t)), dtype=np.uint8)
    pk_dn = np.empty((N_EXP, N_EMBD, N_FF // 128, tqenc.block_bytes(d_t)), dtype=np.uint8)
    kw = dict(gpu=a.gpu, iters=a.iters, ncand=a.ncand, threads=a.threads)
    for e0 in range(0, N_EXP, a.chunk):
        e1 = min(N_EXP, e0 + a.chunk)
        jobs_gu, jobs_dn = [], []
        for e in range(e0, e1):
            tok, slot = np.nonzero(ids == e)
            X = xr[tok]
            w2 = w_[tok, slot] ** 2
            Gr = tqenc.rotate(np.ascontiguousarray(gu[e, :N_FF]), sg)
            Ur = tqenc.rotate(np.ascontiguousarray(gu[e, N_FF:]), sg)
            ff = proxy_moe.silu(X @ Gr.T) * (X @ Ur.T)
            ffr = tqenc.rotate(np.ascontiguousarray(ff, dtype=np.float32), sd)
            jobs_gu.append((np.concatenate([Gr, Ur]),
                            blockldl.feedback(proxy_moe.shrunk(X, w2, Hp_gu, a.kappa), a.damp, gpu=a.gpu)))
            jobs_dn.append((tqenc.rotate(np.ascontiguousarray(dn[e]), sd),
                            blockldl.feedback(proxy_moe.shrunk(ffr, w2, Hp_dn, a.kappa), a.damp, gpu=a.gpu)))
        for i, (pk, dec) in enumerate(blockldl.encode_ldl_multi(jobs_gu, gu_t, **kw)):
            pk_gu[e0 + i] = pk
            dec_gu[e0 + i] = tqenc.unrotate(dec, sg)
        for i, (pk, dec) in enumerate(blockldl.encode_ldl_multi(jobs_dn, d_t, **kw)):
            pk_dn[e0 + i] = pk
            dec_dn[e0 + i] = tqenc.unrotate(dec, sd)
    return dec_gu, dec_dn, pk_gu, pk_dn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="the file whose other tensors are kept (Unsloth's UD-IQ3_XXS MTP GGUF)")
    ap.add_argument("--bf16", required=True, help="the first BF16 shard")
    ap.add_argument("--mode", required=True, choices=["ref", "copy", "deq", "tqp", "pack"])
    ap.add_argument("--ikdeq", default=r"C:\programs\ik_llama.cpp\build\bin\ikdeq.exe")
    ap.add_argument("--ikquant", default=r"C:\programs\ik_llama.cpp\build\bin\ikquant.exe")
    ap.add_argument("--mtp-types", default="iq2_s,iq4_xs", help="pack: the MTP block's experts (gate/up, down), "
                    "requantized from the base's (decoded with ikdeq) to types the expert kernels take; 'keep' keeps them")
    ap.add_argument("--out", required=True)
    ap.add_argument("--plan", default="iq3xxs", choices=sorted(PLANS))
    ap.add_argument("--layers", default=None, help="tqp: only these layers (comma list); the rest stay the base file's")
    ap.add_argument("--dump-dir", default=None, help="tqp: the pooled STRATA_ACT_DUMP (with _COV) directory")
    ap.add_argument("--calib-dir", action="append", default=[], help="tqp: STRATA_ACT_DUMP_RAW directories")
    ap.add_argument("--packed-dir", default=None, help="tqp: also save the packed TQP tensors here")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--ncand", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--damp", type=float, default=0.8)
    ap.add_argument("--kappa", type=float, default=1024.0)
    a = ap.parse_args()

    types = plan_types(a.plan)
    print("plan %s: %.3f bpw on the main experts (UD-IQ3_XXS: 2.889)" % (a.plan, plan_bpw(types)), flush=True)
    src = BF16Model(a.bf16)
    base = gguf_reader.GGUFFile(a.base)
    layers = range(N_LAYER) if not a.layers else [int(x) for x in a.layers.split(",")]
    replace = {}
    t0 = time.time()
    for r in (("gate", "up", "down") if a.mode != "pack" else ()):   # the MTP block (blk.40): zero Q8_0 experts
        nm = "blk.%d.ffn_%s_exps.weight" % (N_LAYER, r)
        t = {x.name: x for x in base.tensors}[nm]
        replace[nm] = (Q8_0, lambda n=t.elements: bytes(n // 32 * 34))
    if a.mode == "deq":
        import subprocess
        import tempfile
        tmp = pathlib.Path(tempfile.gettempdir()) / "a3b_ikdeq.f32"

        def deq(t):
            subprocess.run([a.ikdeq, ggml_name(t.type_name), a.base, str(base.data_start + t.offset), str(t.elements),
                            str(t.shape[0]), str(tmp)], check=True)
            return q8_bytes(np.fromfile(tmp, dtype=np.float32))

        bt = {x.name: x for x in base.tensors}
        for l in layers:
            for r in ("gate", "up", "down"):
                nm = "blk.%d.ffn_%s_exps.weight" % (l, r)
                replace[nm] = (Q8_0, lambda t=bt[nm]: deq(t))
    if a.mode in ("copy", "deq"):
        write_gguf(a.base, a.out, replace)
        print("wrote %s (%.0f s)" % (a.out, time.time() - t0))
        return 0
    if a.mode == "ref":
        for l in layers:
            for r in ("gate", "up", "down"):
                nm = "blk.%d.ffn_%s_exps.weight" % (l, r)
                replace[nm] = (Q8_0, lambda nm=nm: q8_bytes(bf16(np.asarray(src.experts(nm)))))
        write_gguf(a.base, a.out, replace)
        print("wrote %s (%.0f s)" % (a.out, time.time() - t0))
        return 0

    signs = (tqenc.make_signs(N_EMBD, SEED_GU), tqenc.make_signs(N_FF, SEED_DN))
    if a.mode == "pack":
        # the real file: the packed TQP experts from --packed-dir in place of the base's, the MTP experts --mtp-types.
        # The TQP rows are in a Hadamard basis over their inputs: gate/up rows take H(s_gu * x), down rows take H(s_dn * y)
        # for its input y (H is tqenc.hadamard_matrix, 128 wide, per block)
        pk_dir = pathlib.Path(a.packed_dir)
        by_name = {(tqenc.TYPE_NAMES[t], t) for t in (TQP5, TQP6, TQP7, TQP8)}
        if a.mtp_types != "keep":   # the MTP block: no BF16 for it (the BF16 release has none), so from the base's
            import subprocess
            import tempfile
            tf, tq = (pathlib.Path(tempfile.gettempdir()) / x for x in ("a3b_mtp.f32", "a3b_mtp.bin"))
            gu_ty, d_ty = a.mtp_types.split(",")
            ids = {"iq2_s": 22, "iq3_s": 21, "iq4_xs": 23, "q4_k": 12, "q5_k": 13, "q8_0": 8}
            bt = {x.name: x for x in base.tensors}

            def requant(t, ty):
                subprocess.run([a.ikdeq, ggml_name(t.type_name), a.base, str(base.data_start + t.offset), str(t.elements),
                                str(t.shape[0]), str(tf)], check=True)
                subprocess.run([a.ikquant, ggml_name(ty), str(tf), str(t.elements), str(t.shape[0]), str(tq)], check=True)
                return tq.read_bytes()

            for r, ty in (("gate", gu_ty), ("up", gu_ty), ("down", d_ty)):
                nm = "blk.%d.ffn_%s_exps.weight" % (N_LAYER, r)
                replace[nm] = (ids[ty], lambda t=bt[nm], ty=ty: requant(t, ty))
        for l in range(N_LAYER):
            for r in ("gate", "up", "down"):
                f = [p for p in pk_dir.glob("L%d_%s.*" % (l, r))]
                assert len(f) == 1, (l, r, f)
                ty = dict(by_name)[f[0].suffix[1:]]
                replace["blk.%d.ffn_%s_exps.weight" % (l, r)] = (ty, lambda f=f[0]: f.read_bytes())
        # Agention's prism.hadamard metadata (include/strata/kernels/cpu/hadamard.hpp): our rotation is theirs, with
        # explicit signs, one vector per input width
        sg, sd = np.load(pk_dir / "signs_gu.npy"), np.load(pk_dir / "signs_dn.npy")
        pos = write_gguf(a.base, a.out, replace, extra_kv={
            "prism.hadamard.version": 1,
            "prism.hadamard.block_size": 128,
            "prism.hadamard.transform": "normalized-sylvester-walsh-hadamard",
            "prism.hadamard.axis": "input-last-dimension",
            "prism.hadamard.sign_mode": "explicit",
            "prism.hadamard.weight_names": ["blk.%d.ffn_%s_exps.weight" % (l, r)
                                            for l in range(N_LAYER) for r in ("gate", "up", "down")],
            "prism.hadamard.sign_widths": np.array([N_EMBD, N_FF]),
            "prism.hadamard.sign_values": np.concatenate([sg, sd])})
        print("wrote %s: %.2f GB (%.0f s)" % (a.out, pos / 1e9, time.time() - t0))
        return 0
    binfo = {t.name: t for t in base.tensors}
    pk_dir = pathlib.Path(a.packed_dir) if a.packed_dir else None
    if pk_dir:
        pk_dir.mkdir(parents=True, exist_ok=True)
        np.save(pk_dir / "signs_gu.npy", signs[0])
        np.save(pk_dir / "signs_dn.npy", signs[1])
    cache = {}

    def layer(l):
        """encode layer l once, return (gate, up, down) decoded Q8_0 bytes; dropped after its three tensors are written"""
        if l not in cache:
            gu_t, d_t = types[l]
            ti = binfo["blk.%d.ffn_gate_inp.weight" % l]
            router = np.fromfile(a.base, dtype=np.float32, count=N_EXP * N_EMBD,
                                 offset=base.data_start + ti.offset).reshape(N_EXP, N_EMBD)
            g = bf16(np.asarray(src.experts("blk.%d.ffn_gate_exps.weight" % l)))
            u = bf16(np.asarray(src.experts("blk.%d.ffn_up_exps.weight" % l)))
            d = bf16(np.asarray(src.experts("blk.%d.ffn_down_exps.weight" % l)))
            gu = np.concatenate([g, u], axis=1)
            tl = time.time()
            dec_gu, dec_dn, pk_gu, pk_dn = ldlr_layer(l, a, gu, d, gu_t, d_t, signs, router)
            sq = lambda x, y: 10 * np.log10(np.sum(x.astype(np.float64) ** 2) / np.sum((x.astype(np.float64) - y) ** 2))
            print("layer %d %s/%s: SQNR gate+up %.2f dB, down %.2f dB (%.0f s, total %.0f s)"
                  % (l, tqenc.TYPE_NAMES[gu_t], tqenc.TYPE_NAMES[d_t], sq(gu, dec_gu), sq(d, dec_dn), time.time() - tl,
                     time.time() - t0), flush=True)
            if pk_dir:
                pk_gu[:, :N_FF].tofile(pk_dir / ("L%d_gate.%s" % (l, tqenc.TYPE_NAMES[gu_t])))
                pk_gu[:, N_FF:].tofile(pk_dir / ("L%d_up.%s" % (l, tqenc.TYPE_NAMES[gu_t])))
                pk_dn.tofile(pk_dir / ("L%d_down.%s" % (l, tqenc.TYPE_NAMES[d_t])))
            cache[l] = [q8_bytes(dec_gu[:, :N_FF]), q8_bytes(dec_gu[:, N_FF:]), q8_bytes(dec_dn), 0]
        c = cache[l]
        c[3] += 1
        if c[3] == 3:
            del cache[l]
        return c

    for l in layers:
        for i, r in enumerate(("gate", "up", "down")):
            replace["blk.%d.ffn_%s_exps.weight" % (l, r)] = (Q8_0, lambda l=l, i=i: layer(l)[i])
    write_gguf(a.base, a.out, replace)
    print("wrote %s (%.0f s)" % (a.out, time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
