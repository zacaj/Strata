"""a3b_kl.py - teacher-forced KL on the A3B engine (strata --native A3B.gguf --tokens --force --dump-full-logits): the
Flash-Next yardstick (quant_quality.py) needs the server's prompt scores, which the A3B engine's server does not give.

Per segment of an ids file (quant_quality.py's {"segments": [{"name", "ids"}]}): the first --prompt ids are the prompt,
the rest are forced one by one on the decode path, and the logits of every step are dumped.

  logits  python tools/trellis/a3b_kl.py run --model REF.gguf --ids set.ids.json --out DIR      (keeps DIR/<seg>.lp16)
  kl      python tools/trellis/a3b_kl.py run --model X.gguf --ids set.ids.json --out DIR2 --ref DIR  (KL vs DIR's)
  calib   python tools/trellis/a3b_kl.py run --model REF.gguf --ids bart.ids.json --out DIR --no-logits \\
              --env STRATA_ACT_DUMP=DIR/act --env STRATA_ACT_DUMP_LAYERS=0,...,39 --env STRATA_ACT_DUMP_COV=1 ...

--expert-slots 1 (default) puts the routed experts on the CPU path: the dumps see them, and every file runs the same way.
The reference's log-probabilities are kept as fp16 (DIR/<seg>.lp16, 0.5 MB a position); a candidate's are only compared.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

import numpy as np

V = 248320
EXE = r"C:\programs\Strata\build\strata.exe"


def log_softmax(x):
    x = x.astype(np.float64)
    m = x.max(axis=1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=1, keepdims=True))


def segments(path, max_len):
    d = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    for s in d["segments"]:
        ids = s["ids"]
        for i in range(0, len(ids), max_len):
            part = ids[i:i + max_len]
            if len(part) > 16:
                yield ("%s-%d" % (s["name"], i // max_len) if len(ids) > max_len else s["name"]), part


def run_engine(a, ids, logits_path, env):
    p = ids[:a.prompt]
    f = ids[a.prompt:]
    cmd = [a.exe, "--native", a.model, "--tokens", ",".join(map(str, p)), "--force", ",".join(map(str, f)),
           "--mtp", "0", "--expert-slots", str(a.expert_slots), "--max-context", str(len(ids) + 64)]
    if logits_path:
        cmd += ["--dump-full-logits", str(logits_path)]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise SystemExit("engine failed (rc %d):\n%s" % (r.returncode, r.stderr[-3000:]))
    return r.stderr


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--ids", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--ref", default=None, help="a reference run's directory: report KL against its log-probabilities")
    r.add_argument("--no-logits", action="store_true", help="calibration: dump nothing but the STRATA_ACT_DUMP files")
    r.add_argument("--env", action="append", default=[], help="NAME=VALUE for the engine")
    r.add_argument("--dump-dir", default=None, help="calibration: STRATA_ACT_DUMP=<dir>/<segment>/act per engine run (each run "
                   "writes its own files; merge_dumps.py adds the pooled ones, the raw ones are --calib-dir each)")
    r.add_argument("--prompt", type=int, default=1, help="ids given as the prompt; the rest are forced")
    r.add_argument("--seg-max", type=int, default=2000, help="split longer segments (the Windows command line)")
    r.add_argument("--expert-slots", type=int, default=1)
    r.add_argument("--exe", default=EXE)
    a = r.parse_args(sys.argv[2:]) if len(sys.argv) > 1 else ap.parse_args()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    for e in a.env:
        k, v = e.split("=", 1)
        env[k] = v
    rows = []
    for name, ids in segments(a.ids, a.seg_max):
        t0 = time.time()
        if a.dump_dir:
            d = pathlib.Path(a.dump_dir) / name
            d.mkdir(parents=True, exist_ok=True)
            env["STRATA_ACT_DUMP"] = str(d / "act")
        if a.no_logits:
            run_engine(a, ids, None, env)
            print("%s: %d tokens, %.1f s" % (name, len(ids), time.time() - t0), flush=True)
            continue
        tmp = out / (name + ".f32")
        run_engine(a, ids, tmp, env)
        lg = np.memmap(tmp, dtype=np.float32, mode="r").reshape(-1, V)
        n = len(ids) - a.prompt   # positions that predict a known next id
        tgt = np.array(ids[a.prompt:a.prompt + n])
        kl, top1, nll = np.empty(n), np.empty(n, dtype=bool), np.empty(n)
        ref = np.memmap(pathlib.Path(a.ref) / (name + ".lp16"), dtype=np.float16, mode="r").reshape(-1, V) if a.ref else None
        lp_out = None if a.ref else np.memmap(out / (name + ".lp16"), dtype=np.float16, mode="w+", shape=(n, V))
        for i0 in range(0, n, 128):
            i1 = min(n, i0 + 128)
            lq = log_softmax(np.asarray(lg[i0:i1]))
            nll[i0:i1] = -lq[np.arange(i1 - i0), tgt[i0:i1]]
            if ref is not None:
                lp = np.asarray(ref[i0:i1], dtype=np.float64)
                p = np.exp(lp)
                kl[i0:i1] = (p * (lp - lq)).sum(axis=1)
                top1[i0:i1] = lp.argmax(axis=1) == lq.argmax(axis=1)
            else:
                lp_out[i0:i1] = lq.astype(np.float16)
                kl[i0:i1] = 0.0
                top1[i0:i1] = True
        del lg
        tmp.unlink()
        if lp_out is not None:
            lp_out.flush()
        np.savez(out / (name + ".npz"), kl=kl, top1=top1, nll=nll)
        rows.append((name, n, kl.mean(), top1.mean(), nll.mean()))
        print("%-14s %5d positions  KL %.5f  top-1 %.4f  NLL %.4f  (%.0f s)" % (name, n, kl.mean(), top1.mean(), nll.mean(),
                                                                              time.time() - t0), flush=True)
    if rows:
        N = sum(r[1] for r in rows)
        print("%-14s %5d positions  KL %.5f  top-1 %.4f  NLL %.4f" % (
            "ALL", N, sum(r[1] * r[2] for r in rows) / N, sum(r[1] * r[3] for r in rows) / N, sum(r[1] * r[4] for r in rows) / N))
    return 0


if __name__ == "__main__":
    sys.exit(main())
