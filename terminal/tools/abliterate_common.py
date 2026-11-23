"""tools/abliterate_common.py - shared helpers for abliterate_r_extract.py and abliterate_r_apply.py.

Abliteration removes one direction r (the "refusal direction") from every weight that writes into the residual stream:
    W' = W - alpha * r (r^T W)
GGUF stores a 2-D weight as shape [ne0 = input, ne1 = output]; numpy rows are the output (residual) axis, so r has ne1 entries.
A 3-D expert tensor is [input, output, n_expert] and is handled one expert matrix at a time.
"""
from __future__ import annotations

import hashlib
import os
import pathlib
import re
import sys
import tempfile
import time
import urllib.request

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import gguf_reader  # noqa: E402
import strata_quants  # noqa: E402

# the tensors that write into the residual stream (the ones abliteration touches); routed experts are opt-in
RESIDUAL_WRITERS = r"\.(attn_output|ssm_out|ffn_down_shexp|ffn_down)\.weight$"
EXPERT_WRITERS = r"\.ffn_down_exps\.weight$"

_SPLIT = re.compile(r"-(\d{5})-of-(\d{5})\.gguf(?=$|\?)")


def is_url(path) -> bool:
    return str(path).startswith(("http://", "https://"))


CACHE = pathlib.Path(tempfile.gettempdir()) / "strata-abliterate-cache"


def http_range(url: str, start: int, n: int, tries: int = 6) -> bytes:
    """bytes [start, start+n) of url, resumed across short reads and retried on errors"""
    out = bytearray()
    err = None
    for _ in range(tries):
        try:
            while len(out) < n:
                req = urllib.request.Request(url, headers={"Range": f"bytes={start + len(out)}-{start + n - 1}",
                                                           "User-Agent": "strata-abliterate"})
                with urllib.request.urlopen(req, timeout=120) as resp:
                    if resp.status != 206:
                        raise OSError(f"server ignored the Range header (HTTP {resp.status})")
                    chunk = resp.read()
                if not chunk:
                    raise OSError("empty read")
                out += chunk
            return bytes(out)
        except OSError as e:          # urllib errors are OSErrors
            err = e
            time.sleep(2)
    raise SystemExit(f"download failed: {url} @ {start}: {err}")


def _cache_path(url: str, start: int, n: int) -> pathlib.Path:
    return CACHE / hashlib.sha1(f"{url}|{start}|{n}".encode()).hexdigest()


def cached_range(url: str, start: int, n: int) -> bytes:
    p = _cache_path(url, start, n)
    if p.exists() and p.stat().st_size == n:
        return p.read_bytes()
    data = http_range(url, start, n)
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".part")
    tmp.write_bytes(data)
    os.replace(tmp, p)
    return data


def remote_header(url: str) -> gguf_reader.GGUFFile:
    """parse a remote GGUF's header from a growing prefix kept in the cache"""
    n = 8 << 20
    while True:
        p = _cache_path(url, 0, n)
        if not (p.exists() and p.stat().st_size == n):
            cached_range(url, 0, n)
        try:
            return gguf_reader.GGUFFile(p)
        except ValueError:
            if n >= 512 << 20:
                raise
            n *= 4


def split_parts(path):
    """every part of a split set, given any one part (a single file is its own set); local paths or URLs"""
    if is_url(path):
        url = str(path)
        m = _SPLIT.search(url)
        if not m:
            return [url]
        n = int(m.group(2))
        return [f"{url[:m.start()]}-{i:05d}-of-{n:05d}.gguf{url[m.end():]}" for i in range(1, n + 1)]
    p = pathlib.Path(path)
    m = _SPLIT.search(p.name)
    if not m:
        return [p]
    n = int(m.group(2))
    stem = p.name[:m.start()]
    parts = [p.with_name(f"{stem}-{i:05d}-of-{n:05d}.gguf") for i in range(1, n + 1)]
    missing = [q.name for q in parts if not q.exists()]
    if missing:
        raise SystemExit(f"missing split parts: {missing}")
    return parts


class Tensor:
    def __init__(self, part: pathlib.Path, g: gguf_reader.GGUFFile, t: gguf_reader.TensorInfo):
        self.part, self.g, self.t = part, g, t
        self.name, self.type = t.name, t.type_name
        self.shape = list(t.shape)
        bs, ts = strata_quants.BLOCK[self.type] if self.type in strata_quants.BLOCK else (None, None)
        if bs is None:
            self.row_bytes = None
        else:
            if self.shape[0] % bs:
                raise ValueError(f"{self.name}: row of {self.shape[0]} is not whole {self.type} blocks")
            self.row_bytes = self.shape[0] // bs * ts
        self.n_mat = self.shape[2] if len(self.shape) == 3 else 1
        self.rows = self.shape[1] if len(self.shape) >= 2 else 1
        self.mat_bytes = None if self.row_bytes is None else self.row_bytes * self.rows
        self.offset = g.data_start + t.offset           # absolute file offset

    @property
    def readable(self) -> bool:
        return self.row_bytes is not None and (self.type in ("F32", "F16") or self.type in strata_quants._DEQ)

    def read_raw(self, first: int = 0, count: int | None = None) -> np.ndarray:
        count = self.n_mat - first if count is None else count
        start, n = self.offset + first * self.mat_bytes, count * self.mat_bytes
        if is_url(self.part):
            raw = np.frombuffer(cached_range(self.part, start, n), np.uint8)
        else:
            with open(self.part, "rb") as f:
                f.seek(start)
                raw = np.frombuffer(f.read(n), np.uint8)
        return raw.reshape(count, self.rows, self.row_bytes)

    def read(self, first: int = 0, count: int | None = None) -> np.ndarray:
        """float32 [count, rows(out), cols(in)]"""
        return strata_quants.dequantize(self.read_raw(first, count), self.type).astype(np.float32, copy=False)


def tensors(path) -> dict[str, Tensor]:
    out = {}
    for part in split_parts(path):
        g = remote_header(part) if is_url(part) else gguf_reader.GGUFFile(part)
        for t in g.tensors:
            out[t.name] = Tensor(part, g, t)
    return out


def select(names, include: str, experts: bool) -> list[str]:
    pat = re.compile(include)
    ex = re.compile(EXPERT_WRITERS)
    return [n for n in names if pat.search(n) or (experts and ex.search(n))]
