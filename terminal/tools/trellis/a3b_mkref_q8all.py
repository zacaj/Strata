"""a3b_mkref_q8all.py BASE BF16 OUT - BASE (the Q8-expert reference) with every other quantized tensor re-made as Q8_0 from the
BF16 originals: a reference whose only rounding is Q8_0 everywhere (F32 / F16 tensors stay as they are)."""
import collections
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import gguf_reader  # noqa: E402
from a3b_tqp import BF16Model, write_gguf, q8_bytes, Q8_0  # noqa: E402
from build_swap import bf16  # noqa: E402

base, src_path, out = sys.argv[1:4]
src = BF16Model(src_path)
g = gguf_reader.GGUFFile(base)
replace, kinds = {}, collections.Counter()
for t in g.tensors:
    if t.type_id in (0, 1, Q8_0):   # F32, F16, already Q8_0
        continue
    if t.name not in src.where:   # the MTP block (blk.40): not in the BF16 file, and the KL runs never read it
        kinds["kept (no BF16)"] += 1
        continue
    p, off, st = src.where[t.name]
    assert st.type_name == "BF16" and list(st.shape) == list(t.shape), (t.name, st.type_name, st.shape, t.shape)
    n = t.elements

    def make(p=p, off=off, n=n, shape=tuple(reversed(t.shape))):
        return q8_bytes(bf16(np.memmap(p, dtype=np.uint16, mode="r", offset=off, shape=(n,))).reshape(shape))
    replace[t.name] = (Q8_0, make)
    kinds[t.type_name] += 1
print("re-made as Q8_0 from BF16:", dict(kinds))
write_gguf(base, out, replace)
print("wrote", out)
