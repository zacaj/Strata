"""a3b_swap_experts.py BASE SRC OUT - BASE's tensors with the 40 main layers' routed experts taken from SRC (Q8_0 there)."""
import pathlib
import re
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import gguf_reader  # noqa: E402
from a3b_tqp import write_gguf, Q8_0  # noqa: E402

base, src, out = sys.argv[1:4]
g = gguf_reader.GGUFFile(src)
mm = np.memmap(src, dtype=np.uint8, mode="r")
replace = {}
for t in g.tensors:
    m = re.match(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$", t.name)
    if m and int(m[1]) < 40:
        assert t.type_id == Q8_0, t.name
        o = g.data_start + t.offset
        n = t.elements // 32 * 34
        replace[t.name] = (Q8_0, lambda o=o, n=n: bytes(mm[o:o + n]))
assert len(replace) == 120, len(replace)
write_gguf(base, out, replace)
print("wrote", out)
