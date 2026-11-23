"""a3b_swap_experts.py BASE SRC OUT [FIRST-LAST] - BASE's tensors with the main layers' routed experts taken from SRC (Q8_0
there): all 40 layers, or only FIRST..LAST (e.g. 16-39: the layers llama.cpp's --n-cpu-moe 16 leaves on the GPU)."""
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
lo, hi = (int(x) for x in sys.argv[4].split("-")) if len(sys.argv) > 4 else (0, 39)
g = gguf_reader.GGUFFile(src)
mm = np.memmap(src, dtype=np.uint8, mode="r")
replace = {}
for t in g.tensors:
    m = re.match(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$", t.name)
    if m and lo <= int(m[1]) <= hi:
        assert t.type_id == Q8_0, t.name
        o = g.data_start + t.offset
        n = t.elements // 32 * 34
        replace[t.name] = (Q8_0, lambda o=o, n=n: bytes(mm[o:o + n]))
assert len(replace) == 3 * (hi - lo + 1), len(replace)
write_gguf(base, out, replace)
print("wrote", out)
