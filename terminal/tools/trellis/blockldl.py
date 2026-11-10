"""blockldl.py - Hessian-aware trellis encoding by block feedback (the block form of GPTQ / LDLQ, as QTIP does it).

Weights W [m, n] in the rotated basis, H = E[x x^T] [n, n] of the rotated layer input.  Columns are cut into 128-blocks (the
trellis blocks).  Block b is encoded on its own (m blocks of 128, plain-MSE trellis search), its error E_b = W_b - Q_b is
pushed onto the not yet coded columns, exactly as the OBS update does for a whole block:

    W[:, c1:] -= E_b  R_bb^{-1}  R[c0:c1, c1:]          R upper, H^-1 = R^T R

so the later blocks absorb what the earlier ones could not represent, in the directions that matter for ||(W - Q) X||.
The result is a normal TQK6 / TQK7 / TQK5X / TQK4X tensor; only the encoder changed.
"""
from __future__ import annotations

import numpy as np

import tqenc

try:
    from scipy.linalg import solve_triangular
except Exception:   # numpy only: a small back substitution
    solve_triangular = None


def _solve_upper(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    if solve_triangular is not None:
        return solve_triangular(A, B, lower=False)
    return np.linalg.solve(A, B)


def hessian_factor(H: np.ndarray, damp: float = 0.01) -> np.ndarray:
    """R (upper) with (H + damp * mean(diag H) I)^-1 = R^T R."""
    H = np.asarray(H, dtype=np.float64).copy()
    d = damp * float(np.mean(np.diag(H)))
    H[np.diag_indices_from(H)] += d
    Hinv = np.linalg.inv(H)
    Hinv = (Hinv + Hinv.T) * 0.5
    L = np.linalg.cholesky(Hinv)
    return L.T


def feedback_matrices(R: np.ndarray, g: int = 128):
    """For every block b the matrix R_bb^{-1} R[b, b+1:] ([g, n - (b+1) g]); the last block has none."""
    n = R.shape[0]
    out = []
    for c0 in range(0, n, g):
        c1 = c0 + g
        out.append(_solve_upper(R[c0:c1, c0:c1], R[c0:c1, c1:]).astype(np.float32) if c1 < n else None)
    return out


def feedback(H: np.ndarray, damp: float = 0.01, gpu: bool = False):
    """The feedback matrices of H: on the GPU (feedback_gpu) or with numpy."""
    return feedback_gpu(H, damp) if gpu else feedback_matrices(hessian_factor(H, damp))


def feedback_gpu(H: np.ndarray, damp: float = 0.01, g: int = 128):
    """feedback_matrices(hessian_factor(H, damp)) computed by the CUDA encoder library (tqh_ldl_feedback, fp32, cuSOLVER),
    or by numpy when that library has no such entry (the HIP one)."""
    lib = tqenc.hip_lib()
    fn = getattr(lib, "tqh_ldl_feedback", None)
    if fn is None or g != 128:
        return feedback_matrices(hessian_factor(H, damp), g)
    import ctypes
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_double, ctypes.c_void_p, ctypes.c_int]
    H = np.ascontiguousarray(H, dtype=np.float64)
    n = H.shape[0]
    sizes = [g * (n - c1) for c1 in range(g, n, g)]
    buf = np.empty(sum(sizes), dtype=np.float32)
    rc = fn(H.ctypes.data, n, float(damp), buf.ctypes.data, 0)
    if rc != 0:
        raise RuntimeError("tqh_ldl_feedback failed (%d): H + damping is not positive definite?" % rc)
    out, at = [], 0
    for c1, sz in zip(range(g, n, g), sizes):
        out.append(buf[at:at + sz].reshape(g, n - c1))
        at += sz
    return out + [None]


def encode_ldl_multi(jobs, t: int, gpu: bool = False, iters: int = 1, ncand: int = 1, threads: int = 0):
    """encode_ldl for several matrices at once, each with its own feedback: jobs = [(W [m_i, n], fb_i)], all with the same
    n.  Block column b of every matrix goes to ONE trellis encode call (the rows are independent), then each matrix takes
    its own feedback step - the same blocks as encode_ldl matrix by matrix, in far fewer (GPU) calls.
    Returns [(packed_i, decoded_i)]."""
    n = jobs[0][0].shape[1]
    B = n // 128
    bb = tqenc.block_bytes(t)
    Ws = [np.ascontiguousarray(W, dtype=np.float32).copy() for W, _ in jobs]
    ms = [W.shape[0] for W in Ws]
    off = np.cumsum([0] + ms)
    packed = [np.empty((m, B, bb), dtype=np.uint8) for m in ms]
    dec = [np.empty((m, n), dtype=np.float32) for m in ms]
    kw = dict(iters=iters, ncand=ncand)
    if gpu:
        kw["gpu"] = True
    else:
        kw["threads"] = threads
    for b in range(B):
        c0, c1 = b * 128, (b + 1) * 128
        blk = np.concatenate([W[:, c0:c1] for W in Ws])
        pk, _, _, _ = tqenc.encode(blk, t, **kw)
        q = tqenc.decode(pk, t).reshape(-1, 128)
        for i, (W, (_, fb)) in enumerate(zip(Ws, jobs)):
            r0, r1 = off[i], off[i + 1]
            packed[i][:, b] = pk[r0:r1]
            dec[i][:, c0:c1] = q[r0:r1]
            if fb[b] is not None:
                W[:, c1:] -= (blk[r0:r1] - q[r0:r1]) @ fb[b]
    return list(zip(packed, dec))


def encode_ldl(W: np.ndarray, t: int, fb, gpu: bool = False, iters: int = 1, ncand: int = 1, threads: int = 0):
    """W [m, n] float32 (n % 128 == 0), fb from feedback_matrices.  Returns (packed [m, n/128, block_bytes], decoded [m, n])."""
    m, n = W.shape
    B = n // 128
    bb = tqenc.block_bytes(t)
    Wc = np.ascontiguousarray(W, dtype=np.float32).copy()
    packed = np.empty((m, B, bb), dtype=np.uint8)
    dec = np.empty((m, n), dtype=np.float32)
    kw = dict(iters=iters, ncand=ncand)
    if gpu:
        kw["gpu"] = True
    else:
        kw["threads"] = threads
    for b in range(B):
        c0, c1 = b * 128, (b + 1) * 128
        blk = np.ascontiguousarray(Wc[:, c0:c1])
        pk, _, _, _ = tqenc.encode(blk, t, **kw)
        q = tqenc.decode(pk, t).reshape(m, 128)
        packed[:, b] = pk
        dec[:, c0:c1] = q
        if fb[b] is not None:
            Wc[:, c1:] -= (blk - q) @ fb[b]
    return packed, dec
