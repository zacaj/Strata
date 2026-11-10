// tools/trellis/tqenc_cuda.cu - the trellis encoder (tqenc.hpp) on an NVIDIA GPU: TQ2_T, TQK4X / TQK5X / TQK6 / TQK7,
// TQP5 / TQP6 / TQP7 (Gyro-L's pair trellis: the TQK states, 64 steps of 2 weights, the cost of the first point only).
// A port of tqenc_hip.cpp (same search, same C API: tqh_encode), plus TQ2_T, which the HIP encoder leaves to the CPU.
//
// One thread block (512 threads) runs one Viterbi pass over one 128-weight block; every thread keeps 64 path costs in
// registers.  TQK: thread t holds the states 64 t .. 64 t + 63, and a predecessor group (2^K contiguous states) never
// crosses more than one warp.  TQ2_T: a state's predecessor group is s & 127 and its successor group s >> 8, so the 256
// members of a group lie 128 states apart; thread t holds the 64 states (t & 127) + 128 h, h in [64 (t >> 7), +64): they
// all share one predecessor group, whose minimum is a register loop plus a combine of 4 partials in shared memory, and
// their successor groups h >> 1 are 32 consecutive entries of the group-minimum buffer.  The rest (codebook tables per
// step in shared memory, the rotated free pass for the tail-biting start group, the constrained pass, the fp16 scale
// refinement) is the CPU encoder's, batched over all blocks: every stage is one kernel over the whole batch.
//
// Build (from the repository root): nvcc -O3 -std=c++17 -arch=sm_120 -shared -Itools/trellis tools/trellis/tqenc_cuda.cu
// plus tqenc.cpp compiled by the host compiler (under nvcc, ggml-common-tq.h's codebook becomes a __device__ variable).
#include "tqenc.hpp"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CCHECK(x)                                                                                  \
    do {                                                                                           \
        cudaError_t e_ = (x);                                                                      \
        if (e_ != cudaSuccess) {                                                                   \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); \
            std::abort();                                                                          \
        }                                                                                          \
    } while (0)

namespace {

constexpr int kSL = 6;        // log2 of the states per thread (64: 512 threads per block)
constexpr int NT = 32768 >> kSL;
constexpr float kInf = 1e30f;
constexpr int kTQ2T = 144;    // tqenc::TQ2_T

__device__ __forceinline__ float h2f(uint16_t h) { return __half2float(__ushort_as_half(h)); }
__device__ __forceinline__ uint16_t f2h(float f) { return __half_as_ushort(__float2half_rn(f)); }

// weight index of component j at processing position p (TQK streams run backwards, TQ2_T forwards; TQP: 64 x 2)
__device__ __forceinline__ int widx_d(bool t2, int p, int j) { return t2 ? 4 * p + j : 4 * (31 - p) + j; }
__device__ __forceinline__ int widx_g(bool t2, bool pair, int p, int j) { return pair ? 2 * (63 - p) + j : widx_d(t2, p, j); }

// the computed codebook of TQK5X / TQK4X (src/kernels/cpu/tq_computed.hpp): four fp16 from two LCG words, no table
__device__ __forceinline__ void comp_vals(uint32_t s, float* w) {
    const uint32_t r0 = ((s * 89226354u + 64248484u) & 0x8FFF8FFFu) ^ 0x3B603B60u;
    const uint32_t r1 = ((s * 0x9E3779B1u + 0x7F4A7C15u) & 0x8FFF8FFFu) ^ 0x3B603B60u;
    w[0] = h2f((uint16_t) (r0 & 0xffffu));
    w[1] = h2f((uint16_t) (r0 >> 16));
    w[2] = h2f((uint16_t) (r1 & 0xffffu));
    w[3] = h2f((uint16_t) (r1 >> 16));
}

__device__ __forceinline__ float comp_cost(uint32_t s, const float* t, const float* h) {
    float w[4];
    comp_vals(s, w);
    const float e0 = t[0] - w[0], e1 = t[1] - w[1], e2 = t[2] - w[2], e3 = t[3] - w[3];
    return h[0] * e0 * e0 + h[1] * e1 * e1 + h[2] * e2 * e2 + h[3] * e3 * e3;
}

struct Pass {
    const float* w;
    const float* h;            // null: ones
    const uint16_t* dcur;      // fp16 scale per block
    const uint8_t* active;
    int64_t nb;
    int rot;
    const int* cin;            // null: free pass, else the tail-biting group per block (<0: skip)
    const float2* lutq;        // 2048 points
    uint16_t* bp_scratch;
    uint16_t* st_out;          // constrained: [nb][32] by position ([nb][64] for TQP)
    float* cost_out;           // constrained: [nb]
    int* c_out;                // free: junction group [nb]
};

// K: stream bits per step (8 for TQ2_T).  The state of register slot j of thread tid, and the hash stride between slots:
//   TQK:   s = SPT tid + j            (stride 1)
//   TQ2_T: s = (tid & 127) + 128 (SPT (tid >> 7) + j)   (stride 128)
template <int K, int SL, bool COMP, bool T2, bool PAIR>
__global__ void __launch_bounds__(32768 >> SL, 1) pass_kernel(Pass P) {
    constexpr int NS = PAIR ? 64 : 32, M = NS - 1;   // steps per block
    constexpr int SPT = 1 << SL;          // states per thread
    constexpr int NT = 32768 >> SL;       // threads per block
    constexpr int G = T2 ? 128 : 1 << (15 - K);
    constexpr int SPAN = 1 << (T2 ? 0 : K);
    constexpr int GPT = SPT >= SPAN ? SPT / SPAN : 1;   // groups per thread (SPT >= SPAN)
    constexpr int PC = SPT >= SPAN ? 1 : SPAN / SPT;    // threads sharing one group (SPAN > SPT)
    constexpr int NP = 2048 / NT;                       // codebook points per thread
    constexpr int NPART = T2 ? NT / 128 : 1;            // TQ2_T: threads holding a share of one predecessor group
    static_assert(!T2 || (SL == 6 && NT == 512), "the TQ2_T layout assumes 512 threads of 64 states");
    // the step tables (32 KB) are dynamic shared memory: with TQK4's group minima the static total would pass CUDA's
    // 48 KB static limit (launch_pass opts in to the larger dynamic size)
    extern __shared__ float tab_dyn[];
    float (*tab)[2][2048] = reinterpret_cast<float (*)[2][2048]>(tab_dyn);
    __shared__ float mbuf[2][G];
    __shared__ float Tt[NS][4], Hh[NS][4];
    __shared__ float redv[NT / 32];
    __shared__ int redi[NT / 32];
    __shared__ float partv[NPART][T2 ? 128 : 1];
    __shared__ uint16_t parts[NPART][T2 ? 128 : 1];

    const int tid = threadIdx.x;
    uint16_t* bp = P.bp_scratch + (size_t) blockIdx.x * NS * G;
    float2 L[NP];
#pragma unroll
    for (int k = 0; k < NP; ++k) L[k] = P.lutq[tid + NT * k];
    const uint32_t kC = 0x9e3779b1u;
    const int ga = tid & 127, hb = SPT * (tid >> 7);    // TQ2_T: predecessor group and first h of this thread
    const uint32_t s0 = T2 ? (uint32_t) (ga + 128 * hb) : (uint32_t) (SPT * tid);
    const uint32_t sstride = T2 ? 128u : 1u;
    const uint32_t x0 = s0 * kC, xstride = sstride * kC;

    for (int64_t b = blockIdx.x; b < P.nb; b += gridDim.x) {
        if (!P.active[b]) continue;
        int fc = -1;
        if (P.cin) {
            fc = P.cin[b];
            if (fc < 0) continue;
        }
        if (tid < 128) {
            const int p = PAIR ? tid >> 1 : tid >> 2, j = PAIR ? tid & 1 : tid & 3;
            const int wi = widx_g(T2, PAIR, p, j);
            const float dd = h2f(P.dcur[b]);
            Tt[p][j] = P.w[b * 128 + wi] / dd;
            Hh[p][j] = P.h ? P.h[b * 128 + wi] : 1.0f;
        }
        __syncthreads();

        float D[SPT];
        // tables of step q into parity q & 1
        auto build = [&](int q) {
            if constexpr (COMP) return;
            const int p = (P.rot + q) & M;
            if constexpr (PAIR) {   // only the first point of the state counts
                const float t0 = Tt[p][0], t1 = Tt[p][1], h0 = Hh[p][0], h1 = Hh[p][1];
#pragma unroll
                for (int k = 0; k < NP; ++k) {
                    const float a0 = t0 - L[k].x, a1 = t1 - L[k].y;
                    tab[q & 1][0][tid + NT * k] = h0 * a0 * a0 + h1 * a1 * a1;
                }
                return;
            }
            const float t0 = Tt[p][0], t1 = Tt[p][1], t2 = Tt[p][2], t3 = Tt[p][3];
            const float h0 = Hh[p][0], h1 = Hh[p][1], h2 = Hh[p][2], h3 = Hh[p][3];
#pragma unroll
            for (int k = 0; k < NP; ++k) {
                const float a0 = t0 - L[k].x, a1 = t1 - L[k].y, a2 = t2 - L[k].x, a3 = t3 - L[k].y;
                tab[q & 1][0][tid + NT * k] = h0 * a0 * a0 + h1 * a1 * a1;
                tab[q & 1][1][tid + NT * k] = h2 * a2 * a2 + h3 * a3 * a3;
            }
        };
        build(0);
        __syncthreads();
        {
            float tc[4], hc[4];
            if constexpr (COMP) {
                const int p0 = P.rot & M;
                for (int i = 0; i < 4; ++i) { tc[i] = Tt[p0][i]; hc[i] = Hh[p0][i]; }
            }
#pragma unroll
            for (int j = 0; j < SPT; ++j) {
                const uint32_t s = s0 + (uint32_t) j * sstride;
                const uint32_t x = x0 + (uint32_t) j * xstride;
                float v;
                if constexpr (COMP) v = comp_cost(s, tc, hc);
                else if constexpr (PAIR) v = tab[0][0][x >> 21];
                else v = tab[0][0][x >> 21] + tab[0][1][(x >> 10) & 2047u];
                // the first state's successor group must be the tail-biting group
                if (fc >= 0 && (T2 ? (int) (s >> 8) : (int) (s & (G - 1))) != fc) v = kInf;
                D[j] = v;
            }
        }
        for (int q = 1; q < NS; ++q) {
            const int par = q & 1;
            build(q);
            // group minima of the previous column
            uint16_t* bpq = bp + (size_t) q * G;
            if constexpr (T2) {
                // all 64 states of this thread are in predecessor group ga: the first minimum (smallest h) of the thread,
                // then of the NPART threads in h order - the CPU's strict-< scan over h = 0 .. 255
                float mv = D[0];
                int ma = 0;
#pragma unroll
                for (int e = 1; e < SPT; ++e)
                    if (D[e] < mv) { mv = D[e]; ma = e; }
                partv[tid >> 7][ga] = mv;
                parts[tid >> 7][ga] = (uint16_t) (s0 + (uint32_t) ma * 128u);
                __syncthreads();
                if (tid < 128) {
                    float v = partv[0][tid];
                    uint16_t st = parts[0][tid];
#pragma unroll
                    for (int r = 1; r < NPART; ++r)
                        if (partv[r][tid] < v) { v = partv[r][tid]; st = parts[r][tid]; }
                    mbuf[par][tid] = v;
                    bpq[tid] = st;
                }
            } else if constexpr (SPT >= SPAN) {
#pragma unroll
                for (int g = 0; g < GPT; ++g) {
                    float mv = D[g * SPAN];
                    int ma = 0;
#pragma unroll
                    for (int e = 1; e < SPAN; ++e) {
                        const float v = D[g * SPAN + e];
                        if (v < mv) { mv = v; ma = e; }
                    }
                    const int a = ((SPT * tid) >> K) + g;
                    mbuf[par][a] = mv;
                    bpq[a] = (uint16_t) (SPT * tid + g * SPAN + ma);
                }
            } else {
                // a group spans PC consecutive threads (the same warp): min/argmin of the thread, then across the PC
                float mv = D[0];
                int ma = 0;
#pragma unroll
                for (int e = 1; e < SPT; ++e) {
                    const float v = D[e];
                    if (v < mv) { mv = v; ma = e; }
                }
                int st = SPT * tid + ma;
#pragma unroll
                for (int off = 1; off < PC; off <<= 1) {
                    const float ov = __shfl_xor_sync(0xffffffffu, mv, off);
                    const int os = __shfl_xor_sync(0xffffffffu, st, off);
                    if (ov < mv || (ov == mv && os < st)) { mv = ov; st = os; }
                }
                if ((tid & (PC - 1)) == 0) {
                    const int a = (SPT * tid) >> K;
                    mbuf[par][a] = mv;
                    bpq[a] = (uint16_t) st;
                }
            }
            __syncthreads();
            // the successor group of slot j: TQK (s0 + j) & (G-1), contiguous; TQ2_T (hb + j) >> 1, two slots per entry
            const float* mrow = T2 ? &mbuf[par][hb >> 1] : &mbuf[par][(SPT * tid) & (G - 1)];
            // keep the compiler from caching the 2 * SPT hashed table addresses across the steps (they are loop invariant,
            // and 128 live registers spill): an empty asm makes x0 "new" every step
            uint32_t x0s = x0;
            asm volatile("" : "+r"(x0s));
            float tc[4], hc[4];
            if constexpr (COMP) {
                const int pq = (P.rot + q) & M;
                for (int i = 0; i < 4; ++i) { tc[i] = Tt[pq][i]; hc[i] = Hh[pq][i]; }
            }
#pragma unroll
            for (int j = 0; j < SPT; ++j) {
                const float mm = T2 ? mrow[j >> 1] : mrow[j];
                if constexpr (COMP) {
                    D[j] = mm + comp_cost(s0 + (uint32_t) j * sstride, tc, hc);
                } else {
                    const uint32_t x = x0s + (uint32_t) j * xstride;
                    if constexpr (PAIR) D[j] = mm + tab[par][0][x >> 21];
                    else D[j] = mm + tab[par][0][x >> 21] + tab[par][1][(x >> 10) & 2047u];
                }
            }
        }
        // final state: the last state's predecessor group must be the tail-biting group
        float bv = 3e38f;
        int bi = -1;
#pragma unroll
        for (int j = 0; j < SPT; ++j) {
            const int s = (int) (s0 + (uint32_t) j * sstride);
            const bool ok = fc < 0 || (T2 ? (s & 127) : (s >> K)) == fc;
            if (ok && D[j] < bv) { bv = D[j]; bi = s; }
        }
#pragma unroll
        for (int off = 16; off >= 1; off >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, off);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, off);
            if (ov < bv || (ov == bv && oi >= 0 && (bi < 0 || oi < bi))) { bv = ov; bi = oi; }
        }
        if ((tid & 31) == 0) { redv[tid >> 5] = bv; redi[tid >> 5] = bi; }
        __syncthreads();
        if (tid == 0) {
            float v = redv[0];
            int i = redi[0];
            for (int w2 = 1; w2 < NT / 32; ++w2)
                if (redv[w2] < v || (redv[w2] == v && redi[w2] >= 0 && (i < 0 || redi[w2] < i))) { v = redv[w2]; i = redi[w2]; }
            auto gsucc = [](int s) { return T2 ? (s >> 8) : (s & (G - 1)); };
            // traceback
            if (i < 0 || v >= kInf * 0.5f) {
                if (P.cin) P.cost_out[b] = kInf; else P.c_out[b] = 0;
            } else if (!P.cin) {
                const int qj = (M - P.rot) & M;   // the step that processes the last position
                int s = i;
                for (int q = M; q > qj; --q) s = bp[(size_t) q * G + gsucc(s)];
                P.c_out[b] = T2 ? (s & 127) : (s >> K);   // its predecessor group
            } else {
                int s = i;
                P.st_out[b * NS + ((P.rot + M) & M)] = (uint16_t) s;
                for (int q = M; q >= 1; --q) {
                    s = bp[(size_t) q * G + gsucc(s)];
                    P.st_out[b * NS + ((P.rot + q - 1) & M)] = (uint16_t) s;
                }
                P.cost_out[b] = v;
            }
        }
        __syncthreads();
    }
}

// ------------------------------------------------------------------------------------------------ per-block stages

__device__ __forceinline__ void state_cb(const float2* lutq, int comp, uint32_t s, float* c) {
    if (comp) { comp_vals(s, c); return; }
    const uint32_t x = s * 0x9e3779b1u;
    const float2 a = lutq[x >> 21], b = lutq[(x >> 10) & 2047u];
    c[0] = a.x; c[1] = a.y; c[2] = b.x; c[3] = b.y;
}

__global__ void init_kernel(const float* w, const float* h, int64_t nb, uint16_t* dcur, uint8_t* active, uint8_t* status,
                            double* sumhw2, double mult, int first, uint16_t fixed_scale, double cb_rms) {
    const int64_t b = blockIdx.x * (int64_t) blockDim.x + threadIdx.x;
    if (b >= nb) return;
    double s2 = 0, sh2 = 0, amax = 0;
    bool fin = true;
    for (int i = 0; i < 128; ++i) {
        const float x = w[b * 128 + i];
        const float hi = h ? h[b * 128 + i] : 1.0f;
        if (!isfinite(x) || !isfinite(hi)) fin = false;
        s2 += (double) x * x;
        sh2 += (double) hi * x * x;
        amax = fmax(amax, fabs((double) x));
    }
    if (first) {
        status[b] = !fin ? 2 : (amax == 0.0 ? 1 : 0);
        sumhw2[b] = sh2;
    }
    if (status[b] != 0) { active[b] = 0; return; }
    const double rms = sqrt(s2 / 128.0);
    double dd = rms / cb_rms * mult;
    uint16_t dh = f2h((float) fmin(dd, 65504.0));
    if (dh == 0) dh = 1;
    if (fixed_scale) dh = fixed_scale;
    dcur[b] = dh;
    active[b] = 1;
}

// error of (states by position st, fp16 scale dh)
__device__ double fit_d(const float* w, const float* h, int64_t b, const uint16_t* st, uint16_t dh, const float2* lutq, int comp,
                        bool t2, bool pair) {
    const float d = h2f(dh);
    const int ns = pair ? 64 : 32, nd = pair ? 2 : 4;
    double e = 0;
    for (int p = 0; p < ns; ++p) {
        float c[4];
        state_cb(lutq, comp, st[p], c);
        for (int j = 0; j < nd; ++j) {
            const int wi = widx_g(t2, pair, p, j);
            const float rec = d * c[j];
            const double df = (double) w[b * 128 + wi] - rec;
            e += (h ? (double) h[b * 128 + wi] : 1.0) * df * df;
        }
    }
    return e;
}

// pick the best candidate, then the scale refinement step of encode_block
__global__ void refine_kernel(const float* w, const float* h, int64_t nb, int nc, const uint16_t* st_c, const float* cost_c,
                              uint16_t* dcur, uint8_t* active, uint16_t* best_d, double* best_err, uint16_t* best_st,
                              const float2* lutq, int fixed_scale, int comp, int t2, int pair) {
    const int64_t b = blockIdx.x * (int64_t) blockDim.x + threadIdx.x;
    if (b >= nb || !active[b]) return;
    const int ns = pair ? 64 : 32, nd = pair ? 2 : 4;
    int bc = -1;
    float bcost = kInf * 0.5f;
    for (int r = 0; r < nc; ++r) {
        const float c = cost_c[(int64_t) r * nb + b];
        if (c < bcost) { bcost = c; bc = r; }
    }
    if (bc < 0) { active[b] = 0; return; }
    uint16_t st[64];
    for (int p = 0; p < ns; ++p) st[p] = st_c[((int64_t) bc * nb + b) * ns + p];
    const uint16_t dh = dcur[b];
    const double e = fit_d(w, h, b, st, dh, lutq, comp, t2, pair);
    if (e < best_err[b]) {
        best_err[b] = e;
        best_d[b] = dh;
        for (int p = 0; p < ns; ++p) best_st[b * ns + p] = st[p];
    }
    if (e == 0.0 || fixed_scale) { active[b] = 0; return; }
    double num = 0, den = 0;
    for (int p = 0; p < ns; ++p) {
        float c[4];
        state_cb(lutq, comp, st[p], c);
        for (int j = 0; j < nd; ++j) {
            const int wi = widx_g(t2, pair, p, j);
            const double hi = h ? (double) h[b * 128 + wi] : 1.0, cc = c[j];
            num += hi * w[b * 128 + wi] * cc;
            den += hi * cc * cc;
        }
    }
    if (den <= 0 || num <= 0) { active[b] = 0; return; }
    const uint16_t nh = f2h((float) fmin(num / den, 65504.0));
    if (nh == 0 || nh == dh) { active[b] = 0; return; }
    const double en = fit_d(w, h, b, st, nh, lutq, comp, t2, pair);
    if (en < best_err[b]) {
        best_err[b] = en;
        best_d[b] = nh;
        for (int p = 0; p < ns; ++p) best_st[b * ns + p] = st[p];
    }
    if (en >= e) { active[b] = 0; return; }
    dcur[b] = nh;
}

__global__ void dedupe_kernel(int64_t nb, int nc, int* c_all) {
    const int64_t b = blockIdx.x * (int64_t) blockDim.x + threadIdx.x;
    if (b >= nb) return;
    for (int r = 1; r < nc; ++r) {
        const int c = c_all[(int64_t) r * nb + b];
        for (int q = 0; q < r; ++q)
            if (c_all[(int64_t) q * nb + b] == c) { c_all[(int64_t) r * nb + b] = -1; break; }
    }
}

__global__ void pack_kernel(int K, int t2, int pair, int64_t nb, const uint8_t* status, const double* sumhw2, const uint16_t* best_d,
                            const double* best_err, const uint16_t* best_st, uint8_t* dst, double* err_out) {
    const int64_t b = blockIdx.x * (int64_t) blockDim.x + threadIdx.x;
    if (b >= nb) return;
    const int ns = pair ? 64 : 32, bb = 2 + ns / 8 * K;
    uint8_t* o = dst + b * bb;
    for (int i = 0; i < bb; ++i) o[i] = 0;
    if (status[b] == 2) { err_out[b] = NAN; return; }
    if (status[b] == 1 || best_err[b] >= sumhw2[b]) { err_out[b] = status[b] == 1 ? 0.0 : sumhw2[b]; return; }
    o[0] = (uint8_t) (best_d[b] & 255);
    o[1] = (uint8_t) (best_d[b] >> 8);
    uint8_t* qs = o + 2;
    if (t2) {   // TQ2_T: the low byte of each state is the stream byte (tqenc.cpp pack_stream)
        for (int u = 0; u < 32; ++u) qs[u] = (uint8_t) (best_st[b * 32 + u] & 0xff);
    } else {
        for (int u = 0; u < ns; ++u) {
            const uint32_t x = best_st[b * ns + u] & ((1u << K) - 1u);
            for (int bit = 0; bit < K; ++bit)
                if ((x >> bit) & 1u) qs[(u * K + bit) >> 3] |= (uint8_t) (1u << ((u * K + bit) & 7));
        }
    }
    err_out[b] = best_err[b];
}

__global__ void init_best_kernel(int64_t nb, double* best_err) {
    const int64_t b = blockIdx.x * (int64_t) blockDim.x + threadIdx.x;
    if (b >= nb) return;
    best_err[b] = 1e300;
}

template <int K, bool COMP, bool T2, bool PAIR = false>
void launch_pass(const Pass& P, int grid) {
    static constexpr int kTab = COMP ? 0 : 2 * 2 * 2048 * (int) sizeof(float);   // the computed codebook needs no tables
    static const bool attr = [] {
        CCHECK(cudaFuncSetAttribute(pass_kernel<K, kSL, COMP, T2, PAIR>, cudaFuncAttributeMaxDynamicSharedMemorySize, kTab));
        return true;
    }();
    (void) attr;
    pass_kernel<K, kSL, COMP, T2, PAIR><<<grid, NT, kTab>>>(P);
    CCHECK(cudaGetLastError());
}

void launch_pass_k(int K, bool comp, bool t2, bool pair, const Pass& P, int grid) {
    if (t2) { launch_pass<8, false, true>(P, grid); return; }
    if (pair) {   // TQP5 / TQP6 / TQP7
        if (K == 5) launch_pass<5, false, false, true>(P, grid);
        else if (K == 6) launch_pass<6, false, false, true>(P, grid);
        else launch_pass<7, false, false, true>(P, grid);
        return;
    }
    if (comp) {   // TQK5X / TQK4X
        if (K == 4) launch_pass<4, true, false>(P, grid);
        else launch_pass<5, true, false>(P, grid);
        return;
    }
    switch (K) {
        case 4: launch_pass<4, false, false>(P, grid); break;
        case 5: launch_pass<5, false, false>(P, grid); break;
        case 6: launch_pass<6, false, false>(P, grid); break;
        default: launch_pass<7, false, false>(P, grid); break;
    }
}

struct Dev {
    int device = 0;
    int nsm = 0;
    float2* lutq = nullptr;
    bool ready = false;
} g_dev;

void ensure_dev(int device) {
    if (g_dev.ready && g_dev.device == device) return;
    CCHECK(cudaSetDevice(device));
    cudaDeviceProp pr;
    CCHECK(cudaGetDeviceProperties(&pr, device));
    g_dev.device = device;
    g_dev.nsm = pr.multiProcessorCount;
    // the 2048 codebook points as the decoder sees them: round(fp16 * 8192) / 8192 (tq_cpu.cpp fill_table)
    std::vector<float2> pts(2048);
    const float* pt = tqenc::point_table();
    for (int i = 0; i < 2048; ++i) pts[i] = make_float2(pt[2 * i], pt[2 * i + 1]);
    if (g_dev.lutq) CCHECK(cudaFree(g_dev.lutq));
    CCHECK(cudaMalloc(&g_dev.lutq, 2048 * sizeof(float2)));
    CCHECK(cudaMemcpy(g_dev.lutq, pts.data(), 2048 * sizeof(float2), cudaMemcpyHostToDevice));
    g_dev.ready = true;
}

template <typename T>
T* dalloc(size_t n) {
    T* p = nullptr;
    CCHECK(cudaMalloc(&p, n * sizeof(T)));
    return p;
}

}  // namespace

extern "C" {
#if defined(_WIN32)
#define TQH_EXPORT __declspec(dllexport)
#else
#define TQH_EXPORT __attribute__((visibility("default")))
#endif

// GPU encode of nblocks blocks of TQ2_T / TQK4X / 5X / 6 / 7 / TQP5 / 6 / 7.  Same arguments as tqe_encode (iters, ncand, scale_tries;
// exact search is CPU only).  stats[0] = nonfinite blocks, stats[1] = zero blocks; timing[0] = kernel seconds,
// [1] = passes.  Returns -1 for a type it does not encode.
TQH_EXPORT double tqh_encode(int type, const float* w, const float* h, int64_t nblocks, uint8_t* dst, int iters, int ncand,
                             int scale_tries, int device, double* err_out, int64_t* stats, double* timing,
                             int fixed_scale_bits) {
    const int K = tqenc::type_kbits(type);
    const bool t2 = type == kTQ2T;
    if (!t2 && (K < 4 || K > 7)) return -1.0;
    const bool comp = tqenc::type_computed(type);
    const bool pair = tqenc::type_pair(type);
    const int NS = pair ? 64 : 32;   // steps per block
    const double cb_rms = tqenc::codebook_rms(type);
    ensure_dev(device);
    const int bb = tqenc::block_bytes(type), G = t2 ? 128 : 1 << (15 - K);
    const int nc = std::max(1, std::min(ncand, 8));
    static const int rots[8] = {16, 8, 24, 12, 20, 4, 28, 0};
    static const double mult[7] = {1.0, 0.9, 1.1, 0.8, 1.2, 0.7, 1.3};
    const int tries = fixed_scale_bits ? 1 : std::max(1, std::min(scale_tries, 7));
    const int64_t CH = 1 << 16;
    const int grid_cap = g_dev.nsm * 4;
    uint16_t* bp = dalloc<uint16_t>((size_t) grid_cap * NS * G);
    double tot = 0;
    int64_t nf = 0, zb = 0;
    double ksec = 0;
    int64_t npass = 0;
    for (int64_t c0 = 0; c0 < nblocks; c0 += CH) {
        const int64_t nb = std::min(CH, nblocks - c0);
        float* dw = dalloc<float>((size_t) nb * 128);
        float* dh = h ? dalloc<float>((size_t) nb * 128) : nullptr;
        CCHECK(cudaMemcpy(dw, w + c0 * 128, (size_t) nb * 128 * 4, cudaMemcpyHostToDevice));
        if (h) CCHECK(cudaMemcpy(dh, h + c0 * 128, (size_t) nb * 128 * 4, cudaMemcpyHostToDevice));
        uint16_t* dcur = dalloc<uint16_t>(nb);
        uint8_t* active = dalloc<uint8_t>(nb);
        uint8_t* status = dalloc<uint8_t>(nb);
        double* sumhw2 = dalloc<double>(nb);
        uint16_t* best_d = dalloc<uint16_t>(nb);
        double* best_err = dalloc<double>(nb);
        uint16_t* best_st = dalloc<uint16_t>((size_t) nb * NS);
        uint16_t* st_c = dalloc<uint16_t>((size_t) nc * nb * NS);
        float* cost_c = dalloc<float>((size_t) nc * nb);
        int* c_all = dalloc<int>((size_t) nc * nb);
        uint8_t* d_dst = dalloc<uint8_t>((size_t) nb * bb);
        double* d_err = dalloc<double>(nb);
        const int tb = 128, gb = (int) ((nb + tb - 1) / tb);
        init_best_kernel<<<gb, tb>>>(nb, best_err);
        const auto t0 = std::chrono::steady_clock::now();
        for (int tr = 0; tr < tries; ++tr) {
            init_kernel<<<gb, tb>>>(dw, dh, nb, dcur, active, status, sumhw2, mult[tr], tr == 0 ? 1 : 0,
                                    (uint16_t) fixed_scale_bits, cb_rms);
            for (int it = 0; it <= (fixed_scale_bits ? 0 : iters); ++it) {
                Pass P{};
                P.w = dw; P.h = dh; P.dcur = dcur; P.active = active; P.nb = nb; P.lutq = g_dev.lutq; P.bp_scratch = bp;
                const int grid = (int) std::min<int64_t>(nb, grid_cap);
                for (int r = 0; r < nc; ++r) {
                    P.rot = rots[r] * (NS / 32); P.cin = nullptr; P.c_out = c_all + (int64_t) r * nb;
                    launch_pass_k(K, comp, t2, pair, P, grid);
                }
                dedupe_kernel<<<gb, tb>>>(nb, nc, c_all);
                CCHECK(cudaMemset(cost_c, 0x7f, (size_t) nc * nb * sizeof(float)));   // skipped candidates cost "infinity"
                for (int r = 0; r < nc; ++r) {
                    P.rot = 0; P.cin = c_all + (int64_t) r * nb; P.c_out = nullptr;
                    P.st_out = st_c + (int64_t) r * nb * NS; P.cost_out = cost_c + (int64_t) r * nb;
                    launch_pass_k(K, comp, t2, pair, P, grid);
                }
                refine_kernel<<<gb, tb>>>(dw, dh, nb, nc, st_c, cost_c, dcur, active, best_d, best_err, best_st, g_dev.lutq,
                                          fixed_scale_bits, comp ? 1 : 0, t2 ? 1 : 0, pair ? 1 : 0);
                npass += 2 * nc;
            }
        }
        pack_kernel<<<gb, tb>>>(K, t2 ? 1 : 0, pair ? 1 : 0, nb, status, sumhw2, best_d, best_err, best_st, d_dst, d_err);
        CCHECK(cudaDeviceSynchronize());
        ksec += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        CCHECK(cudaMemcpy(dst + (size_t) c0 * bb, d_dst, (size_t) nb * bb, cudaMemcpyDeviceToHost));
        std::vector<double> herr(nb);
        std::vector<uint8_t> hst(nb);
        CCHECK(cudaMemcpy(herr.data(), d_err, nb * 8, cudaMemcpyDeviceToHost));
        CCHECK(cudaMemcpy(hst.data(), status, nb, cudaMemcpyDeviceToHost));
        for (int64_t i = 0; i < nb; ++i) {
            if (hst[i] == 2) ++nf;
            else tot += herr[i];
            if (hst[i] == 1) ++zb;
            if (err_out) err_out[c0 + i] = herr[i];
        }
        for (void* p : {(void*) dw, (void*) dh, (void*) dcur, (void*) active, (void*) status, (void*) sumhw2, (void*) best_d,
                        (void*) best_err, (void*) best_st, (void*) st_c, (void*) cost_c, (void*) c_all, (void*) d_dst,
                        (void*) d_err})
            if (p) CCHECK(cudaFree(p));
    }
    CCHECK(cudaFree(bp));
    if (stats) { stats[0] = nf; stats[1] = zb; }
    if (timing) { timing[0] = ksec; timing[1] = (double) npass; }
    return tot;
}

// blockldl.py's hessian_factor + feedback_matrices on the GPU, in fp32 (the damping keeps H well conditioned):
//   R upper with (H + damp mean(diag H) I)^-1 = R^T R,  then for every 128-block b but the last
//   F_b = R_bb^-1 R[b, b+1:]   ([128, n - (b+1) 128], row-major), written to `out` one after another.
// H is n x n row-major float64 (symmetric, so its column-major view is the same matrix).  Returns 0, or -1 when H + damp
// is not positive definite.
TQH_EXPORT int tqh_ldl_feedback(const double* H, int n, double damp, float* out, int device) {
    ensure_dev(device);
    static cusolverDnHandle_t sol = nullptr;
    static cublasHandle_t blas = nullptr;
    if (!sol) {
        if (cusolverDnCreate(&sol) != CUSOLVER_STATUS_SUCCESS || cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS) return -2;
    }
    const size_t nn = (size_t) n * n;
    double md = 0;
    for (int i = 0; i < n; ++i) md += H[(size_t) i * n + i];
    const double add = damp * md / n;
    std::vector<float> hf(nn);
    for (size_t i = 0; i < nn; ++i) hf[i] = (float) H[i];
    for (int i = 0; i < n; ++i) hf[(size_t) i * n + i] = (float) (H[(size_t) i * n + i] + add);
    float* A = dalloc<float>(nn);
    int* info = dalloc<int>(1);
    CCHECK(cudaMemcpy(A, hf.data(), nn * 4, cudaMemcpyHostToDevice));
    int lw1 = 0, lw2 = 0;
    // column-major, lower: H = L L^T, then the inverse from L (lower triangle), then Hinv = L2 L2^T
    if (cusolverDnSpotrf_bufferSize(sol, CUBLAS_FILL_MODE_LOWER, n, A, n, &lw1) != CUSOLVER_STATUS_SUCCESS) return -2;
    if (cusolverDnSpotri_bufferSize(sol, CUBLAS_FILL_MODE_LOWER, n, A, n, &lw2) != CUSOLVER_STATUS_SUCCESS) return -2;
    float* wk = dalloc<float>((size_t) std::max(lw1, lw2));
    int hinfo = 0;
    cusolverDnSpotrf(sol, CUBLAS_FILL_MODE_LOWER, n, A, n, wk, lw1, info);
    CCHECK(cudaMemcpy(&hinfo, info, 4, cudaMemcpyDeviceToHost));
    if (hinfo == 0) {
        cusolverDnSpotri(sol, CUBLAS_FILL_MODE_LOWER, n, A, n, wk, lw2, info);
        CCHECK(cudaMemcpy(&hinfo, info, 4, cudaMemcpyDeviceToHost));
    }
    if (hinfo == 0) {
        cusolverDnSpotrf(sol, CUBLAS_FILL_MODE_LOWER, n, A, n, wk, lw1, info);
        CCHECK(cudaMemcpy(&hinfo, info, 4, cudaMemcpyDeviceToHost));
    }
    if (hinfo != 0) {
        CCHECK(cudaFree(A)); CCHECK(cudaFree(info)); CCHECK(cudaFree(wk));
        return -1;
    }
    // The buffer now holds L2 in its column-major lower triangle, i.e. R = L2^T in its row-major upper triangle (the
    // other triangle is left over and never read).  F_b, row-major [128, n - c1], is in column-major terms
    // X = B A^-1 with B = R^T[c1:, c0:c1] and A = R^T[c0:c1, c0:c1] lower: a right-side lower trsm, in place.
    const float one = 1.0f;
    float* o = out;
    for (int c0 = 0; c0 + 128 < n; c0 += 128) {
        const int c1 = c0 + 128, m = n - c1;
        float* Ab = A + (size_t) c0 * n + c0;
        float* Bb = A + (size_t) c0 * n + c1;
        if (cublasStrsm(blas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, 128, &one,
                        Ab, n, Bb, n) != CUBLAS_STATUS_SUCCESS)
            return -2;
        // row-major rows c0..c1, columns c1..n of R's buffer: a 2D copy into the packed output
        CCHECK(cudaMemcpy2D(o, (size_t) m * 4, Bb, (size_t) n * 4, (size_t) m * 4, 128, cudaMemcpyDeviceToHost));
        o += (size_t) 128 * m;
    }
    CCHECK(cudaFree(A)); CCHECK(cudaFree(info)); CCHECK(cudaFree(wk));
    return 0;
}
}
