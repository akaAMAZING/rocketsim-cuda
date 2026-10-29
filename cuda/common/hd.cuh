// Host/device portability layer. The same source compiles under nvcc (device and host
// passes), MSVC and gcc/clang. Every float operation of the physics goes through lf_add/lf_sub/lf_mul/lf_div/lf_sqrt:
// on the device they are the correctly rounded intrinsics (__fadd_rn ...), which nvcc never contracts into an FMA
// whatever the flags; on the host they are the plain operators, which stay unfused only when the host build forbids
// contraction (MSVC /fp:precise without /arch:AVX2, gcc/clang -ffp-contract=off) and has no FMA target (lf_host_fma()).
#pragma once
#include <stdint.h>
#include <string.h>
#include <math.h>

#if defined(__CUDACC__)
#define LF_HD __host__ __device__
#define LF_INL __forceinline__
#else
// Host builds (MSVC, gcc, clang) are the test and debugging path: plain inline lets the compiler decide, which keeps the
// one-translation-unit host shim of all ten phases within the compiler's memory (forced inlining made MSVC run out of heap in
// pass 2). Inlining never changes a result here: the host build forbids contraction and reassociation.
#define LF_HD
#define LF_INL inline
#endif
#define LF_HDI LF_HD LF_INL
// A function the device compiles once and calls (the ten phases, cuda/sim/phase.cuh): with all ten inlined, cicc took 8.6 GB on
// one architecture for one kernel, and every tick kernel repeats them. Never changes a result: every device float operation is an
// explicit correctly rounded intrinsic. Host builds leave the choice to the compiler.
#if defined(__CUDACC__)
#define LF_NOINLINE __noinline__
#else
#define LF_NOINLINE
#endif
// A loop the device compiler must not unroll (the decision loop, cuda/sim/arena_step.cuh). Host
// compilers do not see it.
#if defined(__CUDACC__)
#define LF_NO_UNROLL _Pragma("unroll 1")
#else
#define LF_NO_UNROLL
#endif

// 1 when the host translation unit may fuse a*b+c (the host reference must have none).
#if defined(__FMA__) || defined(__AVX2__) || defined(__FP_FAST_FMAF)
#define LF_HOST_FMA 1
#else
#define LF_HOST_FMA 0
#endif

#if defined(__CUDA_ARCH__)
LF_HDI float lf_add(float a, float b) { return __fadd_rn(a, b); }
LF_HDI float lf_sub(float a, float b) { return __fadd_rn(a, -b); }  // IEEE: a - b is a + (-b), zero signs included
LF_HDI float lf_mul(float a, float b) { return __fmul_rn(a, b); }
// Division: IEEE correctly rounded, bit-identical to __fdiv_rn, without __fdiv_rn's out-of-line slow path. That library routine saves and
// restores the convergence barriers it uses (BMOV through a GPR), the code pattern the build's convergence-barrier gate bans from every kernel
// that runs the phases (see crates/rocketsim-cuda-build/src/barrier.rs). The fast path is the library's own sequence (one reciprocal, one Newton step, one correction), taken when both
// operands and the quotient are in the window where it is exact; everything else (zero, subnormal, inf, NaN, extreme exponents) goes to
// lf_div_slow, an exact integer division that has no calls and one shallow nest of branches.
__device__ __noinline__ float lf_div_slow(float a, float b);
LF_HDI float lf_div(float a, float b) {
  uint32_t ia = __float_as_uint(a), ib = __float_as_uint(b);
  uint32_t ea = (ia >> 23) & 0xffu, eb = (ib >> 23) & 0xffu;
  uint32_t d = ea - eb + 100u;  // the quotient's exponent window: 2^-100 .. 2^120
  // A divisor with the all-ones mantissa (b = +-2^k (2 - 2^-23)) is the one class whose fast-path result depends on the exact
  // rcp.approx seed (a 1-2 ulp seed change breaks 192 of 5.6e10 such pairs; every other divisor is immune after the Newton step), and
  // other cards' MUFU.RCP cannot be run here, so that class takes the exact slow path on every card.
  bool fast = (ea - 30u <= 223u) & (eb - 3u <= 249u) & (d <= 220u) & ((ib & 0x7fffffu) != 0x7fffffu);  // not && : no branch
  float r;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(b));
  float e = __fmaf_rn(-b, r, 1.0f);
  r = __fmaf_rn(r, e, r);
  float q = __fmul_rn(a, r);
  float rem = __fmaf_rn(-b, q, a);
  q = __fmaf_rn(r, rem, q);
  if (!fast) q = lf_div_slow(a, b);
  return q;
}
LF_HDI float lf_sqrt(float a) { return __fsqrt_rn(a); }
LF_HDI uint32_t lf_bits(float a) { return __float_as_uint(a); }
LF_HDI float lf_from_bits(uint32_t u) { return __uint_as_float(u); }
LF_HDI float lf_i2f(int32_t v) { return __int2float_rn(v); }
LF_HDI float lf_u2f(uint32_t v) { return __uint2float_rn(v); }
LF_HDI float lf_ull2f(uint64_t v) { return __ull2float_rn(v); }
// The exact quotient, branch-free: no divergent region, so the out-of-line copy that every division site shares has no convergence
// barrier of its own to save around the barriers of its callers.
__device__ __noinline__ float lf_div_slow(float a, float b) {
  uint32_t ia = __float_as_uint(a), ib = __float_as_uint(b);
  uint32_t sign = (ia ^ ib) & 0x80000000u, aa = ia & 0x7fffffffu, ab = ib & 0x7fffffffu;
  bool a_nan = aa > 0x7f800000u, b_nan = ab > 0x7f800000u, a_inf = aa == 0x7f800000u, b_inf = ab == 0x7f800000u;
  bool a_zero = aa == 0, b_zero = ab == 0;
  // x = m * 2^(e - 150) with m in [2^23, 2^24) (a subnormal is shifted up, its exponent lowered); garbage for zero, inf and NaN
  int ea = (int)(aa >> 23), eb = (int)(ab >> 23);
  uint32_t ma = aa & 0x7fffffu, mb = ab & 0x7fffffu;
  int sha = ea == 0 ? (int)__clz(ma) - 8 : 0, shb = eb == 0 ? (int)__clz(mb) - 8 : 0;
  ma = ea == 0 ? ma << (sha & 31) : (ma | 0x800000u);
  mb = eb == 0 ? mb << (shb & 31) : (mb | 0x800000u);
  ea = ea == 0 ? 1 - sha : ea;
  eb = eb == 0 ? 1 - shb : eb;
  mb = mb == 0 ? 1u : mb;  // keep the loop's divisor nonzero for the lanes whose result a special case overrides
  // Q + frac = ma / mb * 2^25 (Q in [2^24, 2^26) for finite nonzero operands), frac > 0 iff the remainder is nonzero.
  uint32_t rem = ma, bit = rem >= mb;
  rem -= bit ? mb : 0u;
  uint32_t Q = bit;
#pragma unroll
  for (int i = 0; i < 25; ++i) {
    rem <<= 1;
    bit = rem >= mb;
    rem -= bit ? mb : 0u;
    Q = (Q << 1) | bit;
  }
  bool sticky = rem != 0;
  // value = (Q + frac) * 2^x. The result's last bit is 2^L, L = max(floor(log2 value) - 23, -149); s = L - x bits of Q are dropped.
  int x = ea - eb - 25, n = 32 - (int)__clz(Q);
  int L = x + n - 24;
  L = L < -149 ? -149 : L;
  int s = L - x;
  int sh = s > 31 ? 31 : (s < 1 ? 1 : s);
  uint32_t keep = Q >> sh, low = Q & ((1u << sh) - 1u), half = 1u << (sh - 1);
  bool up = (low > half) | ((low == half) & (sticky | ((keep & 1u) != 0)));  // | and &, not || and &&: no branch
  uint32_t M = keep + (up ? 1u : 0u);
  uint32_t bits = (((uint32_t)(L + 150)) << 23) + M - 0x800000u;
  bits = bits > 0x7f800000u ? 0x7f800000u : bits;
  bits = L + 150 >= 255 ? 0x7f800000u : bits;
  uint32_t r = sign | bits;
  r = (a_zero | b_inf) ? sign : r;                                       // 0 / x, x / inf
  r = (a_inf | b_zero) ? (sign | 0x7f800000u) : r;                       // inf / x, x / 0
  r = ((a_inf & b_inf) | (a_zero & b_zero)) ? 0x7fffffffu : r;           // inf / inf, 0 / 0
  r = (a_nan | b_nan) ? __float_as_uint(__fadd_rn(a, b)) : r;            // a NaN operand: the library returns a + b
  return __uint_as_float(r);
}
#else
LF_HDI float lf_add(float a, float b) { return a + b; }
LF_HDI float lf_sub(float a, float b) { return a - b; }
LF_HDI float lf_mul(float a, float b) { return a * b; }
LF_HDI float lf_div(float a, float b) { return a / b; }
#if defined(_MSC_VER)
LF_HDI float lf_sqrt(float a) { return sqrtf(a); }
#else
LF_HDI float lf_sqrt(float a) { return __builtin_sqrtf(a); }
#endif
LF_HDI uint32_t lf_bits(float a) { uint32_t u; memcpy(&u, &a, 4); return u; }
LF_HDI float lf_from_bits(uint32_t u) { float a; memcpy(&a, &u, 4); return a; }
LF_HDI float lf_i2f(int32_t v) { return (float)v; }
LF_HDI float lf_u2f(uint32_t v) { return (float)v; }
LF_HDI float lf_ull2f(uint64_t v) { return (float)v; }
#endif

// i64 -> f32 (Rust `as f32`, round to nearest even)
LF_HDI float lf_ll2f(int64_t v) {
  if (v >= -2147483648LL && v <= 2147483647LL) return lf_i2f((int32_t)v);
  float m = lf_ull2f(v < 0 ? 0ull - (uint64_t)v : (uint64_t)v);
  return v < 0 ? lf_from_bits(lf_bits(m) ^ 0x80000000u) : m;
}

// IEEE binary32 -> binary16 bits, round to nearest even, in integers (the host's conversion; NaN becomes 0x7FFF as on the device).
LF_HDI uint16_t lf_f2h_soft(float f) {
  uint32_t x = lf_bits(f), sign = (x >> 16) & 0x8000u, ax = x & 0x7FFFFFFFu;
  if (ax > 0x7F800000u) return 0x7FFF;                       // NaN
  if (ax >= 0x477FF000u) return (uint16_t)(sign | 0x7C00u);  // 65520 and above (inf included) round to inf
  if (ax >= 0x38800000u) {                                   // a normal half: drop 13 mantissa bits, rebias the exponent by 112
    uint32_t q = (ax >> 13) - (112u << 10), rem = ax & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (q & 1u))) q += 1;  // a carry moves into the exponent, as it should
    return (uint16_t)(sign | q);
  }
  if (ax <= 0x33000000u) return (uint16_t)sign;  // at most 2^-25: zero (2^-25 is a tie, to even)
  uint32_t m = (ax & 0x7FFFFFu) | 0x800000u, shift = 126u - (ax >> 23);  // a subnormal half: m * 2^(e - 126) units of 2^-24
  uint32_t q = m >> shift, rem = m & ((1u << shift) - 1u), half = 1u << (shift - 1u);
  if (rem > half || (rem == half && (q & 1u))) q += 1;  // reaching 0x400 is the smallest normal, correctly encoded
  return (uint16_t)(sign | q);
}

// The observation's fp16 conversion: the hardware cvt.rn.f16.f32 on the device, the same rounding in integers on the host.
LF_HDI uint16_t lf_f2h(float f) {
#if defined(__CUDA_ARCH__)
  unsigned short h;
  asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
  return h;
#else
  return lf_f2h_soft(f);
#endif
}
