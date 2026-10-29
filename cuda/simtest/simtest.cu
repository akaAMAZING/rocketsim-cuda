// Device side of the glam parity test: the same ops as the host shim (simtest/ops.cuh), one case per thread. Compiled
// strict (--fmad=false -prec-div=true -prec-sqrt=true -ftz=false) by rocketsim-cuda's build.rs.
#include "common/libm.cuh"
#include "common/libm_cr.cuh"
#include "simtest/libm_inputs.cuh"
#include "simtest/ops.cuh"

enum {
#define X(name, in_n, out_n, ...) OP_##name,
  LF_OPS(X)
#undef X
  OP_COUNT
};

__device__ __forceinline__ void apply(int op, const float* i, uint32_t* o) {
  switch (op) {
#define X(name, in_n, out_n, ...) \
  case OP_##name: {               \
    __VA_ARGS__;                  \
    break;                        \
  }
    LF_OPS(X)
#undef X
    default: break;
  }
}

extern "C" __global__ void simtest_run(int op, int n, int in_n, int out_n, const float* in, uint32_t* out) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n) return;
  apply(op, in + (size_t)t * in_n, out + (size_t)t * out_n);
}

// ---- libm tests (tests/libm_cr.rs): the correctly rounded functions, and CUDA's own library beside them (native != 0) ----

__device__ __forceinline__ float libm1(int fn, int native, float x) {
  if (native) return fn == 0 ? ::sinf(x) : fn == 1 ? ::cosf(x) : ::tanf(x);
  return fn == 0 ? lfcr::sinf_cr(x) : fn == 1 ? lfcr::cosf_cr(x) : lfcr::tanf_cr(x);
}
__device__ __forceinline__ float libm2(int op, int native, float a, float b) {
  if (native) return op == 0 ? ::atan2f(a, b) : ::powf(a, b);
  return op == 0 ? lfcr::atan2f_cr(a, b) : lfcr::powf_cr(a, b);
}

// out[i] = f(float(bits base + i)): sin, cos, tan (fn 0..2) over a range of bit patterns
extern "C" __global__ void libm_unary(int fn, int native, uint64_t base, uint64_t n, uint32_t* out) {
  uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  out[i] = lf_bits(libm1(fn, native, lf_from_bits((uint32_t)(base + i))));
}

// out[i] = op(case base + i of the random stream): atan2f (op 0), powf (op 1)
extern "C" __global__ void libm_stream(int op, int native, uint64_t base, uint64_t n, uint32_t* out) {
  uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float a, b;
  lft::pair(op, base + i, &a, &b);
  out[i] = lf_bits(libm2(op, native, a, b));
}

// the special-value grid (fn 0..2 unary, 3 atan2f, 4 powf over all pairs)
extern "C" __global__ void libm_grid(int fn, int native, uint32_t* out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (fn < 3) {
    if (i < lft::N_SPECIAL) out[i] = lf_bits(libm1(fn, native, lft::special(i)));
  } else if (i < lft::N_SPECIAL * lft::N_SPECIAL) {
    out[i] = lf_bits(libm2(fn - 3, native, lft::special(i / lft::N_SPECIAL), lft::special(i % lft::N_SPECIAL)));
  }
}

// the f64 functions of the auto-flip path (libm.cuh): op 0 atan2_64, 1 hypot64 over the cases of the random stream
extern "C" __global__ void libm_f64(int op, uint64_t base, uint64_t n, uint64_t* out) {
  uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  double a = lft::f64_case(lft::mix((base + i) * 2)), b = lft::f64_case(lft::mix((base + i) * 2 + 1));
  double r = op == 0 ? lf::atan2_64(a, b) : lf::hypot64(a, b);
  out[i] = (uint64_t)__double_as_longlong(r);
}

// ---- lf_div (hd.cuh, tests/div.rs): the fast path with its own slow path against CUDA's __fdiv_rn ----

// A pair of bit patterns of the random stream, by mode: 0 all bits random; 1 exponents at and around the fast path's window and
// edge mantissas; 2 inside the window; 3 zero, subnormal, normal, infinity and NaN (random payload) operands; 4 divisors with the
// all-ones mantissa at every exponent.
__device__ __forceinline__ uint64_t div_sm64(uint64_t& s) {
  uint64_t z = (s += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}
__device__ __forceinline__ uint32_t div_mant(uint64_t r, bool edges) {
  if (edges) {
    switch ((r >> 40) & 15) {
      case 0: return 0;
      case 1: return 1;
      case 2: return 0x7fffff;
      case 3: return 0x400000;
      case 4: return 0x7ffffe;
      case 5: return 2;
      case 6: return 0x400001;
      case 7: return 0x3fffff;
      default: break;
    }
  }
  return (uint32_t)(r & 0x7fffff);
}
__device__ __forceinline__ void div_pair(int mode, uint64_t i, uint64_t seed, uint32_t* pa, uint32_t* pb) {
  uint64_t s = seed ^ (i * 0xD2B74407B1CE6E93ull);
  uint64_t r1 = div_sm64(s), r2 = div_sm64(s), r3 = div_sm64(s);
  uint32_t sa = (uint32_t)(r3 >> 63) << 31, sb = (uint32_t)((r3 >> 62) & 1) << 31;
  if (mode == 0) {
    *pa = (uint32_t)r1;
    *pb = (uint32_t)(r1 >> 32);
  } else if (mode == 2) {
    uint32_t ea = 30 + (uint32_t)(r1 % 224), eb = 3 + (uint32_t)((r1 >> 20) % 250);
    *pa = sa | (ea << 23) | div_mant(r2, true);
    *pb = sb | (eb << 23) | div_mant(r2 >> 24, false);
  } else if (mode == 1) {
    const int edges[16] = {0, 1, 2, 3, 4, 29, 30, 31, 32, 252, 253, 254, 255, 100, 127, 128};
    uint32_t ea = edges[(r1 >> 8) & 15], eb = edges[(r1 >> 16) & 15];
    if ((r1 & 3) == 0) {  // a quotient exponent near the window's ends
      ea = 127 + (uint32_t)((r1 >> 24) % 140) - 20;
      eb = ea + (uint32_t)((r1 >> 40) % 14) - 7 + ((r1 >> 5) & 1) * 100 - ((r1 >> 6) & 1) * 100 + ((r1 >> 7) & 1) * 20;
      ea &= 255;
      eb &= 255;
    }
    *pa = sa | (ea << 23) | div_mant(r2, true);
    *pb = sb | (eb << 23) | div_mant(r2 >> 24, true);
  } else if (mode == 4) {  // every divisor exponent with the all-ones mantissa (the class the fast path leaves to lf_div_slow)
    uint32_t eb = 1 + (uint32_t)(r1 % 254), ea = 1 + (uint32_t)((r1 >> 20) % 254);
    if ((r1 & 1) == 0) ea = (eb + (uint32_t)((r1 >> 40) % 64) + 223) % 254 + 1;  // half near eb: quotients inside the window
    *pa = sa | (ea << 23) | div_mant(r2, true);
    *pb = sb | (eb << 23) | 0x7fffffu;
  } else {
    uint32_t ca = (r1 >> 8) % 6, cb = (r1 >> 16) % 6;
    uint32_t m1 = (uint32_t)r2 & 0x7fffff, m2 = (uint32_t)(r2 >> 32) & 0x7fffff;
    *pa = ca == 0 ? sa : ca == 1 ? sa | m1 : ca == 2 ? sa | (((uint32_t)(r1 >> 24) % 254 + 1) << 23) | m1 : ca == 3 ? sa | 0x7f800000u : ca == 4 ? sa | 0x7f800000u | (m1 ? m1 : 1) : sa | (m1 & 0xfffff);
    *pb = cb == 0 ? sb : cb == 1 ? sb | m2 : cb == 2 ? sb | (((uint32_t)(r1 >> 40) % 254 + 1) << 23) | m2 : cb == 3 ? sb | 0x7f800000u : cb == 4 ? sb | 0x7f800000u | (m2 ? m2 : 1) : sb | (m2 & 0xfffff);
  }
}

// counts the pairs of n (mode, seed) where lf_div differs bitwise from __fdiv_rn (a NaN with another payload counts), and keeps the
// first such pair in first[0..2)
extern "C" __global__ void div_stream(int mode, uint64_t seed, uint64_t n, unsigned long long* bad, uint32_t* first) {
  uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
  unsigned long long nb = 0;
  for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    uint32_t ia, ib;
    div_pair(mode, i, seed, &ia, &ib);
    float a = lf_from_bits(ia), b = lf_from_bits(ib);
    if (lf_bits(lf_div(a, b)) != lf_bits(__fdiv_rn(a, b))) {
      if (nb++ == 0 && atomicCAS(bad, 0ull, 0ull) == 0ull) {
        first[0] = ia;
        first[1] = ib;
      }
    }
  }
  if (nb) atomicAdd(bad, nb);
}

// out[i] = bits of lf_div(a[i], b[i]) for explicit operands (the host compares them with its own IEEE division)
extern "C" __global__ void div_pairs(const uint32_t* a, const uint32_t* b, uint32_t n, uint32_t* out) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = lf_bits(lf_div(lf_from_bits(a[i]), lf_from_bits(b[i])));
}
