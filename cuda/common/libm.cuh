// The transcendentals V3 calls (grep of rocketsim/src at ad8b38c: atan2 6, powi 5, sin 3, tan 2, sin_cos 2, cos 2, powf 1,
// f64 hypot 1). Physics reaches them through the `lf::` names only, and the library behind the names is a compile-time switch:
//   LF_LIBM_CR (the default, host and device alike): correctly rounded binary32 sin, cos, tan, atan2 and pow (libm_cr.cuh, ported
//     from CORE-MATH). The correctly rounded value is unique, so the host and the device agree bit for bit on every platform,
//     both GPU architectures and every toolkit version (CUDA's sinf/cosf are 1-2 ulp off the
//     C runtime's in 0.2% of integrate_trans states, and Windows' sinf is itself not correctly rounded).
//   LF_LIBM_PLATFORM: on the host the platform C library exactly as V3's Rust reaches it (sinf, cosf, tanf, atan2f, powf), for
//     the porting verification against V3 on one platform. The device is not built this way: a device pass under this switch
//     still uses the correctly rounded versions.
//   LF_LIBM_MIXED (test only, host): every function chooses per call, from the thread-local bit mask `libm_platform_mask`
//     (1 sin, 2 cos, 4 tan, 8 atan2, 16 powf, 32 atan2_64, 64 hypot64; a set bit is the platform library), so the phase driver can
//     attribute a difference to the call that causes it.
// The f64 functions of the auto-flip path (car_pre_tick.cuh): hypot64 is CORE-MATH's correctly rounded hypot; atan2_64 is a
// double-double algorithm, the same code on host and device (so bit-identical) but not correctly rounded (libm_cr.cuh). The
// 16384-entry f64 sine table of V3's rebuild_row1 is not computed here: the host fills StaticData.table_sin once with the
// platform f64 sin (dump.rs), so on any platform it equals that platform's V3; only the LF_LIBM_PLATFORM build computes it
// directly.
#pragma once
#include "hd.cuh"
#include "libm_cr.cuh"

#if defined(LF_LIBM_CR) + defined(LF_LIBM_PLATFORM) + defined(LF_LIBM_MIXED) > 1
#error "define at most one of LF_LIBM_CR, LF_LIBM_PLATFORM, LF_LIBM_MIXED"
#endif
#if !defined(LF_LIBM_CR) && !defined(LF_LIBM_PLATFORM) && !defined(LF_LIBM_MIXED)
#define LF_LIBM_CR 1
#endif

#if defined(LF_LIBM_MIXED)
inline thread_local unsigned libm_platform_mask = 0;
#define LF_LIBM_PICK(bit, platform, cr) ((libm_platform_mask & (bit)) ? (platform) : (cr))
#elif defined(LF_LIBM_PLATFORM) && !defined(__CUDA_ARCH__)
#define LF_LIBM_PICK(bit, platform, cr) (platform)
#else
#define LF_LIBM_PICK(bit, platform, cr) (cr)
#endif

namespace lf {

LF_HDI float sin(float x) { return LF_LIBM_PICK(1, ::sinf(x), lfcr::sinf_cr(x)); }
LF_HDI float cos(float x) { return LF_LIBM_PICK(2, ::cosf(x), lfcr::cosf_cr(x)); }
LF_HDI float tan(float x) { return LF_LIBM_PICK(4, ::tanf(x), lfcr::tanf_cr(x)); }
LF_HDI float atan2(float y, float x) { return LF_LIBM_PICK(8, ::atan2f(y, x), lfcr::atan2f_cr(y, x)); }
LF_HDI float powf(float x, float y) { return LF_LIBM_PICK(16, ::powf(x, y), lfcr::powf_cr(x, y)); }

// Rust's `sin_cos` is two separate calls on every target V3 builds for (std: `(self.sin(), self.cos())`), not sincosf.
LF_HDI void sin_cos(float x, float* s, float* c) {
    *s = sin(x);
    *c = cos(x);
}

// `f32::powi` is LLVM's powi: binary exponentiation, the square-and-multiply order of compiler-builtins' `__powisf2`
// (LLVM expands a constant exponent inline in the same order). Uses the same correctly rounded multiplies as everything else.
LF_HDI float powi(float a, int b) {
    bool recip = b < 0;
    unsigned pow = b < 0 ? 0u - (unsigned)b : (unsigned)b;
    float mul = 1.0f;
    for (;;) {
        if (pow & 1u) mul = lf_mul(mul, a);
        pow >>= 1;
        if (pow == 0) break;
        a = lf_mul(a, a);
    }
    return recip ? lf_div(1.0f, mul) : mul;
}

// f64, for the rare auto-flip path only (fp64 is 1/64 rate on GeForce, irrelevant there).
LF_HDI double atan2_64(double y, double x) { return LF_LIBM_PICK(32, ::atan2(y, x), lfcr::atan2_64_cr(y, x)); }
LF_HDI double hypot64(double x, double y) { return LF_LIBM_PICK(64, ::hypot(x, y), lfcr::hypot64_cr(x, y)); }

}  // namespace lf
