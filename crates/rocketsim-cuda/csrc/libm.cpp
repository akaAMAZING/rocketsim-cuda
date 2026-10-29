// Host side of the libm tests (tests/libm_cr.rs): the correctly rounded functions (cuda/common/libm_cr.cuh) and, beside them,
// the platform C library, over an index range so that the device kernels (cuda/simtest/simtest.cu) and these draw the same
// cases without moving them. Single-threaded; the test splits the range over threads.
#include <math.h>
#include "common/libm.cuh"
#include "common/libm_cr.cuh"
#include "simtest/libm_inputs.cuh"

static float cr1(int fn, float x) {
  return fn == 0 ? lfcr::sinf_cr(x) : fn == 1 ? lfcr::cosf_cr(x) : lfcr::tanf_cr(x);
}
static float plat1(int fn, float x) { return fn == 0 ? ::sinf(x) : fn == 1 ? ::cosf(x) : ::tanf(x); }
static float cr2(int op, float a, float b) { return op == 0 ? lfcr::atan2f_cr(a, b) : lfcr::powf_cr(a, b); }
static float plat2(int op, float a, float b) { return op == 0 ? ::atan2f(a, b) : ::powf(a, b); }

extern "C" {
// out[i] = f(float(bits base + i)), fn 0 sin, 1 cos, 2 tan; platform != 0 selects the C library
void lf_libm_unary(int fn, int platform, uint64_t base, uint64_t n, uint32_t* out) {
  for (uint64_t i = 0; i < n; ++i) {
    float x = lf_from_bits((uint32_t)(base + i));
    out[i] = lf_bits(platform ? plat1(fn, x) : cr1(fn, x));
  }
}
// out[i] = op(case base + i of the random stream), op 0 atan2f, 1 powf
void lf_libm_stream(int op, int platform, uint64_t base, uint64_t n, uint32_t* out) {
  for (uint64_t i = 0; i < n; ++i) {
    float a, b;
    lft::pair(op, base + i, &a, &b);
    out[i] = lf_bits(platform ? plat2(op, a, b) : cr2(op, a, b));
  }
}
// out[i] = op(case base + i): op 0 atan2_64, 1 hypot64; platform != 0 selects the C library
void lf_libm_f64(int op, int platform, uint64_t base, uint64_t n, uint64_t* out) {
  for (uint64_t i = 0; i < n; ++i) {
    double a = lft::f64_case(lft::mix((base + i) * 2)), b = lft::f64_case(lft::mix((base + i) * 2 + 1));
    double r = op == 0 ? (platform ? ::atan2(a, b) : lfcr::atan2_64_cr(a, b)) : (platform ? ::hypot(a, b) : lfcr::hypot64_cr(a, b));
    memcpy(&out[i], &r, 8);
  }
}

// How often the auto-flip roll sign's integers (car_pre_tick.cuh axis_angle_to_int) change when atan2 and hypot are the port's
// instead of the platform's (V3's). Samples directions of the car's x axis (yaw from (y, x), pitch from (z, hypot(x, y))) and
// general (a0, a1) pairs of a size up to 1 (the roll's dot products). out: n, atan2 doubles changed, atan2 floats changed,
// hypot doubles changed, hypot floats changed, yaw ints, pitch ints, roll ints changed.
static int axis_int(double ang) {
  float u = lf_div(lf_mul((float)ang, 32768.0f), (float)3.14159265358979323846);
  float v = lf_add(lf_mul(u, 2.0f), 0.5f);
  return (int)nearbyintf(v) >> 1;
}
void lf_libm_autoflip(uint64_t base, uint64_t n, uint64_t* out) {
  for (int k = 0; k < 8; ++k) out[k] = 0;
  const double TAU = 6.283185307179586476925286766559;
  for (uint64_t i = 0; i < n; ++i) {
    uint64_t r = lft::mix((base + i) * 3), s = lft::mix(r), t = lft::mix(s);
    double u1 = (double)(r >> 11) / 9007199254740992.0, u2 = (double)(s >> 11) / 9007199254740992.0, u3 = (double)(t >> 11) / 9007199254740992.0;
    double z = 2 * u1 - 1, rho = ::sqrt(1 - z * z), th = TAU * u2;
    float cx = (float)(rho * ::cos(th)), cy = (float)(rho * ::sin(th)), cz = (float)z;
    // yaw
    double ya = lfcr::atan2_64_cr(cy, cx), yp = ::atan2(cy, cx);
    out[0]++;
    out[1] += ya != yp;
    out[2] += (float)ya != (float)yp;
    out[5] += axis_int(ya) != axis_int(yp);
    // pitch
    double hc = lfcr::hypot64_cr(cx, cy), hp = ::hypot(cx, cy);
    out[3] += hc != hp;
    out[4] += (float)hc != (float)hp;
    double pa = lfcr::atan2_64_cr(cz, (float)hc), pp = ::atan2(cz, (float)hp);
    out[1] += pa != pp;
    out[2] += (float)pa != (float)pp;
    out[6] += axis_int(pa) != axis_int(pp);
    // roll: a general pair of size up to 1
    double sc = u3 < 0.5 ? 1.0 : (double)(float)(u3 * 2 - 1);
    float a0 = (float)(sc * ::sin(th * 1.7)), a1 = (float)(sc * ::cos(th * 1.7 + u1));
    double ra = lfcr::atan2_64_cr(a0, a1), rp = ::atan2(a0, a1);
    out[1] += ra != rp;
    out[2] += (float)ra != (float)rp;
    out[7] += axis_int(ra) != axis_int(rp);
  }
}
// the inputs of case idx of the random stream (for reports)
void lf_libm_input(int op, uint64_t idx, float* a, float* b) { lft::pair(op, idx, a, b); }
// the special-value grid: fn 0..2 unary over the specials, 3 atan2f and 4 powf over all pairs
int lf_libm_grid_len(int fn) { return fn < 3 ? lft::N_SPECIAL : lft::N_SPECIAL * lft::N_SPECIAL; }
void lf_libm_grid(int fn, int platform, uint32_t* out) {
  if (fn < 3) {
    for (int i = 0; i < lft::N_SPECIAL; ++i) out[i] = lf_bits(platform ? plat1(fn, lft::special(i)) : cr1(fn, lft::special(i)));
  } else {
    for (int i = 0; i < lft::N_SPECIAL; ++i)
      for (int j = 0; j < lft::N_SPECIAL; ++j) {
        float a = lft::special(i), b = lft::special(j);
        out[i * lft::N_SPECIAL + j] = lf_bits(platform ? plat2(fn - 3, a, b) : cr2(fn - 3, a, b));
      }
  }
}
}
