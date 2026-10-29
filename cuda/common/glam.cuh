// glam 0.34.1 (x86_64 SSE2 paths, no FMA) reproduced bit for bit for the GPU simulator.
// Every function mirrors the order and rounding of the glam source named beside it; paths are under
// glam-0.34.1/src/f32/ (sse2/vec3a.rs = "vec3a.rs:LINE", vec4.rs, quat.rs, mat3a.rs, affine3a.rs, and ../sse2.rs for the
// dot helpers). Vec3A and Quat keep no w lane: no result lane depends on it, so the SSE garbage lane is not modelled.
//
// INVENTORY. V3 = RocketSim V3 at ad8b38c, rocketsim/src, the 69 files that use glam, minus the GJK/EPA, Dropshot,
// Heatseeker, serde and dump code the GPU port does not carry. Counts are textual hits (25 October 2026); a name shared
// with f32 (abs 49, min 56, max 64, clamp 47, signum 13, normalize, sqrt) counts both, so it bounds the glam use above.
//   Vec3A  new 153, splat 22, consts (ZERO 124, X/Y/Z 25, ONE 4, NEG_* 10, MIN/MAX 2), select 8; + - * / as (v,v), (v,f32),
//          (f32,v) and the assign forms, unary -; dot 148, cross 42, length 23, length_squared 37, normalize 15,
//          normalize_or_zero 16, normalize_or 2, try_normalize 4, distance_squared 8, lerp 1, abs, signum, min, max, clamp,
//          min_element 1, max_element 5, max_position 4, cmplt 7, cmple 12, cmpge 14, cmpgt 1, cmpeq 3, is_finite_mask 2,
//          as_ivec3 1, as_uvec3 2, as_usizevec3 2, field reads x y z, indexing
//   Vec4   new 22, splat 11, ONE/ZERO/MAX, + - * / (v,v), unary -, abs, min, max, max_element 4, max_position 4, cmpge,
//          cmpgt, cmple, cmplt, select 4, is_finite_mask 1, BVec4A & and !
//   Mat3A  IDENTITY 12, from_cols 2, transpose 10, Mat3A * Vec3A (mul_vec3a), mul_transpose_vec3a 15, Mat3A * Mat3A,
//          from_quat 1, x_axis / y_axis / z_axis reads and Vec3A *= writes
//   Affine3A  IDENTITY 4, Affine3A * Affine3A 4, transform_point3a 25, transform_vector3a 2, matrix3 / translation fields
//   Quat   Quat * Quat 1, normalize 1, from_vec4 1, from_mat3a 2, from_xyzw 1 (mul_vec3a: asked for, V3 does not call it)
//   BVec3A / BVec4A  any, all, bitmask, & | !, into [bool; N]
//   Float <-> int: the Rust saturating casts behind as_ivec3 / as_uvec3 / as_usizevec3, and int -> f32 for as_vec3a 3
//   V3's AffineExt::transpose and inv_x_form (bullet/linear_math/affine_ext.rs) are built from these.
// Every item above is one or more of the 84 ops of simtest/ops.cuh, each tested against glam 0.34.1 and on the GPU.
// NOT HERE (separate tasks): sin / cos / atan2 / tan / powf and the libm calls (Quat::from_axis_angle_simd,
// Mat3A::from_euler ZYX, integrate_trans, Angle); f32::floor / ceil / round. Debug-only asserts (is_normalized,
// glam_assert) are dropped: V3 compiles them out in release. Integer vector arithmetic is exact by construction.
#pragma once
#include "common/hd.cuh"

struct Vec3A { float x, y, z; };
struct Vec4 { float x, y, z, w; };
struct Quat { float x, y, z, w; };
struct Mat3A { Vec3A x_axis, y_axis, z_axis; };
struct Affine3A { Mat3A matrix3; Vec3A translation; };
struct BVec3A { uint32_t x, y, z; };  // lanes all-ones or zero, as the SSE compare masks
struct BVec4A { uint32_t x, y, z, w; };

// ---- scalar helpers (global, lf_ / sse_ / rs_ prefixed) ---------------------------------------------------------------
LF_HDI uint32_t lf_all1(bool c) { return c ? 0xFFFFFFFFu : 0u; }
LF_HDI float lf_neg(float a) { return lf_from_bits(lf_bits(a) ^ 0x80000000u); }
LF_HDI float lf_abs(float a) { return lf_from_bits(lf_bits(a) & 0x7FFFFFFFu); }
LF_HDI bool lf_isnan(float a) { return (lf_bits(a) & 0x7FFFFFFFu) > 0x7F800000u; }
LF_HDI bool lf_isfinite(float a) { return (lf_bits(a) & 0x7FFFFFFFu) < 0x7F800000u; }
// _mm_min_ps / _mm_max_ps: the SECOND operand is returned when either is NaN, and when both are zero.
LF_HDI float sse_min(float a, float b) { return a < b ? a : b; }
LF_HDI float sse_max(float a, float b) { return a > b ? a : b; }
// Rust f32::min / max / clamp / signum (std semantics, used by V3 on scalars). NaN is ignored by min and max; Rust leaves
// the order of +0 and -0 unspecified, so the tests skip that tie.
LF_HDI float rs_min(float a, float b) { return lf_isnan(a) ? b : lf_isnan(b) ? a : (a < b ? a : b); }
LF_HDI float rs_max(float a, float b) { return lf_isnan(a) ? b : lf_isnan(b) ? a : (a > b ? a : b); }
LF_HDI float rs_clamp(float x, float lo, float hi) {
  if (x < lo) x = lo;
  if (x > hi) x = hi;
  return x;
}
LF_HDI float rs_signum(float a) { return lf_isnan(a) ? a : lf_from_bits((lf_bits(a) & 0x80000000u) | 0x3F800000u); }
// Rust `as` casts: saturating, NaN -> 0 (vec3a.rs as_ivec3 / as_uvec3 / as_usizevec3 are `self.x as T`).
LF_HDI int32_t lf_f2i(float a) {
  if (lf_isnan(a)) return 0;
  if (a >= 2147483648.0f) return 0x7FFFFFFF;
  if (a <= -2147483648.0f) return (int32_t)0x80000000u;
  return (int32_t)a;
}
LF_HDI uint32_t lf_f2u(float a) {
  if (lf_isnan(a) || a <= 0.0f) return 0u;
  if (a >= 4294967296.0f) return 0xFFFFFFFFu;
  return (uint32_t)a;
}
LF_HDI uint64_t lf_f2ull(float a) {
  if (lf_isnan(a) || a <= 0.0f) return 0ull;
  if (a >= 18446744073709551616.0f) return 0xFFFFFFFFFFFFFFFFull;
  return (uint64_t)a;
}
LF_HDI float lf_sel(uint32_t m, float t, float f) { return lf_from_bits((lf_bits(f) & ~m) | (lf_bits(t) & m)); }

namespace glam {

// ---- BVec ----------------------------------------------------------------------------------------------------------
LF_HDI uint32_t bitmask(BVec3A m) { return (m.x >> 31) | ((m.y >> 31) << 1) | ((m.z >> 31) << 2); }  // bvec3a.rs:72
LF_HDI bool any(BVec3A m) { return bitmask(m) != 0; }
LF_HDI bool all(BVec3A m) { return bitmask(m) == 7; }
LF_HDI uint32_t bitmask(BVec4A m) { return (m.x >> 31) | ((m.y >> 31) << 1) | ((m.z >> 31) << 2) | ((m.w >> 31) << 3); }
LF_HDI bool any(BVec4A m) { return bitmask(m) != 0; }
LF_HDI bool all(BVec4A m) { return bitmask(m) == 15; }
LF_HDI BVec4A operator&(BVec4A a, BVec4A b) { return {a.x & b.x, a.y & b.y, a.z & b.z, a.w & b.w}; }
LF_HDI BVec4A operator|(BVec4A a, BVec4A b) { return {a.x | b.x, a.y | b.y, a.z | b.z, a.w | b.w}; }
LF_HDI BVec4A operator~(BVec4A a) { return {~a.x, ~a.y, ~a.z, ~a.w}; }
LF_HDI BVec3A operator&(BVec3A a, BVec3A b) { return {a.x & b.x, a.y & b.y, a.z & b.z}; }
LF_HDI BVec3A operator|(BVec3A a, BVec3A b) { return {a.x | b.x, a.y | b.y, a.z | b.z}; }
LF_HDI BVec3A operator~(BVec3A a) { return {~a.x, ~a.y, ~a.z}; }

// ---- Vec3A ---------------------------------------------------------------------------------------------------------
LF_HDI Vec3A vec3a(float x, float y, float z) { return {x, y, z}; }
LF_HDI Vec3A splat3(float v) { return {v, v, v}; }
#define LF_V3(op) {op(a.x, b.x), op(a.y, b.y), op(a.z, b.z)}
LF_HDI Vec3A operator+(Vec3A a, Vec3A b) { return LF_V3(lf_add); }  // vec3a.rs:1796 (_mm_add_ps)
LF_HDI Vec3A operator-(Vec3A a, Vec3A b) { return LF_V3(lf_sub); }  // :1920
LF_HDI Vec3A operator*(Vec3A a, Vec3A b) { return LF_V3(lf_mul); }  // :1672
LF_HDI Vec3A operator/(Vec3A a, Vec3A b) { return LF_V3(lf_div); }  // :1548
LF_HDI Vec3A operator*(Vec3A a, float s) { return a * splat3(s); }  // :1718
LF_HDI Vec3A operator*(float s, Vec3A a) { return splat3(s) * a; }  // :1764
LF_HDI Vec3A operator/(Vec3A a, float s) { return a / splat3(s); }  // :1594
LF_HDI Vec3A operator/(float s, Vec3A a) { return splat3(s) / a; }  // :1640
LF_HDI Vec3A operator-(Vec3A a) { return {lf_neg(a.x), lf_neg(a.y), lf_neg(a.z)}; }  // :2225 (sign-bit xor)
LF_HDI Vec3A& operator+=(Vec3A& a, Vec3A b) { a = a + b; return a; }
LF_HDI Vec3A& operator-=(Vec3A& a, Vec3A b) { a = a - b; return a; }
LF_HDI Vec3A& operator*=(Vec3A& a, Vec3A b) { a = a * b; return a; }
LF_HDI Vec3A& operator*=(Vec3A& a, float s) { a = a * s; return a; }
LF_HDI Vec3A& operator/=(Vec3A& a, float s) { a = a / s; return a; }
LF_HDI float at(Vec3A a, int i) { return i == 0 ? a.x : i == 1 ? a.y : a.z; }

// sse2.rs dot3_in_x: (x*x' + y*y') + z*z', an add_ss chain, never fused.
LF_HDI float dot(Vec3A a, Vec3A b) { return lf_add(lf_add(lf_mul(a.x, b.x), lf_mul(a.y, b.y)), lf_mul(a.z, b.z)); }
// vec3a.rs:277: (a.zxy * b - b.zxy * a).zxy, i.e. lane i is  a.j*b.k - b.j*a.k  (the order inside a product is immaterial).
LF_HDI Vec3A cross(Vec3A a, Vec3A b) {
  return {lf_sub(lf_mul(a.y, b.z), lf_mul(b.y, a.z)), lf_sub(lf_mul(a.z, b.x), lf_mul(b.z, a.x)),
          lf_sub(lf_mul(a.x, b.y), lf_mul(b.x, a.y))};
}
LF_HDI float length_squared(Vec3A a) { return dot(a, a); }                        // :608
LF_HDI float length(Vec3A a) { return lf_sqrt(dot(a, a)); }                       // :589
LF_HDI float length_recip(Vec3A a) { return lf_div(1.0f, lf_sqrt(dot(a, a))); }   // :617
LF_HDI Vec3A normalize(Vec3A a) { return a / splat3(lf_sqrt(dot(a, a))); }        // :674 divides by the length
LF_HDI Vec3A normalize_or(Vec3A a, Vec3A fallback) {                              // :710 multiplies by the reciprocal
  float rcp = length_recip(a);
  return (lf_isfinite(rcp) && rcp > 0.0f) ? a * rcp : fallback;
}
LF_HDI Vec3A normalize_or_zero(Vec3A a) { return normalize_or(a, splat3(0.0f)); }  // :727
LF_HDI bool try_normalize(Vec3A a, Vec3A* out) {                                   // :692
  float rcp = length_recip(a);
  if (lf_isfinite(rcp) && rcp > 0.0f) {
    *out = a * rcp;
    return true;
  }
  return false;
}
LF_HDI float distance_squared(Vec3A a, Vec3A b) { return length_squared(a - b); }      // :634
LF_HDI Vec3A lerp(Vec3A a, Vec3A b, float s) { return a * lf_sub(1.0f, s) + b * s; }  // :1012 (not the monotonic form)
LF_HDI Vec3A abs(Vec3A a) { return {lf_abs(a.x), lf_abs(a.y), lf_abs(a.z)}; }         // :499
LF_HDI Vec3A min(Vec3A a, Vec3A b) { return {sse_min(a.x, b.x), sse_min(a.y, b.y), sse_min(a.z, b.z)}; }  // :300
LF_HDI Vec3A max(Vec3A a, Vec3A b) { return {sse_max(a.x, b.x), sse_max(a.y, b.y), sse_max(a.z, b.z)}; }  // :312
LF_HDI Vec3A clamp(Vec3A a, Vec3A lo, Vec3A hi) { return min(max(a, lo), hi); }  // :329 max first, then min
// :342 / :359: min(v, v.zzxx) then min(lane0, lane1) = ((x,z) , (y,z)) -> NaN operands resolve as in the SSE chain.
LF_HDI float min_element(Vec3A a) { return sse_min(sse_min(a.x, a.z), sse_min(a.y, a.z)); }
LF_HDI float max_element(Vec3A a) { return sse_max(sse_max(a.x, a.z), sse_max(a.y, a.z)); }
LF_HDI int min_position(Vec3A a) {  // :372
  float m = a.x;
  int i = 0;
  if (a.y < m) { m = a.y; i = 1; }
  if (a.z < m) i = 2;
  return i;
}
LF_HDI int max_position(Vec3A a) {  // :389
  float m = a.x;
  int i = 0;
  if (a.y > m) { m = a.y; i = 1; }
  if (a.z > m) i = 2;
  return i;
}
LF_HDI BVec3A cmpeq(Vec3A a, Vec3A b) { return {lf_all1(a.x == b.x), lf_all1(a.y == b.y), lf_all1(a.z == b.z)}; }  // :437
LF_HDI BVec3A cmpge(Vec3A a, Vec3A b) { return {lf_all1(a.x >= b.x), lf_all1(a.y >= b.y), lf_all1(a.z >= b.z)}; }  // :459
LF_HDI BVec3A cmpgt(Vec3A a, Vec3A b) { return {lf_all1(a.x > b.x), lf_all1(a.y > b.y), lf_all1(a.z > b.z)}; }     // :470
LF_HDI BVec3A cmple(Vec3A a, Vec3A b) { return {lf_all1(a.x <= b.x), lf_all1(a.y <= b.y), lf_all1(a.z <= b.z)}; }  // :481
LF_HDI BVec3A cmplt(Vec3A a, Vec3A b) { return {lf_all1(a.x < b.x), lf_all1(a.y < b.y), lf_all1(a.z < b.z)}; }     // :492
LF_HDI Vec3A select(BVec3A m, Vec3A t, Vec3A f) { return {lf_sel(m.x, t.x, f.x), lf_sel(m.y, t.y, f.y), lf_sel(m.z, t.z, f.z)}; }  // :136
LF_HDI BVec3A is_nan_mask(Vec3A a) { return {lf_all1(lf_isnan(a.x)), lf_all1(lf_isnan(a.y)), lf_all1(lf_isnan(a.z))}; }  // :581
LF_HDI BVec3A is_finite_mask(Vec3A a) {  // :565
  return {lf_all1(lf_abs(a.x) < INFINITY), lf_all1(lf_abs(a.y) < INFINITY), lf_all1(lf_abs(a.z) < INFINITY)};
}
// :510  (sign bit | 1.0) per lane; NaN lanes pass through.
LF_HDI Vec3A signum(Vec3A a) {
  Vec3A r = {lf_from_bits((lf_bits(a.x) & 0xBF800000u) | 0x3F800000u), lf_from_bits((lf_bits(a.y) & 0xBF800000u) | 0x3F800000u),
             lf_from_bits((lf_bits(a.z) & 0xBF800000u) | 0x3F800000u)};
  return select(is_nan_mask(a), a, r);
}

// ---- Vec4 ----------------------------------------------------------------------------------------------------------
LF_HDI Vec4 vec4(float x, float y, float z, float w) { return {x, y, z, w}; }
LF_HDI Vec4 splat4(float v) { return {v, v, v, v}; }
#define LF_V4(op) {op(a.x, b.x), op(a.y, b.y), op(a.z, b.z), op(a.w, b.w)}
LF_HDI Vec4 operator+(Vec4 a, Vec4 b) { return LF_V4(lf_add); }
LF_HDI Vec4 operator-(Vec4 a, Vec4 b) { return LF_V4(lf_sub); }
LF_HDI Vec4 operator*(Vec4 a, Vec4 b) { return LF_V4(lf_mul); }
LF_HDI Vec4 operator/(Vec4 a, Vec4 b) { return LF_V4(lf_div); }
LF_HDI Vec4 operator-(Vec4 a) { return {lf_neg(a.x), lf_neg(a.y), lf_neg(a.z), lf_neg(a.w)}; }
LF_HDI Vec4 abs(Vec4 a) { return {lf_abs(a.x), lf_abs(a.y), lf_abs(a.z), lf_abs(a.w)}; }  // vec4.rs:475
LF_HDI Vec4 min(Vec4 a, Vec4 b) { return LF_V4(sse_min); }                                 // :268
LF_HDI Vec4 max(Vec4 a, Vec4 b) { return LF_V4(sse_max); }                                 // :280
// sse2.rs dot4_into_m128: (x*x' + z*z') + (y*y' + w*w').
LF_HDI float dot(Vec4 a, Vec4 b) {
  return lf_add(lf_add(lf_mul(a.x, b.x), lf_mul(a.z, b.z)), lf_add(lf_mul(a.y, b.y), lf_mul(a.w, b.w)));
}
LF_HDI float length(Vec4 a) { return lf_sqrt(dot(a, a)); }                                  // :565
LF_HDI Vec4 normalize(Vec4 a) { return a / splat4(lf_sqrt(dot(a, a))); }                    // :652
// :310 / :327: v op v.zwzw (lanes x,z / y,w), then lane0 op lane1.
LF_HDI float min_element(Vec4 a) { return sse_min(sse_min(a.x, a.z), sse_min(a.y, a.w)); }
LF_HDI float max_element(Vec4 a) { return sse_max(sse_max(a.x, a.z), sse_max(a.y, a.w)); }
LF_HDI int max_position(Vec4 a) {  // :361
  float m = a.x;
  int i = 0;
  if (a.y > m) { m = a.y; i = 1; }
  if (a.z > m) { m = a.z; i = 2; }
  if (a.w > m) i = 3;
  return i;
}
LF_HDI BVec4A cmpge(Vec4 a, Vec4 b) { return {lf_all1(a.x >= b.x), lf_all1(a.y >= b.y), lf_all1(a.z >= b.z), lf_all1(a.w >= b.w)}; }  // :435
LF_HDI BVec4A cmpgt(Vec4 a, Vec4 b) { return {lf_all1(a.x > b.x), lf_all1(a.y > b.y), lf_all1(a.z > b.z), lf_all1(a.w > b.w)}; }     // :446
LF_HDI BVec4A cmple(Vec4 a, Vec4 b) { return {lf_all1(a.x <= b.x), lf_all1(a.y <= b.y), lf_all1(a.z <= b.z), lf_all1(a.w <= b.w)}; }  // :457
LF_HDI BVec4A cmplt(Vec4 a, Vec4 b) { return {lf_all1(a.x < b.x), lf_all1(a.y < b.y), lf_all1(a.z < b.z), lf_all1(a.w < b.w)}; }     // :468
LF_HDI Vec4 select(BVec4A m, Vec4 t, Vec4 f) {  // :138
  return {lf_sel(m.x, t.x, f.x), lf_sel(m.y, t.y, f.y), lf_sel(m.z, t.z, f.z), lf_sel(m.w, t.w, f.w)};
}
LF_HDI BVec4A is_finite_mask(Vec4 a) {  // :541
  return {lf_all1(lf_abs(a.x) < INFINITY), lf_all1(lf_abs(a.y) < INFINITY), lf_all1(lf_abs(a.z) < INFINITY),
          lf_all1(lf_abs(a.w) < INFINITY)};
}

// ---- Mat3A (column major; every product-sum is (x*c0 + y*c1) + z*c2, glam never fuses it) ----------------------------
LF_HDI Mat3A mat3a_from_cols(Vec3A x, Vec3A y, Vec3A z) { return {x, y, z}; }
LF_HDI Mat3A mat3a_identity() { return {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}; }  // mat3a.rs IDENTITY
LF_HDI Vec3A operator*(Mat3A m, Vec3A v) {                                    // mat3a.rs mul_vec3a
  Vec3A r = m.x_axis * splat3(v.x);
  r = r + m.y_axis * splat3(v.y);
  r = r + m.z_axis * splat3(v.z);
  return r;
}
LF_HDI Vec3A mul_transpose_vec3a(Mat3A m, Vec3A v) { return {dot(m.x_axis, v), dot(m.y_axis, v), dot(m.z_axis, v)}; }
LF_HDI Mat3A operator*(Mat3A a, Mat3A b) { return {a * b.x_axis, a * b.y_axis, a * b.z_axis}; }  // Mul<Mat3A>
LF_HDI Mat3A transpose(Mat3A m) {                                                                  // shuffle transpose
  return {{m.x_axis.x, m.y_axis.x, m.z_axis.x}, {m.x_axis.y, m.y_axis.y, m.z_axis.y}, {m.x_axis.z, m.y_axis.z, m.z_axis.z}};
}
LF_HDI Mat3A mat3a_from_quat(Quat q) {  // mat3a.rs from_quat
  float x2 = lf_add(q.x, q.x), y2 = lf_add(q.y, q.y), z2 = lf_add(q.z, q.z);
  float xx = lf_mul(q.x, x2), xy = lf_mul(q.x, y2), xz = lf_mul(q.x, z2);
  float yy = lf_mul(q.y, y2), yz = lf_mul(q.y, z2), zz = lf_mul(q.z, z2);
  float wx = lf_mul(q.w, x2), wy = lf_mul(q.w, y2), wz = lf_mul(q.w, z2);
  return {{lf_sub(1.0f, lf_add(yy, zz)), lf_add(xy, wz), lf_sub(xz, wy)},
          {lf_sub(xy, wz), lf_sub(1.0f, lf_add(xx, zz)), lf_add(yz, wx)},
          {lf_add(xz, wy), lf_sub(yz, wx), lf_sub(1.0f, lf_add(xx, yy))}};
}

// ---- Affine3A ------------------------------------------------------------------------------------------------------
LF_HDI Affine3A affine3a_identity() { return {mat3a_identity(), {0, 0, 0}}; }
LF_HDI Vec3A transform_point3a(const Affine3A& t, Vec3A p) { return t.matrix3 * p + t.translation; }  // affine3a.rs:214
LF_HDI Vec3A transform_vector3a(const Affine3A& t, Vec3A v) { return t.matrix3 * v; }                  // :218
LF_HDI Affine3A operator*(const Affine3A& a, const Affine3A& b) {                                      // :322
  return {a.matrix3 * b.matrix3, a.matrix3 * b.translation + a.translation};
}
// V3 AffineExt (bullet/linear_math/affine_ext.rs), composed of the above.
LF_HDI Affine3A affine_transpose(const Affine3A& t) {
  Mat3A m = transpose(t.matrix3);
  return {m, m * (-t.translation)};
}
LF_HDI Vec3A affine_inv_x_form(const Affine3A& t, Vec3A v) { return mul_transpose_vec3a(t.matrix3, v - t.translation); }

// ---- Quat ----------------------------------------------------------------------------------------------------------
LF_HDI Quat quat_from_xyzw(float x, float y, float z, float w) { return {x, y, z, w}; }
LF_HDI Quat quat_from_vec4(Vec4 v) { return {v.x, v.y, v.z, v.w}; }
LF_HDI Vec4 to_vec4(Quat q) { return {q.x, q.y, q.z, q.w}; }
LF_HDI Quat normalize(Quat q) { return quat_from_vec4(normalize(to_vec4(q))); }  // quat.rs:549
// quat.rs:794 mul_quat (rtm). The +-1 control multiplies are exact; the adds are result0 = (A*c1) + lw*r,
// result1 = (C*c2) + (B*c3), out = result0 + result1, with no FMA (m128_mul_add falls back to mul then add).
LF_HDI Quat operator*(Quat l, Quat r) {
  float lx = l.x, ly = l.y, lz = l.z, lw = l.w;
  float r0x = lf_add(lf_mul(lx, r.w), lf_mul(lw, r.x));
  float r0y = lf_add(lf_neg(lf_mul(lx, r.z)), lf_mul(lw, r.y));
  float r0z = lf_add(lf_mul(lx, r.y), lf_mul(lw, r.z));
  float r0w = lf_add(lf_neg(lf_mul(lx, r.x)), lf_mul(lw, r.w));
  float r1x = lf_add(lf_neg(lf_mul(lz, r.y)), lf_mul(ly, r.z));
  float r1y = lf_add(lf_mul(lz, r.x), lf_mul(ly, r.w));
  float r1z = lf_add(lf_mul(lz, r.w), lf_neg(lf_mul(ly, r.x)));
  float r1w = lf_add(lf_neg(lf_mul(lz, r.z)), lf_neg(lf_mul(ly, r.y)));
  return {lf_add(r0x, r1x), lf_add(r0y, r1y), lf_add(r0z, r1z), lf_add(r0w, r1w)};
}
LF_HDI Vec3A mul_vec3a(Quat q, Vec3A v) {  // quat.rs:873
  Vec3A b = {q.x, q.y, q.z};
  float w = q.w, b2 = dot(b, b);
  Vec3A t1 = v * splat3(lf_sub(lf_mul(w, w), b2));
  Vec3A t2 = b * splat3(lf_mul(dot(v, b), 2.0f));
  Vec3A t3 = cross(b, v) * splat3(lf_mul(w, 2.0f));
  return (t1 + t2) + t3;
}
// quat.rs:220 from_rotation_axes (DirectXMath XMQuaternionRotationMatrix); from_mat3a feeds it the three columns.
LF_HDI Quat quat_from_mat3a(Mat3A m) {
  float m00 = m.x_axis.x, m01 = m.x_axis.y, m02 = m.x_axis.z;
  float m10 = m.y_axis.x, m11 = m.y_axis.y, m12 = m.y_axis.z;
  float m20 = m.z_axis.x, m21 = m.z_axis.y, m22 = m.z_axis.z;
  if (m22 <= 0.0f) {
    float dif10 = lf_sub(m11, m00), omm22 = lf_sub(1.0f, m22);
    if (dif10 <= 0.0f) {
      float four_xsq = lf_sub(omm22, dif10), inv4x = lf_div(0.5f, lf_sqrt(four_xsq));
      return {lf_mul(four_xsq, inv4x), lf_mul(lf_add(m01, m10), inv4x), lf_mul(lf_add(m02, m20), inv4x),
              lf_mul(lf_sub(m12, m21), inv4x)};
    }
    float four_ysq = lf_add(omm22, dif10), inv4y = lf_div(0.5f, lf_sqrt(four_ysq));
    return {lf_mul(lf_add(m01, m10), inv4y), lf_mul(four_ysq, inv4y), lf_mul(lf_add(m12, m21), inv4y),
            lf_mul(lf_sub(m20, m02), inv4y)};
  }
  float sum10 = lf_add(m11, m00), opm22 = lf_add(1.0f, m22);
  if (sum10 <= 0.0f) {
    float four_zsq = lf_sub(opm22, sum10), inv4z = lf_div(0.5f, lf_sqrt(four_zsq));
    return {lf_mul(lf_add(m02, m20), inv4z), lf_mul(lf_add(m12, m21), inv4z), lf_mul(four_zsq, inv4z),
            lf_mul(lf_sub(m01, m10), inv4z)};
  }
  float four_wsq = lf_add(opm22, sum10), inv4w = lf_div(0.5f, lf_sqrt(four_wsq));
  return {lf_mul(lf_sub(m12, m21), inv4w), lf_mul(lf_sub(m20, m02), inv4w), lf_mul(lf_sub(m01, m10), inv4w),
          lf_mul(four_wsq, inv4w)};
}

}  // namespace glam
