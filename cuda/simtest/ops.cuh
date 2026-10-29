// The op table of the glam parity test, shared by the host shim (crates/rocketsim-cuda/csrc/shim.cpp) and the device kernel
// (simtest.cu). X(name, floats_in, words_out, body): the body reads `i` (the case's inputs) and writes `o` (u32 words,
// floats as bits). Rust drives both sides with the same random inputs and compares them with glam 0.34.1 itself.
#pragma once
#include "common/glam.cuh"

using namespace glam;

LF_HDI Vec3A ld3(const float* i) { return {i[0], i[1], i[2]}; }
LF_HDI Vec4 ld4(const float* i) { return {i[0], i[1], i[2], i[3]}; }
LF_HDI Quat ldq(const float* i) { return {i[0], i[1], i[2], i[3]}; }
LF_HDI Mat3A ldm(const float* i) { return {ld3(i), ld3(i + 3), ld3(i + 6)}; }
LF_HDI Affine3A lda(const float* i) { return {ldm(i), ld3(i + 9)}; }
LF_HDI void stf(uint32_t* o, float v) { o[0] = lf_bits(v); }
LF_HDI void st3(uint32_t* o, Vec3A v) { o[0] = lf_bits(v.x); o[1] = lf_bits(v.y); o[2] = lf_bits(v.z); }
LF_HDI void st4(uint32_t* o, Vec4 v) { st3(o, {v.x, v.y, v.z}); o[3] = lf_bits(v.w); }
LF_HDI void stq(uint32_t* o, Quat v) { st4(o, to_vec4(v)); }
LF_HDI void stm(uint32_t* o, Mat3A m) { st3(o, m.x_axis); st3(o + 3, m.y_axis); st3(o + 6, m.z_axis); }
LF_HDI void sta(uint32_t* o, const Affine3A& a) { stm(o, a.matrix3); st3(o + 9, a.translation); }
LF_HDI void stb(uint32_t* o, BVec3A m) { o[0] = m.x; o[1] = m.y; o[2] = m.z; }
LF_HDI void stb(uint32_t* o, BVec4A m) { o[0] = m.x; o[1] = m.y; o[2] = m.z; o[3] = m.w; }
LF_HDI bool tie0(float a, float b) { return a == 0.0f && b == 0.0f && lf_bits(a) != lf_bits(b); }  // Rust leaves +-0 order open
LF_HDI Vec3A vsel3(const float* i) { return select(cmplt(ld3(i), ld3(i + 3)), ld3(i + 6), ld3(i + 9)); }

#define LF_OPS(X)                                                                                                          \
  /* scalar: Rust std semantics V3 relies on, and the float -> int casts of as_ivec3 / as_uvec3 / as_usizevec3 */             \
  X(rs_min, 2, 1, if (tie0(i[0], i[1])) o[0] = 0xDEADBEEFu; else stf(o, rs_min(i[0], i[1])))                                \
  X(rs_max, 2, 1, if (tie0(i[0], i[1])) o[0] = 0xDEADBEEFu; else stf(o, rs_max(i[0], i[1])))                                \
  X(rs_clamp, 3, 1, if (!(i[1] <= i[2])) o[0] = 0xDEADBEEFu; else stf(o, rs_clamp(i[0], i[1], i[2])))                       \
  X(rs_signum, 1, 1, stf(o, rs_signum(i[0])))                                                                              \
  X(f2i, 1, 1, o[0] = (uint32_t)lf_f2i(i[0]))                                                                              \
  X(f2u, 1, 1, o[0] = lf_f2u(i[0]))                                                                                        \
  X(f2ull, 1, 2, uint64_t v = lf_f2ull(i[0]); o[0] = (uint32_t)v; o[1] = (uint32_t)(v >> 32))                              \
  X(i2f, 1, 1, stf(o, lf_i2f((int32_t)lf_bits(i[0]))))                                                                     \
  X(u2f, 1, 1, stf(o, lf_u2f(lf_bits(i[0]))))                                                                              \
  X(ull2f, 2, 1, stf(o, lf_ull2f(((uint64_t)lf_bits(i[1]) << 32) | lf_bits(i[0]))))                                        \
  /* Vec3A */                                                                                                              \
  X(v3_add, 6, 3, st3(o, ld3(i) + ld3(i + 3)))                                                                             \
  X(v3_sub, 6, 3, st3(o, ld3(i) - ld3(i + 3)))                                                                             \
  X(v3_mul, 6, 3, st3(o, ld3(i) * ld3(i + 3)))                                                                             \
  X(v3_div, 6, 3, st3(o, ld3(i) / ld3(i + 3)))                                                                             \
  X(v3_mul_s, 4, 3, st3(o, ld3(i) * i[3]))                                                                                 \
  X(s_mul_v3, 4, 3, st3(o, i[3] * ld3(i)))                                                                                 \
  X(v3_div_s, 4, 3, st3(o, ld3(i) / i[3]))                                                                                 \
  X(s_div_v3, 4, 3, st3(o, i[3] / ld3(i)))                                                                                 \
  X(v3_neg, 3, 3, st3(o, -ld3(i)))                                                                                         \
  X(v3_dot, 6, 1, stf(o, dot(ld3(i), ld3(i + 3))))                                                                         \
  X(v3_cross, 6, 3, st3(o, cross(ld3(i), ld3(i + 3))))                                                                     \
  X(v3_length, 3, 1, stf(o, length(ld3(i))))                                                                               \
  X(v3_length_squared, 3, 1, stf(o, length_squared(ld3(i))))                                                               \
  X(v3_length_recip, 3, 1, stf(o, length_recip(ld3(i))))                                                                   \
  X(v3_normalize, 3, 3, st3(o, normalize(ld3(i))))                                                                         \
  X(v3_normalize_or_zero, 3, 3, st3(o, normalize_or_zero(ld3(i))))                                                         \
  X(v3_normalize_or, 6, 3, st3(o, normalize_or(ld3(i), ld3(i + 3))))                                                       \
  X(v3_try_normalize, 3, 4, Vec3A r = {0, 0, 0}; o[3] = try_normalize(ld3(i), &r) ? 1u : 0u; st3(o, r))                    \
  X(v3_distance_squared, 6, 1, stf(o, distance_squared(ld3(i), ld3(i + 3))))                                               \
  X(v3_lerp, 7, 3, st3(o, lerp(ld3(i), ld3(i + 3), i[6])))                                                                 \
  X(v3_abs, 3, 3, st3(o, abs(ld3(i))))                                                                                     \
  X(v3_signum, 3, 3, st3(o, signum(ld3(i))))                                                                               \
  X(v3_min, 6, 3, st3(o, min(ld3(i), ld3(i + 3))))                                                                         \
  X(v3_max, 6, 3, st3(o, max(ld3(i), ld3(i + 3))))                                                                         \
  X(v3_clamp, 9, 3, st3(o, clamp(ld3(i), ld3(i + 3), ld3(i + 6))))                                                         \
  X(v3_min_element, 3, 1, stf(o, min_element(ld3(i))))                                                                     \
  X(v3_max_element, 3, 1, stf(o, max_element(ld3(i))))                                                                     \
  X(v3_min_position, 3, 1, o[0] = (uint32_t)min_position(ld3(i)))                                                          \
  X(v3_max_position, 3, 1, o[0] = (uint32_t)max_position(ld3(i)))                                                          \
  X(v3_cmpeq, 6, 3, stb(o, cmpeq(ld3(i), ld3(i + 3))))                                                                     \
  X(v3_cmpge, 6, 3, stb(o, cmpge(ld3(i), ld3(i + 3))))                                                                     \
  X(v3_cmpgt, 6, 3, stb(o, cmpgt(ld3(i), ld3(i + 3))))                                                                     \
  X(v3_cmple, 6, 3, stb(o, cmple(ld3(i), ld3(i + 3))))                                                                     \
  X(v3_cmplt, 6, 3, stb(o, cmplt(ld3(i), ld3(i + 3))))                                                                     \
  X(v3_cmp_any_all, 6, 2, BVec3A m = cmple(ld3(i), ld3(i + 3)); o[0] = any(m) ? 1u : 0u; o[1] = all(m) ? 1u : 0u)          \
  X(v3_select, 12, 3, st3(o, vsel3(i)))                                                                                    \
  X(v3_is_finite_mask, 3, 3, stb(o, is_finite_mask(ld3(i))))                                                               \
  X(v3_as_ivec3, 3, 3, o[0] = (uint32_t)lf_f2i(i[0]); o[1] = (uint32_t)lf_f2i(i[1]); o[2] = (uint32_t)lf_f2i(i[2]))        \
  /* Vec4 */                                                                                                               \
  X(v4_add, 8, 4, st4(o, ld4(i) + ld4(i + 4)))                                                                             \
  X(v4_sub, 8, 4, st4(o, ld4(i) - ld4(i + 4)))                                                                             \
  X(v4_mul, 8, 4, st4(o, ld4(i) * ld4(i + 4)))                                                                             \
  X(v4_div, 8, 4, st4(o, ld4(i) / ld4(i + 4)))                                                                             \
  X(v4_one_div, 4, 4, st4(o, splat4(1.0f) / ld4(i)))                                                                       \
  X(v4_neg_div, 8, 4, st4(o, -ld4(i) / ld4(i + 4)))                                                                        \
  X(v4_abs, 4, 4, st4(o, abs(ld4(i))))                                                                                     \
  X(v4_min, 8, 4, st4(o, min(ld4(i), ld4(i + 4))))                                                                         \
  X(v4_max, 8, 4, st4(o, max(ld4(i), ld4(i + 4))))                                                                         \
  X(v4_dot, 8, 1, stf(o, dot(ld4(i), ld4(i + 4))))                                                                         \
  X(v4_length, 4, 1, stf(o, length(ld4(i))))                                                                               \
  X(v4_normalize, 4, 4, st4(o, normalize(ld4(i))))                                                                         \
  X(v4_max_element, 4, 1, stf(o, max_element(ld4(i))))                                                                     \
  X(v4_max_position, 4, 1, o[0] = (uint32_t)max_position(ld4(i)))                                                          \
  X(v4_cmpge, 8, 4, stb(o, cmpge(ld4(i), ld4(i + 4))))                                                                     \
  X(v4_cmpgt, 8, 4, stb(o, cmpgt(ld4(i), ld4(i + 4))))                                                                     \
  X(v4_cmple, 8, 4, stb(o, cmple(ld4(i), ld4(i + 4))))                                                                     \
  X(v4_cmplt, 8, 4, stb(o, cmplt(ld4(i), ld4(i + 4))))                                                                     \
  X(v4_mask_ops, 12, 4, stb(o, (cmpge(ld4(i), ld4(i + 4)) & cmple(ld4(i), ld4(i + 8))) | ~cmpgt(ld4(i + 4), ld4(i + 8))))  \
  X(v4_select, 16, 4, st4(o, select(cmplt(ld4(i), ld4(i + 4)), ld4(i + 8), ld4(i + 12))))                                  \
  X(v4_is_finite_mask, 4, 4, stb(o, is_finite_mask(ld4(i))))                                                               \
  /* Mat3A, Affine3A */                                                                                                    \
  X(m3_mul_v3, 12, 3, st3(o, ldm(i) * ld3(i + 9)))                                                                         \
  X(m3_mul_transpose_v3, 12, 3, st3(o, mul_transpose_vec3a(ldm(i), ld3(i + 9))))                                           \
  X(m3_mul_m3, 18, 9, stm(o, ldm(i) * ldm(i + 9)))                                                                         \
  X(m3_transpose, 9, 9, stm(o, transpose(ldm(i))))                                                                         \
  X(m3_from_quat, 4, 9, stm(o, mat3a_from_quat(ldq(i))))                                                                   \
  X(aff_mul, 24, 12, sta(o, lda(i) * lda(i + 12)))                                                                         \
  X(aff_transform_point3a, 15, 3, st3(o, transform_point3a(lda(i), ld3(i + 12))))                                          \
  X(aff_transform_vector3a, 15, 3, st3(o, transform_vector3a(lda(i), ld3(i + 12))))                                        \
  X(aff_transpose, 12, 12, sta(o, affine_transpose(lda(i))))                                                               \
  X(aff_inv_x_form, 15, 3, st3(o, affine_inv_x_form(lda(i), ld3(i + 12))))                                                 \
  /* Quat */                                                                                                               \
  X(q_mul, 8, 4, stq(o, ldq(i) * ldq(i + 4)))                                                                              \
  X(q_normalize, 4, 4, stq(o, normalize(ldq(i))))                                                                          \
  X(q_mul_vec3a, 7, 3, st3(o, mul_vec3a(ldq(i), ld3(i + 4))))                                                              \
  X(q_from_mat3a, 9, 4, stq(o, quat_from_mat3a(ldm(i))))                                                                   \
  /* the integrate_trans tail, glam half: (dorn * rot).normalize() then Mat3A::from_quat */                                \
  X(q_integrate_tail, 8, 13, Quat q = normalize(ldq(i) * ldq(i + 4)); stq(o, q); stm(o + 4, mat3a_from_quat(q)))
