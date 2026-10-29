// V3: shared/quantize.rs (only `quantize` of a body: UE3 position / velocity / angular velocity quantization).
#pragma once
#include "common/glam.cuh"
#include "sim/consts.cuh"
#include "common/libm.cuh"

namespace lfsim {

// glam 0.34.1 sse2.rs:88 m128_floor (DirectXMath XMVectorFloor), per lane: lanes with |v| < 2^23 become
// trunc(v) + (trunc(v) > v ? -1 : 0) (so floor(-0.0) is +0.0); NaN, inf and |v| >= 2^23 pass through. Not in glam.cuh.
LF_HDI float glam_floor(float v) {
  if ((int32_t)(lf_bits(v) & 0x7FFFFFFFu) < (int32_t)0x4B000000) {
    float result = lf_i2f(lf_f2i(v));  // _mm_cvttps_epi32 + _mm_cvtepi32_ps (|v| < 2^23: no overflow)
    float larger = (result > v) ? -1.0f : 0.0f;
    return lf_add(result, larger);
  }
  return v;
}
LF_HDI Vec3A glam_floor(Vec3A v) { return {glam_floor(v.x), glam_floor(v.y), glam_floor(v.z)}; }

// V3: quantize.rs:8-11 vec3a_sign_3
LF_HDI Vec3A vec3a_sign_3(Vec3A v) {
  Vec3A sg = glam::signum(v);
  return glam::select(glam::cmpeq(v, glam::splat3(0.0f)), glam::splat3(0.0f), sg);
}

// V3: quantize.rs:19-39 quantize_vec_ue3, Position mode
LF_HDI Vec3A quantize_vec_position(Vec3A vec, float scale) {
  return glam_floor(vec * scale + glam::splat3(0.5f)) / scale;
}
// V3: quantize.rs:19-39 quantize_vec_ue3, Velocity mode
LF_HDI Vec3A quantize_vec_velocity(Vec3A vec, float scale) {
  float inv_scale = lf_div(1.0f, scale);
  Vec3A scaled = vec * scale;
  Vec3A rounded = glam::splat3(0.0f);
  rounded.x = lf_mul(lf_i2f(lf_f2i(scaled.x)), inv_scale);
  rounded.y = lf_mul(lf_i2f(lf_f2i(scaled.y)), inv_scale);
  rounded.z = lf_mul(lf_i2f(lf_f2i(scaled.z)), inv_scale);
  float offset_mag = lf_mul(c::QUANT_OFFSET_CORRECT_FRAC, inv_scale);
  return rounded + (vec3a_sign_3(rounded) * offset_mag);
}

// V3: quantize.rs:42-60 quantize(body): position (world_trans.translation), lin_vel, ang_vel of one body.
LF_HDI void quantize_body(Vec3A* pos, Vec3A* lin_vel, Vec3A* ang_vel) {
  Vec3A new_pos = quantize_vec_position(*pos * c::BT_TO_UU, c::QUANT_POS_SCALE) * c::UU_TO_BT;
  Vec3A new_vel = quantize_vec_velocity(*lin_vel * c::BT_TO_UU, c::QUANT_VEL_SCALE) * c::UU_TO_BT;
  Vec3A new_ang_vel = quantize_vec_velocity(*ang_vel, c::QUANT_ANG_VEL_SCALE);
  *pos = new_pos;
  *lin_vel = new_vel;
  *ang_vel = new_ang_vel;
}

// ---- shared/quantize.rs: the 8-bit input quantisation of the controls ----
// glam 0.34.1 sse2.rs:165 m128_round_ties_even (DirectXMath XMVectorRound), one lane. PS_NO_FRACTION = 8388608.0.
LF_HDI float sse_round_ties_even(float v) {
  uint32_t sign = lf_bits(v) & 0x80000000u;
  float s_magic = lf_from_bits(lf_bits(8388608.0f) | sign);
  float r1 = lf_sub(lf_add(v, s_magic), s_magic);
  float r2 = lf_from_bits(lf_bits(v) & 0x7FFFFFFFu);
  bool mask = r2 <= 8388608.0f;
  uint32_t r2b = mask ? 0u : lf_bits(v);
  uint32_t r1b = mask ? lf_bits(r1) : 0u;
  return lf_from_bits(r1b ^ r2b);
}

// glam 0.34.1 sse2.rs:181 m128_round (f32::round: ties away from zero), one lane.
LF_HDI float sse_round(float v) {
  float rounded = sse_round_ties_even(v);
  uint32_t sign = lf_bits(v) & 0x80000000u;
  bool ties_down = lf_sub(v, rounded) == lf_from_bits(lf_bits(0.5f) | sign);
  float add = ties_down ? lf_from_bits(lf_bits(1.0f) | sign) : 0.0f;
  return lf_add(rounded, add);
}

// V3: shared/quantize.rs:67 quantize_axis_inputs
LF_HDI Vec3A quantize_axis_inputs(Vec3A ctrls) {
  const float UPPER = 128.0f, LOWER = 127.0f;
  Vec3A clamped = glam::clamp(ctrls, glam::splat3(-1.0f), glam::splat3(1.0f));
  Vec3A scale = glam::select(glam::cmplt(clamped, glam::splat3(0.0f)), glam::splat3(UPPER), glam::splat3(LOWER));
  Vec3A biased = clamped * scale + glam::splat3(UPPER);
  Vec3A w = biased + biased + glam::splat3(0.5f);
  Vec3A r = {sse_round(w.x), sse_round(w.y), sse_round(w.z)};
  int32_t ix = (lf_f2i(r.x) >> 1) & 0xFF, iy = (lf_f2i(r.y) >> 1) & 0xFF, iz = (lf_f2i(r.z) >> 1) & 0xFF;
  Vec3A byte = {lf_i2f(ix), lf_i2f(iy), lf_i2f(iz)};
  Vec3A s = byte - glam::splat3(UPPER);
  return glam::select(glam::cmplt(s, glam::splat3(0.0f)), s * glam::splat3(lf_div(1.0f, UPPER)), s / glam::splat3(LOWER));
}

// V3: shared/quantize.rs:87 quantize_axis_input
LF_HDI float quantize_axis_input(float x) {
  float clamped = rs_clamp(x, -1.0f, 1.0f);
  float y = clamped < 0.0f ? rs_max(lf_mul(clamped, 128.0f), -128.0f) : rs_min(lf_mul(clamped, 127.0f), 127.0f);
  float w = lf_add(lf_add(lf_add(y, 128.0f), lf_add(y, 128.0f)), 0.5f);
  int32_t eax = lf_f2i(sse_round_ties_even(w));
  uint32_t byte = (uint32_t)((eax >> 1) & 0xFF);
  float s = lf_sub(lf_u2f(byte), 128.0f);
  if (byte < 0x80u) return lf_mul(s, lf_div(1.0f, 128.0f));
  return lf_div(s, 127.0f);
}

}  // namespace lfsim
