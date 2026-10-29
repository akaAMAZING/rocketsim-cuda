// V3: sim/linear_piece_curve.rs. The curve is given by its control points; the pieces V3 builds in `new` (x_diff, y_diff in f32)
// are recomputed per lookup with the same f32 operations.
#pragma once
#include "common/glam.cuh"

namespace lfsim {

// V3: linear_piece_curve.rs:65-93 get_output. `pts` are N ascending (x, y) pairs.
LF_HDI float curve_get_output(const float (*pts)[2], int n, float input) {
  if (n == 1) return pts[0][1];
  if (input <= pts[0][0]) return pts[0][1];
  int i = 1;
  while (i < n) {
    if (pts[i][0] > input) break;
    i += 1;
  }
  if (i == n) return pts[n - 1][1];
  float base_x = pts[i - 1][0], base_y = pts[i - 1][1];
  float x_diff = lf_sub(pts[i][0], pts[i - 1][0]);   // linear_piece_curve.rs:41-42
  float y_diff = lf_sub(pts[i][1], pts[i - 1][1]);
  float interp_frac = lf_div(lf_sub(input, base_x), x_diff);
  return lf_add(lf_mul(y_diff, interp_frac), base_y);
}

// V3: consts.rs:435-441 BALL_CAR_EXTRA_IMPULSE_FACTOR
LF_HDI float curve_ball_car_extra_impulse_factor(float x) {
  const float p[5][2] = {{0.f, 0.65f}, {500.f, 0.65f}, {1400.f, 0.60f}, {2300.f, 0.55f}, {4600.f, 0.30f}};
  return curve_get_output(p, 5, x);
}
// V3: consts.rs:443-446 BUMP_VEL_AMOUNT_GROUND
LF_HDI float curve_bump_vel_amount_ground(float x) {
  const float p[3][2] = {{0.f, lf_div(150.0f, 180.0f)}, {1400.f, lf_div(200000.0f, 180.0f)}, {2200.f, lf_div(275000.0f, 180.0f)}};
  return curve_get_output(p, 3, x);
}
// V3: consts.rs:448-451 BUMP_VEL_AMOUNT_AIR
LF_HDI float curve_bump_vel_amount_air(float x) {
  const float p[3][2] = {{0.f, lf_div(150.0f, 180.0f)}, {1400.f, lf_div(250000.0f, 180.0f)}, {2200.f, lf_div(350000.0f, 180.0f)}};
  return curve_get_output(p, 3, x);
}
// V3: consts.rs:453-457 BUMP_UPWARD_VEL_AMOUNT
LF_HDI float curve_bump_upward_vel_amount(float x) {
  const float p[3][2] = {{0.f, lf_div(50.0f, 180.0f)}, {1400.f, lf_div(50000.0f, 180.0f)}, {2200.f, lf_div(75000.0f, 180.0f)}};
  return curve_get_output(p, 3, x);
}

// ---- consts.rs:409-434 curves used by the wheels and the steering ----
LF_HDI float curve_steer_angle_from_speed(float x) {
  const float p[6][2] = {{0.f, 0.53356f}, {500.f, 0.31930f}, {1000.f, 0.18203f}, {1500.f, 0.10570f}, {1750.f, 0.08507f}, {3000.f, 0.03454f}};
  return curve_get_output(p, 6, x);
}
LF_HDI float curve_steer_angle_from_speed_threewheel(float x) {
  const float p[2][2] = {{0.f, 0.342473f}, {2300.f, 0.034837f}};
  return curve_get_output(p, 2, x);
}
LF_HDI float curve_powerslide_steer_angle_from_speed(float x) {
  const float p[2][2] = {{0.f, 0.39235f}, {2500.f, 0.12610f}};
  return curve_get_output(p, 2, x);
}
LF_HDI float curve_drive_speed_torque_factor(float x) {
  const float p[3][2] = {{0.f, 1.0f}, {1400.f, 0.1f}, {1410.f, 0.0f}};
  return curve_get_output(p, 3, x);
}
LF_HDI float curve_non_sticky_friction_factor(float x) {
  const float p[3][2] = {{0.f, 0.1f}, {0.7075f, 0.5f}, {1.0f, 1.0f}};
  return curve_get_output(p, 3, x);
}
LF_HDI float curve_lat_friction(float x) {
  const float p[2][2] = {{0.f, 1.0f}, {1.0f, 0.2f}};
  return curve_get_output(p, 2, x);
}
LF_HDI float curve_lat_friction_threewheel(float x) {
  const float p[2][2] = {{0.f, 0.30f}, {1.0f, 0.25f}};
  return curve_get_output(p, 2, x);
}
LF_HDI float curve_handbrake_lat_friction_factor(float x) {
  const float p[1][2] = {{0.f, 0.1f}};
  return curve_get_output(p, 1, x);
}
LF_HDI float curve_handbrake_long_friction_factor(float x) {
  const float p[2][2] = {{0.f, 0.5f}, {1.0f, 0.9f}};
  return curve_get_output(p, 2, x);
}

}  // namespace lfsim
