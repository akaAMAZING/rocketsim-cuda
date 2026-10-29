// V3 bullet/linear_math/transform_util.rs: the quaternion integrator.
#pragma once
#include "common/glam.cuh"
#include "common/libm.cuh"

namespace lfsim {
using namespace glam;

// V3: transform_util.rs:5 ANGULAR_MOTION_THRESHOLD = FRAC_PI_4 (f32 0x3F490FDB)
LF_HDI float angular_motion_threshold() { return lf_from_bits(0x3F490FDBu); }

// V3: transform_util.rs:8 integrate_trans_no_rot
LF_HDI void integrate_trans_no_rot(Vec3A* cur_trans, Vec3A lin_vel, float time_step) {
  *cur_trans = *cur_trans + lin_vel * time_step;
}

// V3: transform_util.rs:12 integrate_trans (Affine3A, Quat, lin_vel, ang_vel, time_step)
LF_HDI void integrate_trans(Affine3A* cur_trans, Quat* cur_rot, Vec3A lin_vel, Vec3A ang_vel, float time_step) {
  integrate_trans_no_rot(&cur_trans->translation, lin_vel, time_step);

  float angle = length(ang_vel);
  if (lf_mul(angle, time_step) > angular_motion_threshold()) {
    angle = lf_div(angular_motion_threshold(), time_step);
  }

  float half_angle = lf_mul(lf_mul(angle, time_step), 0.5f);
  float k;
  if (angle < 0.001f) {
    // 0.5 * ts - ts*ts*ts*0.020_833_334*angle*angle, left to right
    float t3 = lf_mul(lf_mul(time_step, time_step), time_step);
    float sub = lf_mul(lf_mul(lf_mul(t3, 0.020833334f), angle), angle);
    k = lf_sub(lf_mul(0.5f, time_step), sub);
  } else {
    k = lf_div(lf::sin(half_angle), angle);
  }
  Vec3A axis = ang_vel * k;

  Quat dorn = quat_from_xyzw(axis.x, axis.y, axis.z, lf::cos(half_angle));
  *cur_rot = normalize(dorn * *cur_rot);
  cur_trans->matrix3 = mat3a_from_quat(*cur_rot);
}

}  // namespace lfsim
