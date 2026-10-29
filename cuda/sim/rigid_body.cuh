// V3 bullet/dynamics/rigid_body.rs: the per-body steps of P2 and P6 and the methods the game-logic phases call.
#pragma once
#include "sim/arena_state.cuh"
#include "sim/bullet_consts.cuh"
#include "common/libm.cuh"
#include "sim/consts.cuh"
#include "sim/transform_util.cuh"

namespace lfsim {
using namespace glam;

enum : uint32_t { ACT_ACTIVE = 0, ACT_SLEEPING = 1, ACT_DISABLED = 2, ACT_WANTS_DEACTIVATION = 3 };          // rigid_body.rs:44-50

// V3: rigid_body.rs:292 is_active
LF_HDI bool rb_is_active(const Body& b) { return !(b.activation == ACT_SLEEPING || b.activation == ACT_DISABLED); }

// V3: rigid_body.rs:303 set_activation_state
LF_HDI void rb_set_activation_state(Body& b, uint32_t new_state) {
  if (b.activation != ACT_DISABLED) b.activation = new_state;
}

// V3: rigid_body.rs:266 set_world_trans (the debug-only static-transform assert is dropped)
LF_HDI void rb_set_world_trans(Body& b, const Affine3A& world_trans) {
  if ((b.collision_flags & CF_NO_ANGULAR_MOTION) == 0) b.world_quat = quat_from_mat3a(world_trans.matrix3);
  b.world_trans = world_trans;
}

// V3: rigid_body.rs:332 get_inertia_tensor
LF_HDI Mat3A rb_get_inertia_tensor(Mat3A world_mat, Vec3A inv_inertia_local) {
  Mat3A scaled_mat = transpose(world_mat);
  scaled_mat.x_axis *= inv_inertia_local;
  scaled_mat.y_axis *= inv_inertia_local;
  scaled_mat.z_axis *= inv_inertia_local;
  return world_mat * scaled_mat;
}

// V3: rigid_body.rs:341 update_inertia_tensor
LF_HDI void rb_update_inertia_tensor(Body& b, const BodyConst& c) {
  b.inv_inertia_tensor_world = rb_get_inertia_tensor(b.world_trans.matrix3, c.inv_inertia_local);
}

// V3: rigid_body.rs:353 add_impulse(Impulse::Linear(v), massed = false, accum = true): lin_impulse = v * 1.0, ang_impulse = ZERO
LF_HDI void rb_add_linear_impulse_accum(Body& b, Vec3A v) {
  Vec3A lin_impulse = v * 1.0f;
  Vec3A ang_impulse = splat3(0.0f);
  b.accum_lin_vel += lin_impulse;
  b.accum_ang_vel += ang_impulse;
}

// V3: rigid_body.rs:383 apply_damping. The cache is recomputed only when the time step or the damping value differs from the
// cached ones; otherwise the cached powf (possibly the exporting platform's) is reused.
LF_HDI void rb_apply_damping(Body& b, const BodyConst& c, float time_step) {
  if (c.linear_damping != 0.0f) {
    if (b.cached_damping_time_step != time_step || b.cached_linear_damping_value != c.linear_damping) {
      b.cached_linear_damping = lf::powf(lf_sub(1.0f, c.linear_damping), time_step);
      b.cached_damping_time_step = time_step;
      b.cached_linear_damping_value = c.linear_damping;
    }
    b.lin_vel *= b.cached_linear_damping;
  }
}

// V3: rigid_body.rs:397 predict_integration_trans
LF_HDI Affine3A rb_predict_integration_trans(const Body& b, float time_step) {
  Affine3A trans = b.world_trans;
  Quat quat = b.world_quat;
  if ((b.collision_flags & CF_NO_ANGULAR_MOTION) != 0) {
    integrate_trans_no_rot(&trans.translation, b.lin_vel, time_step);
  } else {
    integrate_trans(&trans, &quat, b.lin_vel, b.ang_vel, time_step);
  }
  return trans;
}

// V3: rigid_body.rs:410 set_center_of_mass_trans
LF_HDI void rb_set_center_of_mass_trans(Body& b, const BodyConst& c, const Affine3A& xform) {
  b.interp_world_trans = xform;
  rb_set_world_trans(b, xform);
  rb_update_inertia_tensor(b, c);
}

// V3: rigid_body.rs:432-456 update_activation_state
LF_HDI void rb_update_activation_state(Body& b, const BodyConst& c, float time_step) {
  if ((b.collision_flags & CF_CAN_SLEEP) == 0) {
    rb_set_activation_state(b, ACT_ACTIVE);
    return;
  }
  if (b.activation == ACT_SLEEPING) return;
  float lin_thresh_sq = lf_mul(c.linear_sleeping_threshold, c.linear_sleeping_threshold);
  float ang_thresh_sq = lf_mul(c.angular_sleeping_threshold, c.angular_sleeping_threshold);
  bool within = length_squared(b.lin_vel) < lin_thresh_sq && length_squared(b.ang_vel) < ang_thresh_sq;
  if (within) {
    b.deactivation_time = lf_add(b.deactivation_time, time_step);
    if (b.deactivation_time > 2.0f && b.activation == ACT_ACTIVE) rb_set_activation_state(b, ACT_WANTS_DEACTIVATION);
  } else {
    b.deactivation_time = 0.0f;
    rb_set_activation_state(b, ACT_ACTIVE);
  }
}

// V3: rigid_body.rs:462-466 sleep
LF_HDI void rb_sleep(Body& b) {
  rb_set_activation_state(b, ACT_SLEEPING);
  b.lin_vel = splat3(0.0f);
  b.ang_vel = splat3(0.0f);
}

// V3: rigid_body.rs:344-347 force_activate
LF_HDI void rb_force_activate(Body& b) {
  b.activation = ACT_ACTIVE;
  b.deactivation_time = 0.0f;
}


// V3: rigid_body.rs:492-500 limit_vels
LF_HDI void rb_limit_vels(Body* b, float max_lin_speed, float max_ang_speed) {
  if (glam::length_squared(b->lin_vel) > lf::powi(max_lin_speed, 2)) {
    b->lin_vel = glam::normalize_or_zero(b->lin_vel) * max_lin_speed;
  }
  if (glam::length_squared(b->ang_vel) > lf::powi(max_ang_speed, 2)) {
    b->ang_vel = glam::normalize_or_zero(b->ang_vel) * max_ang_speed;
  }
}

// V3: rigid_body.rs:353-381 add_impulse(Impulse::Linear(v), massed = false, accum = false). The angular impulse is
// Vec3A::ZERO and is still added (-0.0 + 0.0 is +0.0).
LF_HDI void rb_add_linear_impulse(Body* b, Vec3A v) {
  Vec3A lin_impulse = v * 1.0f;  // massed_scaler = 1.0
  Vec3A ang_impulse = glam::splat3(0.0f);
  b->lin_vel += lin_impulse;
  b->ang_vel += ang_impulse;
}

// V3: rigid_body.rs:439-442 clear_accum_vels
LF_HDI void rb_clear_accum_vels(Body* b) {
  b->accum_lin_vel = glam::splat3(0.0f);
  b->accum_ang_vel = glam::splat3(0.0f);
}

}  // namespace lfsim
