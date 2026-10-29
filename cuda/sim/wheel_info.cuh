// V3 bullet/dynamics/vehicle/wheel_info.rs: one wheel's raycast response, friction curves, suspension and friction
// impulses. `Wheel` is the dynamic part (ArenaState), `WheelConst` the suspension constants (StaticData).
#pragma once
#include "sim/raycaster.cuh"
#include "sim/contact_constraint.cuh"

namespace lfsim {
using namespace glam;

constexpr int NUM_WHEELS_F = NUM_WHEELS;  // vehicle/mod.rs NUM_WHEELS as f32 below

// V3: wheel_info.rs:93 prepare_for_raycast
LF_HDI void wheel_prepare_for_raycast(Wheel& w, const WheelConst& wc, const Affine3A& chassis_trans, Vec3A* source, Vec3A* target) {
  w.hard_point = transform_point3a(chassis_trans, wc.chassis_connection_point_cs);
  *target = w.hard_point - (chassis_trans.matrix3.z_axis * wc.real_ray_length);
  *source = w.hard_point;
}

// V3: wheel_info.rs:100 reset_wheel_suspension
LF_HDI void wheel_reset_wheel_suspension(Wheel& w) {
  w.extra_pushback = 0.0f;
  w.raycast_info = Opt<RaycastInfo>{};
}

// V3: wheel_info.rs:105 apply_ray_cast. `hit_body` is the body the ray hit.
LF_HDI void wheel_apply_ray_cast(Wheel& w, const WheelConst& wc, const RbView& chassis, const Affine3A& chassis_trans,
                                 Vec3A front_axle_dir, const VehicleRaycasterResult& ray_results, const RbView& hit_body,
                                 float time_step, bool front) {
  Vec3A contact_point = ray_results.hit_point_in_world;
  Vec3A contact_normal = ray_results.hit_normal_in_world;
  bool is_in_contact_with_world = rb_is_static_p1(hit_body);

  w.axle_dir = front ? front_axle_dir : chassis_trans.matrix3.y_axis;

  Vec3A up = chassis_trans.matrix3.z_axis;
  float wheel_trace_len_sq = dot(w.hard_point - contact_point, up);

  float suspension_travel = lf_mul(c::MAX_SUSPENSION_TRAVEL, c::UU_TO_BT);
  float max_suspension_len = lf_add(wc.suspension_rest_length_1, suspension_travel);

  Vec3A rel_pos = contact_point - chassis_trans.translation;
  w.vel_at_contact_point = rb_get_vel_in_local_point_p1(chassis, rel_pos);

  float proj_vel = dot(contact_normal, w.vel_at_contact_point);
  float denom = dot(contact_normal, up);

  float suspension_relative_vel, clipped_inv_contact_dot_suspension;
  if (denom > 0.1f) {
    float inv = lf_div(1.0f, denom);
    suspension_relative_vel = lf_mul(proj_vel, inv);
    clipped_inv_contact_dot_suspension = inv;
  } else {
    suspension_relative_vel = 0.0f;
    clipped_inv_contact_dot_suspension = 10.0f;
  }

  float suspension_length = rs_min(lf_sub(wheel_trace_len_sq, wc.wheels_radius), max_suspension_len);

  w.extra_pushback = 0.0f;
  bool is_car_hit = hit_body.user_idx == c::UI_CAR;
  if (is_in_contact_with_world || is_car_hit) {
    float ray_pushback_thresh = lf_sub(lf_add(wc.suspension_rest_length_1, wc.wheels_radius), c::SUSPENSION_SUBTRACTION);
    if (wheel_trace_len_sq < ray_pushback_thresh) {
      float wheel_trace_dist_delta = lf_sub(wheel_trace_len_sq, ray_pushback_thresh);

      float collision_result = resolve_single_collision(chassis, hit_body, ray_results.hit_point_in_world,
                                                        ray_results.hit_normal_in_world, time_step, wheel_trace_dist_delta);

      w.extra_pushback = lf_div(collision_result, (float)NUM_WHEELS);
    }
  }

  Opt<Vec3A> ground_stick = Opt<Vec3A>{};
  if (!is_in_contact_with_world) {
    ground_stick.some = 1;
    ground_stick.v = -contact_normal;
  }

  RaycastInfo ri;
  ri.contact_normal = contact_normal;
  ri.contact_point = contact_point;
  ri.ground_body_idx = ray_results.rigid_body_idx;
  ri.suspension_length = suspension_length;
  ri.impulse = splat3(0.0f);
  ri.ground_stick = ground_stick;
  ri.is_in_contact_with_world = is_in_contact_with_world ? 1u : 0u;
  ri.clipped_inv_contact_dot_suspension = clipped_inv_contact_dot_suspension;
  ri.suspension_relative_vel = suspension_relative_vel;
  w.raycast_info.some = 1;
  w.raycast_info.v = ri;
}

struct FrictionCurveInput {
  Vec3A chassis_translation, contact_normal;
  float handbrake_val, real_throttle;
  bool three_wheels, is_dynamic_hit;
};

// V3: wheel_info.rs:188 refresh_friction_curves
LF_HDI void wheel_refresh_friction_curves(Wheel& w, const RbView& chassis, const FrictionCurveInput& input) {
  Vec3A lat_dir = w.axle_dir;
  Vec3A long_dir = cross(lat_dir, input.contact_normal);
  Vec3A wheel_delta = w.hard_point - input.chassis_translation;
  Vec3A cross_vec = (cross(chassis.ang_vel, wheel_delta) + chassis.lin_vel) * c::BT_TO_UU;
  float base_friction = lf_abs(dot(cross_vec, lat_dir));
  float friction_curve_input = base_friction > 5.0f ? lf_div(base_friction, lf_add(lf_abs(dot(cross_vec, long_dir)), base_friction)) : 0.0f;

  float lat_friction = input.three_wheels ? curve_lat_friction_threewheel(friction_curve_input)
                                          : curve_lat_friction(friction_curve_input);

  float long_friction = 1.0f;
  if (input.handbrake_val != 0.0f) {
    lat_friction = lf_mul(lat_friction, lf_add(1.0f, lf_mul(lf_sub(curve_handbrake_lat_friction_factor(friction_curve_input), 1.0f),
                                                            input.handbrake_val)));
    long_friction = lf_mul(long_friction, lf_add(1.0f, lf_mul(lf_sub(curve_handbrake_long_friction_factor(friction_curve_input), 1.0f),
                                                              input.handbrake_val)));
  }

  if (input.real_throttle == 0.0f || input.is_dynamic_hit) {
    float non_sticky_scale = curve_non_sticky_friction_factor(input.contact_normal.z);
    lat_friction = lf_mul(lat_friction, non_sticky_scale);
    long_friction = lf_mul(long_friction, non_sticky_scale);
  }

  w.lat_friction = lat_friction;
  w.long_friction = long_friction;
}

// V3: wheel_info.rs:228 calc_friction_impulses
LF_HDI Vec3A wheel_calc_friction_impulses(const Wheel& w, const RbView& chassis, Vec3A chassis_inv_inertia_local, Vec3A contact_normal,
                                          Vec3A contact_point, float time_step, float friction_scale) {
  Vec3A axle_dir = normalize_or_zero(w.axle_dir);

  Vec3A forward_dir = normalize_or_zero(cross(contact_normal, axle_dir));

  // Lateral friction resolves against a fixed ground body, never the landed rigid body.
  float side_impulse = resolve_single_bilateral_fake_ground(chassis, chassis_inv_inertia_local, contact_point, axle_dir);

  float rolling_friction;
  if (w.engine_force == 0.0f) {
    if (w.brake == 0.0f) {
      rolling_friction = 0.0f;
    } else {
      const float ROLLING_FRICTION_SCALE = 113.73963f;

      Vec3A contact_vel = w.vel_at_contact_point;
      float rel_vel = dot(contact_vel, forward_dir);

      if (time_step > lf_div(1.0f, 80.0f)) {
        float threshold = lf_sub(0.8f, lf_div(1.0f, lf_mul(time_step, 150.0f)));
        if (lf_abs(rel_vel) < threshold) rel_vel = 0.0f;
      }

      rolling_friction = rs_clamp(lf_mul(lf_neg(rel_vel), ROLLING_FRICTION_SCALE), lf_neg(w.brake), w.brake);
    }
  } else {
    rolling_friction = lf_div(lf_neg(w.engine_force), friction_scale);
  }

  Vec3A total_friction_force = forward_dir * rolling_friction * w.long_friction + axle_dir * side_impulse * w.lat_friction;
  return total_friction_force * friction_scale;
}

// V3: wheel_info.rs:271 update_friction_impulse
LF_HDI void wheel_update_friction_impulse(Wheel& w, const RbView& chassis, Vec3A chassis_inv_inertia_local, float time_step, float friction_scale) {
  if (!w.raycast_info.some) return;
  Vec3A contact_normal = w.raycast_info.v.contact_normal;
  Vec3A contact_point = w.raycast_info.v.contact_point;
  Vec3A impulse = wheel_calc_friction_impulses(w, chassis, chassis_inv_inertia_local, contact_normal, contact_point, time_step, friction_scale);
  w.raycast_info.v.impulse = impulse;
}

// V3: wheel_info.rs:296 update_suspension
LF_HDI void wheel_update_suspension(Wheel& w, const WheelConst& wc, Body& cb, const BodyConst& cbc, Vec3A chassis_translation, float delta_time) {
  if (!w.raycast_info.some) return;
  const RaycastInfo& ri = w.raycast_info.v;

  float force = lf_mul(lf_mul(lf_sub(wc.suspension_rest_length_1, ri.suspension_length), c::SUSPENSION_STIFFNESS),
                       ri.clipped_inv_contact_dot_suspension);

  float damping_vel_scale = ri.suspension_relative_vel < 0.0f ? c::WHEELS_DAMPING_COMPRESSION : c::WHEELS_DAMPING_RELAXATION;

  float wheels_suspension_force = lf_sub(force, lf_mul(damping_vel_scale, ri.suspension_relative_vel));
  wheels_suspension_force = lf_mul(wheels_suspension_force, wc.suspension_force_scale);
  if (wheels_suspension_force <= 0.0f) return;
  float base_force_scale = lf_add(lf_mul(wheels_suspension_force, delta_time), w.extra_pushback);
  Vec3A contact_point_offset = ri.contact_point - chassis_translation;

  Vec3A f = ri.contact_normal * base_force_scale;
  rb_add_impulse_p1(cb, cbc, IMPULSE_LINEAR_REL_POS, f, contact_point_offset, true, false);
}

// V3: wheel_info.rs:333 apply_friction_impulses
LF_HDI void wheel_apply_friction_impulses(const Wheel& w, Body& cb, const BodyConst& cbc, const Affine3A& chassis_trans, float time_step) {
  if (!w.raycast_info.some) return;
  const RaycastInfo& ri = w.raycast_info.v;

  Vec3A wheel_contact_offset = ri.contact_point - chassis_trans.translation;
  float contact_up_dot = dot(chassis_trans.matrix3.z_axis, wheel_contact_offset);
  Vec3A wheel_rel_pos = wheel_contact_offset - chassis_trans.matrix3.z_axis * contact_up_dot;
  rb_add_impulse_p1(cb, cbc, IMPULSE_LINEAR_REL_POS, ri.impulse * time_step, wheel_rel_pos, true, false);
}

}  // namespace lfsim
