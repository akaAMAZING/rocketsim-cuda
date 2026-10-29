// V3 bullet/dynamics/vehicle/vehicle_rl.rs: the btVehicleRL tick of one car: four wheel rays, per-wheel response with the
// victim pushback applied in wheel order before the next wheel reads the victim's velocity, then suspension for all wheels, then
// the friction impulses (computed for all wheels after all suspension, applied last).
#pragma once
#include "sim/wheel_info.cuh"

namespace lfsim {
using namespace glam;

// V3: quat_ext.rs:10 Quat::from_axis_angle_simd(axis, angle) * v, i.e. the steered axle (vehicle_rl.rs:127-132)
LF_HDI Vec3A steered_axle(const Affine3A& chassis_trans, float steer_angle) {
  float s, c;
  lf::sin_cos(lf_mul(steer_angle, 0.5f), &s, &c);  // V3: f32::sin_cos
  Vec3A v = chassis_trans.matrix3.z_axis * s;
  Quat q = quat_from_xyzw(v.x, v.y, v.z, c);
  return mul_vec3a(q, chassis_trans.matrix3.y_axis);
}

// V3: vehicle_rl.rs:55 get_upwards_dir_from_wheel_contacts; `chassis_up` is cb.get_up_vector() (rigid_body.rs:463)
LF_HDI Vec3A get_upwards_dir_from_wheel_contacts(const Car& car, Vec3A chassis_up) {
  Vec3A sum_contact_dir = splat3(0.0f);
  for (int i = 0; i < NUM_WHEELS; ++i) {
    const Wheel& wheel = car.wheels[i];
    if (wheel.raycast_info.some && wheel.raycast_info.v.is_in_contact_with_world) sum_contact_dir += wheel.raycast_info.v.contact_normal;
  }
  Vec3A out;
  if (try_normalize(sum_contact_dir, &out)) return out;
  return chassis_up;
}

// V3: vehicle_rl.rs:167 apply_single_wheel_dynamic_effects
LF_HDI void apply_single_wheel_dynamic_effects(ArenaState* s, const StaticData* sd, const Wheel& wheel) {
  if (!wheel.raycast_info.some) return;
  const RaycastInfo& info = wheel.raycast_info.v;

  if (info.ground_stick.some) {
    int d = dyn_slot_p1(sd, info.ground_body_idx);
    Body& ground = s->bodies[d];
    Vec3A ground_offset = info.contact_point - ground.world_trans.translation;
    rb_add_impulse_p1(ground, sd->bodies[info.ground_body_idx], IMPULSE_LINEAR_REL_POS, info.ground_stick.v, ground_offset, true, false);
  }

  if (wheel.extra_pushback <= 0.0f) return;

  if (sd->bodies[info.ground_body_idx].user_idx != c::UI_CAR) return;
  int d = dyn_slot_p1(sd, info.ground_body_idx);
  Body& victim = s->bodies[d];

  float full_pushback = lf_mul(wheel.extra_pushback, (float)NUM_WHEELS);
  Vec3A victim_offset = info.contact_point - victim.world_trans.translation;
  rb_add_impulse_p1(victim, sd->bodies[info.ground_body_idx], IMPULSE_LINEAR_REL_POS, -info.contact_normal * full_pushback, victim_offset, true,
                    false);
}

// V3: vehicle_rl.rs:203 update. `ci` is the car's index.
LF_HDI void vehicle_update(ArenaState* s, const StaticData* sd, int ci, float time_step, float handbrake_val, float real_throttle,
                           bool three_wheels) {
  Car& car = s->cars[ci];
  const CarConst& cc = sd->cars[ci];
  const uint32_t chassis_idx = cc.rigid_body_idx;
  const int cd = dyn_slot_p1(sd, chassis_idx);
  const BodyConst& chassis_bc = sd->bodies[chassis_idx];

  Affine3A chassis_trans = s->bodies[cd].world_trans;
  Vec3A chassis_translation = chassis_trans.translation;
  float friction_scale = lf_div(rb_get_mass_p1(chassis_bc), 3.0f);

  Vec3A sources[NUM_WHEELS], targets[NUM_WHEELS];
  for (int i = 0; i < NUM_WHEELS; ++i) wheel_prepare_for_raycast(car.wheels[i], cc.wheels[i], chassis_trans, &sources[i], &targets[i]);

  VehicleRaycasterResult hits[NUM_WHEELS];
  raycaster_cast_rays(s, sd, sources, targets, chassis_idx, hits);

  // Front wheels normally share one steer angle: compute the steered axle lazily, reuse it while the angle matches.
  bool cache_some = false;
  float cache_angle = 0.0f;
  Vec3A cache_axle = splat3(0.0f);
  int num_wheels_in_contact = 0;
  for (int i = 0; i < NUM_WHEELS; ++i) {
    Wheel& wheel = car.wheels[i];
    if (!hits[i].some) {
      wheel_reset_wheel_suspension(wheel);
      continue;
    }

    {
      RbView chassis = rb_view_p1(s, sd, chassis_idx);
      RbView hit_body = rb_view_p1(s, sd, hits[i].rigid_body_idx);
      num_wheels_in_contact += 1;
      bool front = i < 2;
      float steer_angle = wheel.steer_angle;
      Vec3A axle_dir;
      if (front) {
        if (cache_some && cache_angle == steer_angle) {
          axle_dir = cache_axle;
        } else {
          Vec3A axle = steered_axle(chassis_trans, steer_angle);
          cache_some = true;
          cache_angle = steer_angle;
          cache_axle = axle;
          axle_dir = axle;
        }
      } else {
        axle_dir = chassis_trans.matrix3.y_axis;
      }

      wheel_apply_ray_cast(wheel, cc.wheels[i], chassis, chassis_trans, axle_dir, hits[i], hit_body, time_step, front);

      bool is_dynamic_hit = !rb_is_static_p1(hit_body);
      FrictionCurveInput in;
      in.chassis_translation = chassis_translation;
      in.contact_normal = hits[i].hit_normal_in_world;
      in.handbrake_val = handbrake_val;
      in.real_throttle = real_throttle;
      in.three_wheels = three_wheels;
      in.is_dynamic_hit = is_dynamic_hit;
      wheel_refresh_friction_curves(wheel, chassis, in);
    }
    apply_single_wheel_dynamic_effects(s, sd, wheel);
  }

  if (num_wheels_in_contact < 3) {
    for (int i = 0; i < NUM_WHEELS; ++i) car.wheels[i].engine_force = lf_div(car.wheels[i].engine_force, 4.0f);
  }

  for (int i = 0; i < NUM_WHEELS; ++i) wheel_update_suspension(car.wheels[i], cc.wheels[i], s->bodies[cd], chassis_bc, chassis_translation, time_step);

  {
    RbView chassis = rb_view_p1(s, sd, chassis_idx);
    for (int i = 0; i < NUM_WHEELS; ++i)
      wheel_update_friction_impulse(car.wheels[i], chassis, chassis_bc.inv_inertia_local, time_step, friction_scale);
  }

  // note: all suspension MUST be updated before impulses are applied
  for (int i = 0; i < NUM_WHEELS; ++i) wheel_apply_friction_impulses(car.wheels[i], s->bodies[cd], chassis_bc, chassis_trans, time_step);
}

}  // namespace lfsim
