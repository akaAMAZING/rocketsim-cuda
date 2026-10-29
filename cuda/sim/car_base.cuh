// V3: sim/car/base.rs (demolish, post_tick_update, finish_physics_tick).
#pragma once
#include "sim/arena_state.cuh"
#include "sim/consts.cuh"
#include "sim/rigid_body.cuh"

namespace lfsim {

// V3: car/base.rs:226-229 demolish
LF_HDI void car_demolish(Car* car, float respawn_delay) {
  car->is_demoed = 1;
  car->demo_respawn_timer = respawn_delay;
}

// V3: car/base.rs:334 sync_state_to_body: rb.set_world_trans, interp, velocities, clear_accum_vels, update_inertia_tensor, and the
// velocity cache cleared (what set_car_state_full, respawn and the environment layers' state writes do).
LF_HDI void car_sync_state_to_body(Car* car, Body& rb, const BodyConst& bc) {
  rb_set_world_trans(rb, Affine3A{car->rot_mat, car->pos * c::UU_TO_BT});
  rb.interp_world_trans = rb.world_trans;
  rb.lin_vel = car->vel * c::UU_TO_BT;
  rb.ang_vel = car->ang_vel;
  rb_clear_accum_vels(&rb);
  rb_update_inertia_tensor(rb, bc);
  car->vel_impulse_cache = splat3(0.0f);
}

// V3: car/base.rs:1063-1130 post_tick_update; `rb` is the car's body.
LF_HDI void car_post_tick_update(Car* car, const Body& rb) {
  const float START_SPEED_SQ = lf_mul(c::SUPERSONIC_START_SPEED, c::SUPERSONIC_START_SPEED);
  const float MAINTAIN_MIN_SPEED_SQ = lf_mul(c::SUPERSONIC_MAINTAIN_MIN_SPEED, c::SUPERSONIC_MAINTAIN_MIN_SPEED);

  if (car->is_demoed) return;

  car->rot_mat = rb.world_trans.matrix3;

  float speed_squared = glam::length_squared(rb.lin_vel * c::BT_TO_UU);
  if (car->is_supersonic) {
    if (speed_squared >= START_SPEED_SQ) {
      // Back above start speed: reset the maintain timer.
      car->supersonic_grace_timer = 0.0f;
    } else if (speed_squared < MAINTAIN_MIN_SPEED_SQ) {
      // Dropped below the minimum maintain speed: lose supersonic immediately.
      car->is_supersonic = 0;
      car->supersonic_grace_timer = 0.0f;
    } else {
      // Between maintain min and start speed: keep supersonic for the grace period.
      car->supersonic_grace_timer = lf_add(car->supersonic_grace_timer, c::TICK_TIME);
      if (car->supersonic_grace_timer >= c::SUPERSONIC_MAINTAIN_MAX_TIME) {
        car->is_supersonic = 0;
        car->supersonic_grace_timer = 0.0f;
      }
    }
  } else if (speed_squared >= START_SPEED_SQ) {
    car->is_supersonic = 1;
    car->supersonic_grace_timer = 0.0f;
  } else {
    car->supersonic_grace_timer = 0.0f;
  }

  // Re-arm the jump when the car lands after an airborne tick.
  if (car->has_jumped && !car->is_jumping && car->is_on_ground && car->air_time > 0.0f) {
    int num_contacts = 0;
    bool all_contacts_extending = true;  // wheels.filter_map(raycast_info).all(..) stops at the first failure
    for (int w = 0; w < NUM_WHEELS; ++w) {
      if (!car->wheels[w].raycast_info.some) continue;
      num_contacts += 1;
      if (!(car->wheels[w].raycast_info.v.suspension_relative_vel > 1.0f)) {
        all_contacts_extending = false;
        break;
      }
    }
    bool leaving_ground = num_contacts != 0 && all_contacts_extending;
    if (!leaving_ground) car->has_jumped = 0;
  } else if (car->has_jumped && !car->is_jumping && car->is_on_ground && car->jump_ticks > c::JUMP_SETTLED_REARM_TICKS) {
    car->has_jumped = 0;
  }

  car->bump_cooldown_timer = rs_max(lf_sub(car->bump_cooldown_timer, c::TICK_TIME), 0.0f);
  car->prev_controls = car->controls;
}

// V3: car/base.rs:1132-1147 finish_physics_tick; `rb` is the car's body.
LF_HDI void car_finish_physics_tick(Car* car, Body* rb) {
  if (car->is_demoed) return;

  Vec3A v = car->vel_impulse_cache;
  if (!(v.x == 0.0f && v.y == 0.0f && v.z == 0.0f)) {  // Vec3A != ZERO
    rb->lin_vel += car->vel_impulse_cache;
    car->vel_impulse_cache = glam::splat3(0.0f);
  }

  car->pos = rb->world_trans.translation * c::BT_TO_UU;
  car->vel = rb->lin_vel * c::BT_TO_UU;
  car->ang_vel = rb->ang_vel;
}

}  // namespace lfsim
