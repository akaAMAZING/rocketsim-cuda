// V3: sim/arena/base.rs, the parts of step_tick that are plain game logic: P0 (clear accumulators, limit_vels, quantize, state
// refresh), P7 (contact post-processing in record order) and P8 (car post_tick_update, finish_physics_tick, pad pickup, ball
// finish, tick_count). The events go into ArenaState.events as EventBit bits.
#pragma once
#include "common/libm.cuh"
#include "sim/arena_state.cuh"
#include "sim/ball_base.cuh"
#include "sim/boost_pad_grid.cuh"
#include "sim/car_base.cuh"
#include "sim/consts.cuh"
#include "sim/linear_piece_curve.cuh"
#include "sim/quantize.cuh"
#include "sim/rigid_body.cuh"

namespace lfsim {

// ---- P0 ---------------------------------------------------------------------------------------------------------------

// V3: arena/base.rs:612-641 (step_tick) / 1749-1775 (step_phase 0); bullet_world.clear_accum_forces is
// discrete_dynamics_world.rs:192 (clear_accum_vels on each dynamic body).
LF_HDI void phase_p0(ArenaState* s, const StaticData* sd) {
  s->events = 0;  // events.clear() (base.rs:612)
  for (int d = 0; d < NUM_DYN; ++d) rb_clear_accum_vels(&s->bodies[d]);

  // !ball_only: cars in index order (their bodies are bodies[1 + car_idx])
  for (int car_idx = 0; car_idx < NUM_CARS; ++car_idx) {
    Body* car_rb = &s->bodies[1 + car_idx];
    rb_limit_vels(car_rb, lf_mul(c::CAR_MAX_SPEED, c::UU_TO_BT), c::CAR_MAX_ANG_SPEED);
    quantize_body(&car_rb->world_trans.translation, &car_rb->lin_vel, &car_rb->ang_vel);
    s->cars[car_idx].vel = car_rb->lin_vel * c::BT_TO_UU;
    s->cars[car_idx].ang_vel = car_rb->ang_vel;
  }
  Body* ball_rb = &s->bodies[0];
  rb_limit_vels(ball_rb, lf_mul(sd->mut_f[c::MF_BALL_MAX_SPEED], c::UU_TO_BT), c::BALL_MAX_ANG_SPEED);
  quantize_body(&ball_rb->world_trans.translation, &ball_rb->lin_vel, &ball_rb->ang_vel);
  s->ball.vel = ball_rb->lin_vel * c::BT_TO_UU;
  s->ball.ang_vel = ball_rb->ang_vel;
}

// ---- P7 ---------------------------------------------------------------------------------------------------------------

// V3: arena/base.rs:1286-1297 is_within_rect_cone
LF_HDI bool is_within_rect_cone(Vec3A forward, Vec3A right, Vec3A up, Vec3A delta, bool b_reverse, float yaw_limit_deg,
                                float pitch_limit_deg) {
  Vec3A fwd = b_reverse ? -forward : forward;
  Vec3A d;
  if (!glam::try_normalize(delta, &d)) return false;

  float f = glam::dot(d, fwd);
  if (f <= 0.0f) return false;

  // to_radians: x * (PI / 180) (core f32::to_radians); angle = atan2(|component|, f) <= limit <=> |component| <= f * tan(limit)
  const float rads_per_deg = lf_div(c::PI_F, 180.0f);
  return lf_abs(glam::dot(d, right)) <= lf_mul(f, lf::tan(lf_mul(yaw_limit_deg, rads_per_deg))) &&
         lf_abs(glam::dot(d, up)) <= lf_mul(f, lf::tan(lf_mul(pitch_limit_deg, rads_per_deg)));
}
// V3: arena/base.rs:1299-1317 is_within_demo_cone
LF_HDI bool is_within_demo_cone(Vec3A forward, Vec3A right, Vec3A up, Vec3A delta, bool b_reverse) {
  return is_within_rect_cone(forward, right, up, delta, b_reverse, c::DEMO_YAW_LIMIT_DEG, c::DEMO_PITCH_LIMIT_DEG);
}
// V3: arena/base.rs:1319-1337 is_within_bump_cone
LF_HDI bool is_within_bump_cone(Vec3A forward, Vec3A right, Vec3A up, Vec3A delta, bool b_reverse) {
  return is_within_rect_cone(forward, right, up, delta, b_reverse, c::BUMP_YAW_LIMIT_DEG, c::BUMP_PITCH_LIMIT_DEG);
}

// V3: arena/base.rs:1542-1570 on_car_ball_collision (the event's contact point and extra hit velocity are not stored)
LF_HDI void on_car_ball_collision(ArenaState* s, const StaticData* sd, int car_idx) {
  ball_on_hit(s, car_idx, sd);
  s->events |= 1u << (EV_CAR_HIT_BALL + car_idx);
}

// V3: arena/base.rs:1572-1580 on_car_world_collision and :1516-1540 on_ball_world_collision(_parts) are applied from the folded
// bits and normals (phase_p7, arena_contact_tracker.cuh), not per record.

// V3: arena/base.rs:1582-1707 on_car_car_collision
LF_HDI void on_car_car_collision(ArenaState* s, const StaticData* sd, int car_1_idx, int car_2_idx, const RecordPoint& mp) {
  int attacker_idx = car_1_idx;
  int victim_idx = car_2_idx;

  // NOTE: Checking the victim first, because in many use-cases, repeat-demo-victims are more likely
  if (s->cars[victim_idx].is_demoed || s->cars[attacker_idx].is_demoed) return;

  // Test collision both ways
  for (int is_swapped_i = 0; is_swapped_i < 2; ++is_swapped_i) {
    bool is_swapped = is_swapped_i != 0;
    if (is_swapped) {
      int t = attacker_idx;
      attacker_idx = victim_idx;
      victim_idx = t;
    }
    Car& attacker = s->cars[attacker_idx];
    Car& victim = s->cars[victim_idx];

    if (attacker.bump_cooldown_timer > 0.0f &&
        (!attacker.bump_cooldown_victim.some || attacker.bump_cooldown_victim.v == (uint32_t)victim_idx)) {
      // In cooldown for this victim pair (None = all victims)
      continue;
    }

    Vec3A delta_pos = victim.pos - attacker.pos;
    if (glam::dot(attacker.vel, delta_pos) < 0.0f) {
      // Moving away from the other car
      continue;
    }

    Vec3A vel_dir = glam::normalize_or_zero(attacker.vel);
    Vec3A dir_to_victim = glam::normalize_or_zero(delta_pos);

    float speed_towards_other_car = glam::dot(attacker.vel, dir_to_victim);
    float other_car_away_speed = glam::dot(victim.vel, vel_dir);
    if (speed_towards_other_car <= other_car_away_speed) {
      // Going towards other car slower than they're going away
      continue;
    }

    Vec3A a_fwd = attacker.rot_mat.x_axis, a_right = attacker.rot_mat.y_axis, a_up = attacker.rot_mat.z_axis;
    bool center_in_cone = is_within_bump_cone(a_fwd, a_right, a_up, delta_pos, false);

    // Contact-relative residual
    Vec3A contact_on_attacker = is_swapped ? mp.pos_world_on_b : mp.pos_world_on_a;
    Vec3A contact_on_victim = is_swapped ? mp.pos_world_on_a : mp.pos_world_on_b;
    Vec3A gate_vec = (victim.pos - contact_on_victim * c::BT_TO_UU) - (attacker.pos - contact_on_attacker * c::BT_TO_UU);
    bool residual_in_cone = is_within_bump_cone(a_fwd, a_right, a_up, gate_vec, false);

    if (!center_in_cone && !residual_in_cone) continue;

    uint32_t demo_mode = sd->demo_mode;
    bool is_demo;
    if (demo_mode == c::DEMO_ON_CONTACT) {
      is_demo = true;
    } else if (demo_mode == c::DEMO_DISABLED) {
      is_demo = false;
    } else {
      is_demo = false;
      if (attacker.is_supersonic) {
        float fwd_speed = glam::dot(attacker.vel, a_fwd);
        is_demo = lf_abs(fwd_speed) >= c::SUPERSONIC_MAINTAIN_MIN_SPEED &&
                  is_within_demo_cone(a_fwd, a_right, a_up, delta_pos, fwd_speed < 0.0f);
      }
    }
    if (is_demo && !sd->enable_team_demos) {
      is_demo = sd->cars[attacker_idx].team != sd->cars[victim_idx].team;
    }

    if (is_demo) {
      car_demolish(&victim, sd->mut_f[c::MF_RESPAWN_DELAY]);
    } else {
      bool ground_hit = victim.is_on_ground != 0;
      float attacker_speed = rs_min(glam::length(attacker.vel), c::BUMP_ATTACKER_SPEED_MAX);
      if (ground_hit) {
        // Grounded victim: Ground and Z curves on total attacker speed.
        float base_scale = curve_bump_vel_amount_ground(attacker_speed);
        Vec3A hit_up_dir = victim.rot_mat.z_axis;
        float upward_force = lf_mul(curve_bump_upward_vel_amount(attacker_speed), sd->mut_f[c::MF_BUMP_FORCE_SCALE]);
        Vec3A bump_impulse = (vel_dir * base_scale) + (hit_up_dir * upward_force);
        victim.vel_impulse_cache += bump_impulse * c::UU_TO_BT;
      } else {
        // Airborne victim: Air curve on total attacker speed; no up term.
        float base_scale = curve_bump_vel_amount_air(attacker_speed);
        Vec3A bump_impulse = vel_dir * base_scale;
        victim.vel_impulse_cache += bump_impulse * c::UU_TO_BT;
      }
    }

    attacker.bump_cooldown_timer = sd->mut_f[c::MF_BUMP_COOLDOWN_TIME];
    attacker.bump_cooldown_victim.some = 1;
    attacker.bump_cooldown_victim.v = (uint32_t)victim_idx;

    s->events |= 1u << ((is_demo ? EV_DEMO : EV_BUMP) + victim_idx);  // CarHitCar (base.rs:1724)
  }
}

// V3: arena/base.rs:678-733 (step_tick) / 1821-1884 (step_phase 8), not ball_only. The ball-world and car-world records were
// folded at push time (arena_contact_tracker.cuh) and are applied first: no callback below reads the events
// word or world_contact_normal, so the order against the stored car-ball and car-car records does not matter.
LF_HDI void phase_p7(ArenaState* s, const StaticData* sd) {
  s->events |= s->fold_events;
  for (int car_idx = 0; car_idx < NUM_CARS; ++car_idx) {
    if (s->fold_events & (1u << (EV_CAR_HIT_WORLD + car_idx))) {
      s->cars[car_idx].world_contact_normal.some = 1;
      s->cars[car_idx].world_contact_normal.v = s->fold_car_normal[car_idx];
    }
  }
  s->fold_events = 0;
  s->fold_car_normal[0] = Vec3A{};
  s->fold_car_normal[1] = Vec3A{};

  uint32_t contact_count = s->n_records;
  for (uint32_t idx = 0; idx < contact_count; ++idx) {
    ContactRecord contact = s->records[idx];
    const BodyConst& rb_a = sd->bodies[contact.rb_idx_a];
    const BodyConst& rb_b = sd->bodies[contact.rb_idx_b];
    int user_pointer_a = (int)rb_a.user_pointer, user_pointer_b = (int)rb_b.user_pointer;
    // only car-ball (user b = Ball) and car-car records are stored
    if (rb_b.user_idx == c::UI_BALL) {
      on_car_ball_collision(s, sd, user_pointer_a);
    } else {
      on_car_car_collision(s, sd, user_pointer_a, user_pointer_b, contact.point);
    }
  }

  // contact_tracker.clear_records
  for (uint32_t idx = 0; idx < s->n_records; ++idx) s->records[idx] = ContactRecord{};
  s->n_records = 0;
}

// ---- P8 ---------------------------------------------------------------------------------------------------------------

// V3: arena/base.rs:825-871 (step_tick) / step_phase 9, not ball_only, Soccar (boost pads present).
LF_HDI void phase_p8(ArenaState* s, const StaticData* sd) {
  for (int car_idx = 0; car_idx < NUM_CARS; ++car_idx) {
    car_post_tick_update(&s->cars[car_idx], s->bodies[1 + car_idx]);
    car_finish_physics_tick(&s->cars[car_idx], &s->bodies[1 + car_idx]);
  }
  if (sd->boost_pads_enabled) boost_pads_tick(s, sd);

  ball_finish_physics_tick(s);

  s->tick_count = make_u64(get_u64(s->tick_count) + 1);
}

}  // namespace lfsim
