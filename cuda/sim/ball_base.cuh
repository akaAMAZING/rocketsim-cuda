// V3: sim/ball/base.rs (Soccar only: no Heatseeker, Dropshot, Hoops or Snowday branches).
#pragma once
#include "sim/arena_state.cuh"
#include "sim/consts.cuh"
#include "sim/linear_piece_curve.cuh"
#include "sim/rigid_body.cuh"

namespace lfsim {

// V3: ball/base.rs:176-185 finish_physics_tick
LF_HDI void ball_finish_physics_tick(ArenaState* s) {
  const Body& rb = s->bodies[0];
  s->ball.vel = rb.lin_vel * c::BT_TO_UU;
  s->ball.ang_vel = rb.ang_vel;
  Affine3A trans = rb.world_trans;
  s->ball.pos = trans.translation * c::BT_TO_UU;
  s->ball.rot_mat = trans.matrix3;
  s->ball.tick_count_since_kickoff = make_u64(get_u64(s->ball.tick_count_since_kickoff) + 1);
}

// V3: ball/base.rs:187-249 on_hit, the Psyonix extra impulse (Soccar branch). The ball body is bodies[0].
LF_HDI void ball_on_hit(ArenaState* s, int car_idx, const StaticData* sd) {
  Car& car = s->cars[car_idx];
  Body* rb = &s->bodies[0];
  Vec3A car_forward = car.rot_mat.x_axis;
  Vec3A rel_pos = s->ball.pos - car.pos;
  Vec3A rel_vel = s->ball.vel - car.vel;

  float rel_speed = rs_min(glam::length(rel_vel), c::HIT_MAX_DELTA_VEL_UU);

  // Prevent repeated extra impulses: last_extra_hit_tick.is_none_or(|last| last + 1 < tick_count)
  bool can_accel = !car.last_extra_hit_tick.some || (get_u64(car.last_extra_hit_tick.v) + 1 < get_u64(s->tick_count));

  if (rel_speed > 0.0f && can_accel) {
    float z_scale = c::HIT_Z_SCALE_NORMAL;  // extra_z_scale is Hoops only
    Vec3A hit_dir = glam::normalize_or_zero(rel_pos * Vec3A{1.0f, 1.0f, z_scale});
    Vec3A forward_dir_adjustment = (car_forward * glam::dot(hit_dir, car_forward)) * lf_sub(1.0f, c::HIT_FORWARD_SCALE);
    hit_dir = glam::normalize_or_zero(hit_dir - forward_dir_adjustment);

    Vec3A added_hit_impulse = ((hit_dir * rel_speed) * curve_ball_car_extra_impulse_factor(rel_speed)) *
                              sd->mut_f[c::MF_BALL_HIT_EXTRA_FORCE_SCALE];
    // Apply the extra impulse after contact solving.
    rb_add_linear_impulse(rb, added_hit_impulse * c::UU_TO_BT);

    car.last_extra_hit_tick.some = 1;
    car.last_extra_hit_tick.v = s->tick_count;
  }
}

}  // namespace lfsim
