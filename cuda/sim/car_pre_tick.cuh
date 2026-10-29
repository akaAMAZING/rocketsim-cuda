// V3 sim/car/base.rs (post_tick_update and finish_physics_tick are in car_base.cuh): the car's pre_tick_update (P1) and what it calls: the flip z-damp prepass, the demo respawn,
// update_wheels, update_jump, update_auto_flip, update_double_jump_or_flip, update_air_torque, update_auto_roll, update_boost and
// update_contact_state. post_tick_update and finish_physics_tick (P8) are in car_base.cuh.
#pragma once
#include "sim/car_base.cuh"
#include "sim/vehicle_rl.cuh"
#include "sim/quantize.cuh"

namespace lfsim {
using namespace glam;


LF_HDI bool vec_is_zero(Vec3A v) { return v.x == 0.0f && v.y == 0.0f && v.z == 0.0f; }  // Vec3A == Vec3A::ZERO
LF_HDI void set_at(Vec3A* v, int i, float x) {
  if (i == 0) v->x = x; else if (i == 1) v->y = x; else v->z = x;
}
LF_HDI uint32_t num_wheels_in_contact(const Car& car) {  // car_state.rs:487
  uint32_t n = 0;
  for (int i = 0; i < NUM_WHEELS; ++i) n += car.wheels_with_contact[i].some ? 1u : 0u;
  return n;
}

// ---- the auto-flip roll sign (car/base.rs:68-118): f64 trigonometry, the tables rebuilt as V3 does ---------------------------------------------

// V3: car/base.rs:101 table_sin. V3 computes (idx as f64 * TAU / 16384.0).sin() as f32 with the platform f64 sin on every call.
// The port reads StaticData.table_sin, which the host filled once with exactly that expression (dump.rs, the platform sin of
// the build that makes the static data), so host-CR and device agree bit for bit and equal that platform's V3. Only the
// LF_LIBM_PLATFORM build (the porting verification) evaluates it directly.
LF_HDI float table_sin(const StaticData* sd, uint32_t idx) {
#if defined(LF_LIBM_PLATFORM) && !defined(__CUDA_ARCH__)
  const double TAU = 6.283185307179586476925286766559;
  return (float)::sin((double)idx * TAU / 16384.0);
#else
  return sd->table_sin[idx];
#endif
}

// V3: car/base.rs:93 axis_angle_to_int
LF_HDI int32_t axis_angle_to_int(float a0, float a1) {
  double ang = lf::atan2_64((double)a0, (double)a1);
  float u = lf_div(lf_mul((float)ang, 32768.0f), c::PI_F);
  float v = lf_add(lf_mul(u, 2.0f), 0.5f);
  return lf_f2i(sse_round_ties_even(v)) >> 1;
}

// V3: car/base.rs:106 rebuild_row1
LF_HDI Vec3A rebuild_row1(const StaticData* sd, int32_t pitch, int32_t yaw) {
  float tp = table_sin(sd, (uint32_t)((pitch >> 2) & 0x3FFF));
  float ty = table_sin(sd, (uint32_t)((yaw >> 2) & 0x3FFF));
  float tr = table_sin(sd, 0);
  float tpq = table_sin(sd, (uint32_t)(((pitch + 0x4000) >> 2) & 0x3FFF));
  float tyq = table_sin(sd, (uint32_t)(((yaw + 0x4000) >> 2) & 0x3FFF));
  float trq = table_sin(sd, 0x1000);
  float r1x = lf_sub(lf_mul(lf_mul(tp, tr), tyq), lf_mul(trq, ty));
  float r1y = lf_add(lf_mul(tyq, trq), lf_mul(lf_mul(tp, tr), ty));
  float r1z = lf_mul(lf_neg(tr), tpq);
  return Vec3A{r1x, r1y, r1z};
}

// V3: car/base.rs:68 autoflip_roll_units
LF_HDI int16_t autoflip_roll_units(const StaticData* sd, const Mat3A& rot_mat) {
  Vec3A col0 = rot_mat.x_axis, col1 = rot_mat.y_axis, col2 = rot_mat.z_axis;
  int32_t yaw = axis_angle_to_int(col0.y, col0.x);
  float len = (float)lf::hypot64((double)col0.x, (double)col0.y);
  int32_t pitch = axis_angle_to_int(col0.z, len);
  Vec3A r1 = rebuild_row1(sd, pitch, yaw);
  int32_t roll = axis_angle_to_int(dot(col2, r1), dot(col1, r1));
  return (int16_t)roll;
}

// V3: car/base.rs:84 autoflip_impulse_sign
LF_HDI float autoflip_impulse_sign(const StaticData* sd, const Mat3A& rot_mat) { return autoflip_roll_units(sd, rot_mat) > 0 ? 1.0f : -1.0f; }

// ---- fastrand::Rng (WyRand, fastrand 2.5.0 lib.rs:149, :187, :606) --------------------------------------------------------------------------

LF_HDI uint64_t mulhi_u64(uint64_t a, uint64_t b) {
  uint64_t a_lo = a & 0xFFFFFFFFull, a_hi = a >> 32, b_lo = b & 0xFFFFFFFFull, b_hi = b >> 32;
  uint64_t p0 = a_lo * b_lo, p1 = a_lo * b_hi, p2 = a_hi * b_lo, p3 = a_hi * b_hi;
  uint64_t mid = (p0 >> 32) + (p1 & 0xFFFFFFFFull) + (p2 & 0xFFFFFFFFull);
  return p3 + (p1 >> 32) + (p2 >> 32) + (mid >> 32);
}
LF_HDI uint64_t rng_gen_u64(U64* st) {
  uint64_t s = get_u64(*st) + 0x2d358dccaa6c78a5ull;
  *st = make_u64(s);
  uint64_t other = s ^ 0x8bb84b93962eacc9ull;
  return (s * other) ^ mulhi_u64(s, other);
}
LF_HDI uint64_t rng_gen_mod_u64(U64* st, uint64_t n) {
  uint64_t r = rng_gen_u64(st);
  uint64_t hi = mulhi_u64(r, n);
  uint64_t lo = r * n;
  if (lo < n) {
    uint64_t t = (0ull - n) % n;
    while (lo < t) {
      r = rng_gen_u64(st);
      hi = mulhi_u64(r, n);
      lo = r * n;
    }
  }
  return hi;
}

// ---- respawn (car/base.rs:234) ----------------------------------------------------------------------------------------------------------------

// glam 0.34.1 euler.rs:233 Mat3A::from_euler(EulerRot::ZYX, yaw, 0.0, 0.0): i = Z, j = Y, k = X, static frame, odd parity.
LF_HDI Mat3A mat3a_from_euler_zyx_yaw(float yaw) {
  float si, ci, sj, cj, sh, ch;
  lf::sin_cos(yaw, &si, &ci);
  lf::sin_cos(0.0f, &sj, &cj);
  lf::sin_cos(0.0f, &sh, &ch);
  float cc = lf_mul(ci, ch), cs = lf_mul(ci, sh), sc = lf_mul(si, ch), ss = lf_mul(si, sh);
  float m_ii = lf_mul(cj, ch);
  float m_ij = lf_sub(lf_mul(sj, sc), cs);
  float m_ik = lf_add(lf_mul(sj, cc), ss);
  float m_ji = lf_mul(cj, sh);
  float m_jj = lf_add(lf_mul(sj, ss), cc);
  float m_jk = lf_sub(lf_mul(sj, cs), sc);
  float m_ki = lf_neg(sj);
  float m_kj = lf_mul(cj, si);
  float m_kk = lf_mul(cj, ci);
  // from_cols_array_2d(&m): m[a] is column a; i = 2, j = 1, k = 0
  return Mat3A{Vec3A{m_kk, m_kj, m_ki}, Vec3A{m_jk, m_jj, m_ji}, Vec3A{m_ik, m_ij, m_ii}};
}

// V3: car/base.rs:234 respawn (with set_state :329 and sync_state_to_body :334; CarState::from(CarCore::grounded(phys, boost)))
LF_HDI void car_respawn(ArenaState* s, const StaticData* sd, int ci, float boost_amount) {
  Car& car = s->cars[ci];
  const CarConst& cc = sd->cars[ci];
  const int cd = dyn_slot_p1(sd, cc.rigid_body_idx);
  Body& rb = s->bodies[cd];
  const BodyConst& bc = sd->bodies[cc.rigid_body_idx];

  // get_respawn_locations(Soccar): consts.rs:288 (x, y, yaw = FRAC_PI_2)
  const float loc_x[4] = {-2304.0f, -2688.0f, 2304.0f, 2688.0f};
  const float loc_y = -4608.0f;
  const float FRAC_PI_2_F = 1.57079632679489661923132169163975144f;
  uint32_t spawn_pos_idx = (uint32_t)rng_gen_mod_u64(&s->rng, 4);

  float y_dir = cc.team == 0 ? -1.0f : 1.0f;  // Team::get_y_dir
  Vec3A pos = Vec3A{loc_x[spawn_pos_idx], lf_mul(loc_y, lf_neg(y_dir)), c::RESPAWN_Z};
  Mat3A rot_mat = mat3a_from_euler_zyx_yaw(lf_add(FRAC_PI_2_F, cc.team == 0 ? 0.0f : c::PI_F));

  // the new CarState (CarCore::grounded + From<CarCore>, car_state.rs:41 DEFAULT): everything default but pose and boost
  car.pos = pos;
  car.rot_mat = rot_mat;
  car.vel = splat3(0.0f);
  car.ang_vel = splat3(0.0f);
  car.controls = Controls{};
  car.has_jumped = car.has_double_jumped = car.has_flipped = 0;
  car.flip_rel_torque = splat3(0.0f);
  car.jump_ticks = 0;
  car.flip_time = 0.0f;
  car.is_flipping = car.is_jumping = car.jump_hold_broken = 0;
  car.air_time = 0.0f;
  car.air_time_since_jump = 0.0f;
  car.boost = rs_clamp(boost_amount, 0.0f, c::BOOST_MAX);
  car.handbrake_val = 0.0f;
  car.is_supersonic = 0;  // CarTuning::from_speed(ZERO)
  car.supersonic_grace_timer = 0.0f;
  car.time_since_boosted = 0.0f;
  car.boosting_time = 0.0f;
  car.is_boosting = 0;
  car.is_auto_flipping = 0;
  car.auto_flip_timer = 0.0f;
  car.auto_flip_torque_scale = 0.0f;
  car.bump_cooldown_timer = 0.0f;
  car.bump_cooldown_victim = Opt<uint32_t>{};
  car.last_extra_hit_tick = Opt<U64>{};
  car.is_demoed = 0;
  car.demo_respawn_timer = 0.0f;
  car.is_on_ground = 1;
  for (int i = 0; i < NUM_WHEELS; ++i) car.wheels_with_contact[i] = Opt<HitInfo>{};
  car.world_contact_normal = Opt<Vec3A>{};
  car.prev_controls = Controls{};

  car_sync_state_to_body(&car, rb, bc);  // sync_state_to_body

  // self.bullet_vehicle.reset_transient_contacts(); self.sticky_gate_prev = false;
  for (int i = 0; i < NUM_WHEELS; ++i) {
    Wheel& w = car.wheels[i];
    w.raycast_info = Opt<RaycastInfo>{};
    w.hard_point = splat3(0.0f);
    w.axle_dir = splat3(0.0f);
    w.engine_force = 0.0f;
    w.brake = 0.0f;
    w.steer_angle = 0.0f;
    w.vel_at_contact_point = splat3(0.0f);
    w.lat_friction = 1.0f;
    w.long_friction = 1.0f;
    w.extra_pushback = 0.0f;
  }
  car.sticky_gate_prev = 0;
}

// ---- the per-tick updates ---------------------------------------------------------------------------------------------------------------------------

// V3: car/base.rs:354 update_wheels
LF_HDI void car_update_wheels(ArenaState* s, const StaticData* sd, int ci, Body& rb, float forward_speed_uu, bool* cached_some,
                              Vec3A* cached_upwards_dir) {
  Car& car = s->cars[ci];
  const CarConst& cc = sd->cars[ci];
  const BodyConst& bc = sd->bodies[cc.rigid_body_idx];

  float handbrake_delta = lf_mul(car.controls.handbrake ? c::POWERSLIDE_RISE_RATE : -c::POWERSLIDE_FALL_RATE, c::TICK_TIME);
  car.handbrake_val = rs_clamp(lf_add(car.handbrake_val, handbrake_delta), 0.0f, 1.0f);

  float real_brake = 0.0f;
  float raw_throttle = car.controls.throttle;
  Vec3A q = quantize_axis_inputs(Vec3A{raw_throttle, car.controls.steer, 0.0f});
  float proc_throttle = q.x, proc_steer = q.y;

  bool all_wheels_contact = num_wheels_in_contact(car) == (uint32_t)NUM_WHEELS;
  float real_throttle = (car.controls.boost && car.boost > 0.0f && all_wheels_contact) ? 1.0f : proc_throttle;

  float abs_forward_speed_uu = lf_abs(forward_speed_uu);
  float engine_throttle = real_throttle;
  if (!car.controls.handbrake) {
    if (lf_abs(real_throttle) > c::THROTTLE_DEADZONE) {
      if (abs_forward_speed_uu > c::STOPPING_FORWARD_VEL && rs_signum(real_throttle) != rs_signum(forward_speed_uu)) {
        real_brake = 1.0f;
        if (abs_forward_speed_uu > c::BRAKING_NO_THROTTLE_SPEED_THRESH) engine_throttle = 0.0f;
      }
    } else {
      engine_throttle = 0.0f;
      real_brake = abs_forward_speed_uu < c::STOPPING_FORWARD_VEL ? 1.0f : c::COASTING_BRAKE_FACTOR;
    }
  }

  float drive_speed_scale = curve_drive_speed_torque_factor(abs_forward_speed_uu);

  float drive_engine_force = lf_mul(lf_mul(engine_throttle, c::THROTTLE_TORQUE_AMOUNT * c::UU_TO_BT), drive_speed_scale);
  float drive_brake_force = lf_mul(real_brake, c::BRAKE_TORQUE_AMOUNT * c::UU_TO_BT);
  for (int i = 0; i < NUM_WHEELS; ++i) {
    car.wheels[i].engine_force = drive_engine_force;
    car.wheels[i].brake = drive_brake_force;
  }

  float steer_angle = cc.three_wheels ? curve_steer_angle_from_speed_threewheel(abs_forward_speed_uu)
                                      : curve_steer_angle_from_speed(abs_forward_speed_uu);
  if (car.handbrake_val != 0.0f) {
    steer_angle = lf_add(steer_angle, lf_mul(lf_sub(curve_powerslide_steer_angle_from_speed(abs_forward_speed_uu), steer_angle),
                                             car.handbrake_val));
  }

  steer_angle = lf_mul(steer_angle, proc_steer);
  car.wheels[0].steer_angle = steer_angle;
  car.wheels[1].steer_angle = steer_angle;

  // fresh raycast contact must not produce sticky force within its own tick
  if (car.sticky_gate_prev) {
    // The wheel contacts are unchanged between the sticky force and auto-roll in one tick, so share one cached upwards direction.
    Vec3A upwards_dir;
    if (*cached_some) {
      upwards_dir = *cached_upwards_dir;
    } else {
      upwards_dir = get_upwards_dir_from_wheel_contacts(car, rb.world_trans.matrix3.z_axis);
      *cached_upwards_dir = upwards_dir;
      *cached_some = true;
    }

    // Sticky keeps raw throttle
    bool full_stick = raw_throttle != 0.0f || abs_forward_speed_uu > c::STOPPING_FORWARD_VEL;
    float sticky_force_scale = lf_mul(cc.three_wheels ? 0.0f : 1.0f, 0.5f);
    if (full_stick) sticky_force_scale = lf_add(sticky_force_scale, lf_sub(1.0f, lf_abs(upwards_dir.z)));

    rb_add_impulse_p1(rb, bc, IMPULSE_LINEAR,
                      upwards_dir * sticky_force_scale * -lf_abs(sd->gravity_uu.z) * c::TICK_TIME * c::UU_TO_BT, splat3(0.0f), false, true);
  }
}

// V3: car/base.rs:467 update_air_torque
LF_HDI void car_update_air_torque(ArenaState* s, const StaticData* sd, int ci, Body& rb, uint32_t nwic, bool prev_is_flipping,
                                  float prev_flip_time) {
  Car& car = s->cars[ci];
  const BodyConst& bc = sd->bodies[sd->cars[ci].rigid_body_idx];

  bool allow_dodge = nwic < 3;
  bool allow_air = nwic == 0;

  if (car.is_flipping && allow_dodge && !vec_is_zero(car.flip_rel_torque)) {
    Vec3A rel_dodge_torque = car.flip_rel_torque;

    float pitch_scale = 1.0f;
    if (rel_dodge_torque.y != 0.0f && car.controls.pitch != 0.0f && rs_signum(rel_dodge_torque.y) == rs_signum(car.controls.pitch) &&
        prev_flip_time >= c::FLIP_PITCH_CANCEL_GATE_MIN_TIME) {
      pitch_scale = lf_sub(1.0f, rs_min(lf_abs(quantize_axis_input(lf_neg(car.controls.pitch))), 1.0f));
    }

    rel_dodge_torque.y = lf_mul(rel_dodge_torque.y, pitch_scale);
    Vec3A dodge_torque = rel_dodge_torque * Vec3A{c::FLIP_TORQUE_X, c::FLIP_TORQUE_Y, c::FLIP_TORQUE_Z} * c::TICK_TIME;

    rb_add_impulse_p1(rb, bc, IMPULSE_ANGULAR, rb.world_trans.matrix3 * dodge_torque, splat3(0.0f), false, true);
  }

  bool do_air_control = allow_air;
  if (do_air_control) {
    Vec3A inputs = quantize_axis_inputs(Vec3A{car.controls.pitch, car.controls.yaw, car.controls.roll});

    float pitch_torque_scale = 1.0f;
    if (!vec_is_zero(inputs) && (prev_is_flipping || car.is_flipping || prev_flip_time < c::FLIP_PITCHLOCK_EXTRA_TIME)) {
      pitch_torque_scale = 0.0f;
    }

    Mat3A dirs = Mat3A{-car.rot_mat.y_axis,   // pitch: -right
                       car.rot_mat.z_axis,    // yaw: up
                       -car.rot_mat.x_axis};  // roll: -forward

    Vec3A scaled_inputs = inputs * Vec3A{pitch_torque_scale, 1.0f, 1.0f};
    Vec3A torque = !vec_is_zero(inputs) ? dirs * (scaled_inputs * Vec3A{c::AIR_TORQUE_X, c::AIR_TORQUE_Y, c::AIR_TORQUE_Z})
                                        : splat3(0.0f);

    Vec3A retain = splat3(1.0f) - glam::abs(scaled_inputs);
    retain.z = 1.0f;
    Vec3A damping = dirs * (mul_transpose_vec3a(dirs, car.ang_vel) * Vec3A{c::AIR_DAMPING_X, c::AIR_DAMPING_Y, c::AIR_DAMPING_Z} * retain);

    Vec3A rb_torque = (torque - damping) * c::AIR_TORQUE_APPLY_SCALE * c::TICK_TIME;

    rb_add_impulse_p1(rb, bc, IMPULSE_ANGULAR, rb_torque, splat3(0.0f), false, true);
  }

  if (allow_air) {
    float throttle_scale = car.is_boosting ? 1.0f : quantize_axis_input(car.controls.throttle);

    if (throttle_scale != 0.0f) {
      Vec3A throttle_force = car.rot_mat.x_axis * throttle_scale * c::THROTTLE_AIR_ACCEL * c::UU_TO_BT * c::TICK_TIME;
      rb_add_impulse_p1(rb, bc, IMPULSE_LINEAR, throttle_force, splat3(0.0f), false, true);
    }
  }
}

// V3: car/base.rs:559 update_jump
LF_HDI void car_update_jump(ArenaState* s, const StaticData* sd, int ci, Body& rb, bool jump_pressed) {
  Car& car = s->cars[ci];
  const BodyConst& bc = sd->bodies[sd->cars[ci].rigid_body_idx];

  Vec3A up_dir = car.rot_mat.z_axis;

  // Check jump activation
  if (!car.has_jumped && car.is_on_ground && jump_pressed) {
    car.is_jumping = 1;
    car.has_jumped = 1;
    car.jump_ticks = 0;
    car.jump_hold_broken = 0;
  }

  car.jump_ticks += 1;

  if (car.is_jumping) {
    // Sustain needs continuous hold. Released hold ends sustain after the minimum ticks. A re-press never resumes it.
    if (!car.controls.jump) car.jump_hold_broken = 1;

    car.is_jumping = (car.jump_ticks <= c::JUMP_MIN_TICKS ||
                      (!car.jump_hold_broken && car.controls.jump && car.jump_ticks <= c::JUMP_MAX_TICKS))
                         ? 1u
                         : 0u;
    if (!car.is_jumping) car.jump_ticks = 1;  // Jump ended this tick: counter restarts
  }

  // Apply forces (only if still jumping)
  if (car.is_jumping) {
    if (car.jump_ticks == 1) {
      // First tick of jumping: apply initial impulse.
      Vec3A jump_start_force = up_dir * sd->mut_f[c::MF_JUMP_IMMEDIATE_FORCE] * c::UU_TO_BT;
      rb_add_impulse_p1(rb, bc, IMPULSE_LINEAR, jump_start_force, splat3(0.0f), false, false);
      // Clamp speed after the impulse, as after a dodge impulse.
      rb_limit_vels(&rb, lf_mul(c::CAR_MAX_SPEED, c::UU_TO_BT), c::CAR_MAX_ANG_SPEED);
    }

    Vec3A jump_force = up_dir * sd->mut_f[c::MF_JUMP_ACCEL] * (c::UU_TO_BT * c::TICK_TIME);
    rb_add_impulse_p1(rb, bc, IMPULSE_LINEAR, jump_force, splat3(0.0f), false, true);
  }
}

// V3: car/base.rs:611 update_auto_flip
LF_HDI void car_update_auto_flip(ArenaState* s, const StaticData* sd, int ci, Body& rb, bool jump_pressed) {
  Car& car = s->cars[ci];
  const BodyConst& bc = sd->bodies[sd->cars[ci].rigid_body_idx];

  if (jump_pressed && car.world_contact_normal.some && car.world_contact_normal.v.z > c::AUTOFLIP_NORM_Z_THRESH) {
    Mat3A rot_mat = car.rot_mat;
    float y_z = rot_mat.y_axis.z;
    float z_z = rot_mat.z_axis.z;
    float cy = lf_sqrt(lf_add(lf_mul(z_z, z_z), lf_mul(y_z, y_z)));
    float roll = cy > lf_mul(16.0f, c::F32_EPSILON) ? lf::atan2(y_z, z_z) : 0.0f;

    float abs_roll = lf_abs(roll);
    if (abs_roll > c::AUTOFLIP_ROLL_THRESH) {
      car.auto_flip_timer = lf_mul(c::AUTOFLIP_TIME, lf_div(abs_roll, c::PI_F));
      car.auto_flip_torque_scale = autoflip_impulse_sign(sd, car.rot_mat);
      car.is_auto_flipping = 1;

      Vec3A force = -car.rot_mat.z_axis * (c::AUTOFLIP_IMPULSE * c::UU_TO_BT);
      rb_add_impulse_p1(rb, bc, IMPULSE_LINEAR, force, splat3(0.0f), false, false);
    }
  }

  if (car.is_auto_flipping) {
    if (car.auto_flip_timer <= 0.0f) {
      car.is_auto_flipping = 0;
      car.auto_flip_timer = 0.0f;
    } else {
      rb.ang_vel += car.rot_mat.x_axis * c::AUTOFLIP_ANG_VEL_DELTA * car.auto_flip_torque_scale;
      rb_limit_vels(&rb, lf_mul(c::CAR_MAX_SPEED, c::UU_TO_BT), c::CAR_MAX_ANG_SPEED);
      car.auto_flip_timer = lf_sub(car.auto_flip_timer, c::TICK_TIME);
    }
  }
}

// V3: car/base.rs:657 update_double_jump_or_flip; returns flip_ended
LF_HDI bool car_update_double_jump_or_flip(ArenaState* s, const StaticData* sd, int ci, Body& rb, bool jump_pressed, float forward_speed_uu,
                                           bool was_is_jumping) {
  Car& car = s->cars[ci];
  const BodyConst& bc = sd->bodies[sd->cars[ci].rigid_body_idx];

  if (car.is_on_ground) {
    car.has_double_jumped = 0;
    car.has_flipped = 0;
    car.air_time = 0.0f;
    car.air_time_since_jump = 0.0f;
    car.flip_time = lf_add(car.flip_time, c::TICK_TIME);
    return false;
  }

  car.air_time = lf_add(car.air_time, c::TICK_TIME);

  if (car.has_jumped && !was_is_jumping && !car.is_jumping) {
    car.air_time_since_jump = lf_add(car.air_time_since_jump, c::TICK_TIME);
  } else {
    car.air_time_since_jump = 0.0f;
  }

  if (jump_pressed && car.air_time_since_jump < c::JUMP_DOUBLEJUMP_MAX_DELAY) {
    const Vec3A PYR_SCALE = {-1.0f, 1.0f, 1.0f};
    Vec3A dodge_input = quantize_axis_inputs(Vec3A{car.controls.pitch, car.controls.yaw, car.controls.roll} * PYR_SCALE);
    float input_magnitude = lf_add(lf_add(lf_abs(dodge_input.x), lf_abs(dodge_input.y)), lf_abs(dodge_input.z));
    bool is_flip_input = input_magnitude >= s->cars[ci].dodge_deadzone;

    bool unlimited = is_flip_input ? (sd->unlimited_flips != 0) : (sd->unlimited_double_jumps != 0);
    bool can_use = (!car.is_auto_flipping && !car.has_double_jumped && !car.has_flipped) || unlimited;

    if (can_use) {
      car.has_jumped = 1;

      if (is_flip_input) {
        car.flip_time = 0.0f;
        car.has_flipped = 1;
        car.is_flipping = 1;

        Vec3A dodge_dir = dodge_input;
        dodge_dir.y = lf_add(dodge_dir.y, dodge_dir.z);

        if (lf_abs(dodge_dir.x) < 0.1f && lf_abs(dodge_dir.y) < 0.1f) {
          dodge_dir = splat3(0.0f);
        } else {
          dodge_dir.z = 0.0f;
          dodge_dir = normalize_or_zero(dodge_dir);
        }

        Vec3A deadzone_dodge_dir = glam::select(glam::cmplt(glam::abs(dodge_dir), splat3(0.1f)), splat3(0.0f), dodge_dir);

        if (length_squared(deadzone_dodge_dir) > lf_mul(c::F32_EPSILON, c::F32_EPSILON)) {
          car.flip_rel_torque = Vec3A{lf_neg(dodge_dir.y), dodge_dir.x, 0.0f};

          bool should_dodge_backwards = lf_abs(forward_speed_uu) < 100.0f ? (lf_bits(deadzone_dodge_dir.x) >> 31) != 0
                                                                          : rs_signum(deadzone_dodge_dir.x) != rs_signum(forward_speed_uu);

          float max_speed_scale_x = should_dodge_backwards ? c::FLIP_BACKWARD_IMPULSE_MAX_SPEED_SCALE : c::FLIP_FORWARD_IMPULSE_MAX_SPEED_SCALE;

          float forward_speed_ratio = lf_div(lf_abs(forward_speed_uu), c::CAR_MAX_SPEED);

          Vec3A initial_dodge_vel = deadzone_dodge_dir * c::FLIP_INITIAL_VEL_SCALE;
          initial_dodge_vel.x = lf_mul(initial_dodge_vel.x, lf_add(lf_mul(lf_sub(max_speed_scale_x, 1.0f), forward_speed_ratio), 1.0f));
          initial_dodge_vel.y = lf_mul(initial_dodge_vel.y, lf_add(lf_mul(lf_sub(c::FLIP_SIDE_IMPULSE_MAX_SPEED_SCALE, 1.0f), forward_speed_ratio), 1.0f));
          if (should_dodge_backwards) initial_dodge_vel.x = lf_mul(initial_dodge_vel.x, c::FLIP_BACKWARD_IMPULSE_SCALE_X);

          Vec3A fd = car.rot_mat.x_axis;
          fd.z = 0.0f;
          Vec3A forward_dir_2d = normalize_or_zero(fd);
          Vec3A right_dir_2d = Vec3A{lf_neg(forward_dir_2d.y), forward_dir_2d.x, 0.0f};
          Vec3A final_delta_vel = initial_dodge_vel.x * forward_dir_2d + initial_dodge_vel.y * right_dir_2d;

          rb_add_impulse_p1(rb, bc, IMPULSE_LINEAR, final_delta_vel * c::UU_TO_BT, splat3(0.0f), false, false);
          rb_limit_vels(&rb, lf_mul(c::CAR_MAX_SPEED, c::UU_TO_BT), c::CAR_MAX_ANG_SPEED);
        }
      } else {
        Vec3A jump_start_force = car.rot_mat.z_axis * sd->mut_f[c::MF_JUMP_IMMEDIATE_FORCE] * c::UU_TO_BT;
        rb_add_impulse_p1(rb, bc, IMPULSE_LINEAR, jump_start_force, splat3(0.0f), false, false);
        // Clamp speed after the impulse, as after a dodge impulse.
        rb_limit_vels(&rb, lf_mul(c::CAR_MAX_SPEED, c::UU_TO_BT), c::CAR_MAX_ANG_SPEED);
        car.has_double_jumped = 1;
      }
    }
  }

  if (car.is_flipping) {
    float flip_time_pre = car.flip_time;
    bool still_flipping = car.has_flipped && flip_time_pre < c::FLIP_TORQUE_TIME;
    car.is_flipping = still_flipping ? 1u : 0u;
    car.flip_time = still_flipping ? lf_add(flip_time_pre, c::TICK_TIME) : c::TICK_TIME;

    return !still_flipping;
  }

  car.flip_time = lf_add(car.flip_time, c::TICK_TIME);

  return false;
}

// V3: car/base.rs:789 update_auto_roll
LF_HDI void car_update_auto_roll(ArenaState* s, const StaticData* sd, int ci, Body& rb, uint32_t nwic, bool* cached_some,
                                 Vec3A* cached_upwards_dir) {
  Car& car = s->cars[ci];
  const BodyConst& bc = sd->bodies[sd->cars[ci].rigid_body_idx];

  if (!car.sticky_gate_prev && !car.world_contact_normal.some) return;

  Vec3A ground_up_dir;
  if (car.sticky_gate_prev && nwic > 0) {
    // Same contacts as the sticky force above: reuse its cached direction.
    if (*cached_some) {
      ground_up_dir = *cached_upwards_dir;
    } else {
      ground_up_dir = get_upwards_dir_from_wheel_contacts(car, rb.world_trans.matrix3.z_axis);
      *cached_upwards_dir = ground_up_dir;
      *cached_some = true;
    }
  } else {
    ground_up_dir = car.world_contact_normal.v;
  }

  Vec3A ground_down_dir = -ground_up_dir;

  Vec3A forward_dir = car.rot_mat.x_axis;
  Vec3A right_dir = car.rot_mat.y_axis;

  Vec3A cross_right_dir = cross(ground_up_dir, forward_dir);
  Vec3A cross_forward_dir = cross(ground_down_dir, cross_right_dir);

  float right_torque_factor = lf_sub(1.0f, rs_clamp(dot(right_dir, cross_right_dir), 0.0f, 1.0f));
  float forward_torque_factor = lf_sub(1.0f, rs_clamp(dot(forward_dir, cross_forward_dir), 0.0f, 1.0f));

  Vec3A torque_dir_right = forward_dir * lf_neg(rs_signum(dot(right_dir, ground_up_dir)));
  Vec3A torque_dir_forward = right_dir * rs_signum(dot(forward_dir, ground_up_dir));

  Vec3A torque_right = torque_dir_right * right_torque_factor;
  Vec3A torque_forward = torque_dir_forward * forward_torque_factor;

  rb_add_impulse_p1(rb, bc, IMPULSE_LINEAR, ground_down_dir * (c::AUTOROLL_FORCE * c::UU_TO_BT * c::TICK_TIME), splat3(0.0f), false, true);

  rb_add_impulse_p1(rb, bc, IMPULSE_ANGULAR, (torque_forward + torque_right) * (c::AUTOROLL_TORQUE * c::TICK_TIME), splat3(0.0f), false, true);
}

// V3: car/base.rs:848 update_boost
LF_HDI void car_update_boost(ArenaState* s, const StaticData* sd, int ci, Body& rb) {
  Car& car = s->cars[ci];
  const BodyConst& bc = sd->bodies[sd->cars[ci].rigid_body_idx];
  const float accel_ground = sd->mut_f[c::MF_BOOST_ACCEL_GROUND], accel_air = sd->mut_f[c::MF_BOOST_ACCEL_AIR];

  if (car.is_boosting) {
    float cost = lf_mul(sd->mut_f[c::MF_BOOST_USED_PER_SECOND], c::TICK_TIME);
    car.boost = rs_max(lf_sub(car.boost, cost), 0.0f);
    bool depleted = car.boost == 0.0f;
    bool latch_expired = !car.controls.boost && car.boosting_time >= c::BOOST_MIN_TIME;
    if (depleted || latch_expired) {
      car.is_boosting = 0;
      car.boosting_time = 0.0f;
      car.time_since_boosted = lf_add(car.time_since_boosted, c::TICK_TIME);

      if (sd->recharge_boost_enabled && car.time_since_boosted >= sd->recharge_boost_delay) {
        car.boost = lf_add(car.boost, lf_mul(sd->recharge_boost_per_second, c::TICK_TIME));
      }
    } else {
      car.is_boosting = 1;
      car.boosting_time = lf_add(car.boosting_time, c::TICK_TIME);
      car.time_since_boosted = 0.0f;
      float accel = car.is_on_ground ? accel_ground : accel_air;

      rb_add_impulse_p1(rb, bc, IMPULSE_LINEAR, accel * car.rot_mat.x_axis * (c::UU_TO_BT * c::TICK_TIME), splat3(0.0f), false, true);
    }
  } else if (car.controls.boost && car.boost > 0.0f) {
    car.is_boosting = 1;
    car.boosting_time = lf_add(car.boosting_time, c::TICK_TIME);
    car.time_since_boosted = 0.0f;
    float accel = car.is_on_ground ? accel_ground : accel_air;

    rb_add_impulse_p1(rb, bc, IMPULSE_LINEAR, accel * car.rot_mat.x_axis * (c::UU_TO_BT * c::TICK_TIME), splat3(0.0f), false, true);
  } else {
    car.is_boosting = 0;
    car.boosting_time = 0.0f;
    car.time_since_boosted = lf_add(car.time_since_boosted, c::TICK_TIME);

    if (sd->recharge_boost_enabled && car.time_since_boosted >= sd->recharge_boost_delay) {
      car.boost = lf_add(car.boost, lf_mul(sd->recharge_boost_per_second, c::TICK_TIME));
    }
  }

  car.boost = rs_clamp(car.boost, 0.0f, sd->mut_f[c::MF_CAR_MAX_BOOST_AMOUNT]);  // car/base.rs:908-911
}

// V3: car/base.rs:1038 update_contact_state
LF_HDI void car_update_contact_state(ArenaState* s, const StaticData* sd, int ci) {
  Car& car = s->cars[ci];
  uint32_t num = 0;
  for (int i = 0; i < NUM_WHEELS; ++i) {
    const Wheel& wheel = car.wheels[i];
    Opt<HitInfo> hit = Opt<HitInfo>{};
    if (wheel.raycast_info.some) {
      const RaycastInfo& info = wheel.raycast_info.v;
      // Wheel traces run in Bullet units; report uu like `cast_rays`.
      float trace_len = length(wheel.hard_point - info.contact_point);
      hit.some = 1;
      hit.v.hit_point = info.contact_point * c::BT_TO_UU;
      hit.v.hit_normal = info.contact_normal;
      hit.v.hit_fraction = lf_div(trace_len, sd->cars[ci].wheels[i].real_ray_length);
      hit.v.user_info = sd->bodies[info.ground_body_idx].user_idx;
    }
    car.wheels_with_contact[i] = hit;
    num += hit.some ? 1u : 0u;
  }
  car.is_on_ground = num >= 3 ? 1u : 0u;
}

// V3: car/base.rs:913 apply_flip_zdamp_prepass
LF_HDI void car_apply_flip_zdamp_prepass(ArenaState* s, const StaticData* sd, int ci) {
  Car& car = s->cars[ci];
  if (car.is_demoed || car.is_on_ground || !car.is_flipping) return;

  // !(Z_DAMP_START..=TORQUE_TIME).contains(&flip_time)
  if (!(car.flip_time >= c::FLIP_Z_DAMP_START && car.flip_time <= c::FLIP_TORQUE_TIME)) return;

  Body& rb = s->bodies[dyn_slot_p1(sd, sd->cars[ci].rigid_body_idx)];
  if (rb.lin_vel.z < 0.0f || car.flip_time < c::FLIP_Z_DAMP_END) {
    rb.lin_vel.z = lf_mul(rb.lin_vel.z, lf_sub(1.0f, c::FLIP_Z_DAMP_120));
  }
}

// V3: car/base.rs:930 pre_tick_update
LF_HDI void car_pre_tick_update(ArenaState* s, const StaticData* sd, int ci) {
  Car& car = s->cars[ci];
  const CarConst& cc = sd->cars[ci];
  Body& rb = s->bodies[dyn_slot_p1(sd, cc.rigid_body_idx)];

  if (car.is_demoed) {
    car.demo_respawn_timer = rs_max(lf_sub(car.demo_respawn_timer, c::TICK_TIME), 0.0f);
    if (car.demo_respawn_timer == 0.0f) car_respawn(s, sd, ci, sd->mut_f[c::MF_CAR_SPAWN_BOOST_AMOUNT]);

    if (rb.activation != ACT_DISABLED) rb.activation = ACT_DISABLED;  // set_activation_state(DisableSimulation)
    rb.collision_flags |= CF_NO_CONTACT_RESPONSE;
    return;
  }

  rb_force_activate(rb);  // rigid_body.rs:344-347
  rb.collision_flags &= ~CF_NO_CONTACT_RESPONSE;
  car.controls.throttle = rs_clamp(car.controls.throttle, -1.0f, 1.0f);  // CarControls::clamp
  car.controls.steer = rs_clamp(car.controls.steer, -1.0f, 1.0f);
  car.controls.pitch = rs_clamp(car.controls.pitch, -1.0f, 1.0f);
  car.controls.yaw = rs_clamp(car.controls.yaw, -1.0f, 1.0f);
  car.controls.roll = rs_clamp(car.controls.roll, -1.0f, 1.0f);

  float forward_speed_uu = lf_mul(dot(rb.lin_vel, rb.world_trans.matrix3.x_axis), c::BT_TO_UU);
  bool jump_pressed = car.controls.jump && !car.prev_controls.jump;
  uint32_t nwic = num_wheels_in_contact(car);

  // The wheel contacts only change in `vehicle_update` below, so the sticky force and auto-roll share one upwards direction.
  bool cached_some = false;
  Vec3A cached_upwards_dir = splat3(0.0f);
  car_update_wheels(s, sd, ci, rb, forward_speed_uu, &cached_some, &cached_upwards_dir);

  if (car.is_on_ground) {
    if (car.is_flipping) car.flip_time = 0.0f;
    car.is_flipping = 0;
  }

  bool was_is_jumping = car.is_jumping != 0;
  car_update_jump(s, sd, ci, rb, jump_pressed);
  car_update_auto_flip(s, sd, ci, rb, jump_pressed);
  bool prev_is_flipping = car.is_flipping != 0;
  float prev_flip_time = car.flip_time;
  bool flip_ended = car_update_double_jump_or_flip(s, sd, ci, rb, jump_pressed, forward_speed_uu, was_is_jumping);

  if (!car.is_on_ground) car_update_air_torque(s, sd, ci, rb, nwic, prev_is_flipping, prev_flip_time);

  float quant_throttle = quantize_axis_input(car.controls.throttle);

  // Skip auto-roll on the tick a flip ends. The car still counts as flipping for auto-roll on its last flip tick.
  if (quant_throttle != 0.0f && !car.is_flipping && !flip_ended && ((0 < nwic && nwic < 4) || car.world_contact_normal.some)) {
    car_update_auto_roll(s, sd, ci, rb, nwic, &cached_some, &cached_upwards_dir);
  }

  car.world_contact_normal = Opt<Vec3A>{};

  car_update_boost(s, sd, ci, rb);

  float real_throttle = (car.controls.boost && car.boost > 0.0f) ? 1.0f : quant_throttle;
  vehicle_update(s, sd, ci, c::TICK_TIME, car.handbrake_val, real_throttle, cc.three_wheels != 0);

  car_update_contact_state(s, sd, ci);

  uint32_t any_world = 0;
  for (int i = 0; i < NUM_WHEELS; ++i)
    if (car.wheels[i].raycast_info.some && car.wheels[i].raycast_info.v.is_in_contact_with_world) any_world = 1;
  car.sticky_gate_prev = any_world;
}

}  // namespace lfsim
