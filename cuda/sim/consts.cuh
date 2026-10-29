// V3 constants used by the game-logic phases P0, P7, P8 (sim/consts.rs of rev ad8b38c). f32 const expressions are written with
// the same f32 operations Rust's const evaluator uses (correctly rounded), so every value is the f32 V3 holds.
#pragma once
#include "common/glam.cuh"
#include "common/hd.cuh"

namespace lfsim {
using namespace glam;  // the Vec3A operators live in glam
namespace c {

constexpr float BT_TO_UU = 50.0f;                    // V3: consts.rs:36
constexpr float UU_TO_BT = 1.0f / 50.0f;             // V3: consts.rs:39
constexpr float TICK_RATE = 120.0f;                  // V3: consts.rs:42
constexpr float TICK_TIME = 1.0f / TICK_RATE;        // V3: consts.rs:45

constexpr float CAR_MAX_SPEED = 2300.0f;             // V3: consts.rs:111
constexpr float CAR_MAX_ANG_SPEED = 5.5f;            // V3: consts.rs:131
constexpr float SUPERSONIC_START_SPEED = 2200.0f;    // V3: consts.rs:134
constexpr float SUPERSONIC_MAINTAIN_MIN_SPEED = 2200.0f - 100.0f;  // V3: consts.rs:135
constexpr float SUPERSONIC_MAINTAIN_MAX_TIME = 1.0f; // V3: consts.rs:136
constexpr float DEMO_YAW_LIMIT_DEG = 45.572994f;     // V3: consts.rs:140
constexpr float DEMO_PITCH_LIMIT_DEG = 36.869896f;   // V3: consts.rs:141
constexpr uint32_t JUMP_SETTLED_REARM_TICKS = 15;    // V3: consts.rs:173
constexpr float BUMP_COOLDOWN_TIME = 0.25f;          // V3: consts.rs:225 (car::bump)
constexpr float BUMP_YAW_LIMIT_DEG = 70.0f;          // V3: consts.rs:226
constexpr float BUMP_PITCH_LIMIT_DEG = 36.869896f;   // V3: consts.rs:227
constexpr float BALL_MAX_ANG_SPEED = 6.0f;           // V3: consts.rs:335
constexpr float HIT_FORWARD_SCALE = 0.65f;           // V3: consts.rs:350
constexpr float HIT_MAX_DELTA_VEL_UU = 4600.0f;      // V3: consts.rs:351
constexpr float HIT_Z_SCALE_NORMAL = 0.35f;          // V3: consts.rs:353
constexpr float BUMP_ATTACKER_SPEED_MAX = 2200.0f;   // V3: arena/base.rs:1571 (.min(2200.0))
constexpr float CYL_HEIGHT = 95.0f;                  // V3: consts.rs:559

constexpr float QUANT_POS_SCALE = 100.0f;            // V3: consts.rs:660
constexpr float QUANT_VEL_SCALE = 100.0f;            // V3: consts.rs:661
constexpr float QUANT_ANG_VEL_SCALE = 10000.0f;      // V3: consts.rs:662
constexpr float QUANT_OFFSET_CORRECT_FRAC = 0.1f;    // V3: shared/quantize.rs:38

constexpr float PI_F = 3.14159265358979323846264338327950288f;  // core::f32::consts::PI

// MutatorConfig floats in StaticData.mut_f (in that order).
enum MutF { MF_CAR_MASS, MF_BALL_MASS, MF_BALL_MAX_SPEED, MF_BALL_DRAG, MF_JUMP_ACCEL, MF_JUMP_IMMEDIATE_FORCE,
            MF_BOOST_ACCEL_GROUND, MF_BOOST_ACCEL_AIR, MF_BOOST_USED_PER_SECOND, MF_RESPAWN_DELAY, MF_BUMP_COOLDOWN_TIME,
            MF_CAR_MAX_BOOST_AMOUNT, MF_CAR_SPAWN_BOOST_AMOUNT, MF_PAD_AMOUNT_SMALL, MF_PAD_AMOUNT_BIG, MF_PAD_COOLDOWN_BIG,
            MF_PAD_COOLDOWN_SMALL, MF_BALL_HIT_EXTRA_FORCE_SCALE, MF_BUMP_FORCE_SCALE, MF_BALL_RADIUS };
enum DemoMode { DEMO_NORMAL = 0, DEMO_ON_CONTACT = 1, DEMO_DISABLED = 2 };       // mutator_config.rs:8
enum UserInfo { UI_NONE = 0, UI_CAR = 1, UI_BALL = 2, UI_DROPSHOT_TILE = 3 };    // user_info_types.rs

// ---- P1: car pre-tick, vehicle -------------------------------------------------------------------------------------
constexpr float F32_EPSILON = 1.1920928955078125e-7f;  // f32::EPSILON
constexpr float F32_MAX = 3.40282347e+38f;             // f32::MAX
constexpr float BOOST_MAX = 100.0f;                    // V3: consts.rs:115
constexpr float BOOST_MIN_TIME = 0.1f;                 // V3: consts.rs:119
constexpr float THROTTLE_TORQUE_AMOUNT = 180.0f * 400.0f;                    // V3: consts.rs:145 (MASS_BT = 180)
constexpr float BRAKE_TORQUE_AMOUNT = 180.0f * (14.25f + (1.0f / 3.0f));     // V3: consts.rs:146
constexpr float STOPPING_FORWARD_VEL = 25.0f;
constexpr float COASTING_BRAKE_FACTOR = 0.15f;
constexpr float BRAKING_NO_THROTTLE_SPEED_THRESH = 0.01f;
constexpr float THROTTLE_DEADZONE = 0.01f;
constexpr float THROTTLE_AIR_ACCEL = 200.0f / 3.0f;
constexpr float POWERSLIDE_RISE_RATE = 5.0f;
constexpr float POWERSLIDE_FALL_RATE = 2.0f;           // V3: consts.rs:157-158
constexpr uint32_t JUMP_MIN_TICKS = 3;                 // V3: consts.rs:161 (car::jump)
constexpr uint32_t JUMP_MAX_TICKS = 24;
constexpr float JUMP_DOUBLEJUMP_MAX_DELAY = 1.25f;
constexpr float FLIP_Z_DAMP_120 = 0.35f;               // V3: consts.rs:178 (car::flip)
constexpr float FLIP_Z_DAMP_START = 0.15f;
constexpr float FLIP_Z_DAMP_END = 0.21f;
constexpr float FLIP_TORQUE_TIME = 0.65f;
constexpr float FLIP_PITCH_CANCEL_GATE_MIN_TIME = 0.041f;
constexpr float FLIP_PITCHLOCK_EXTRA_TIME = 0.3f;
constexpr float FLIP_INITIAL_VEL_SCALE = 500.0f;
constexpr float FLIP_TORQUE_X = 260.0f, FLIP_TORQUE_Y = 224.0f, FLIP_TORQUE_Z = 0.0f;
constexpr float FLIP_FORWARD_IMPULSE_MAX_SPEED_SCALE = 1.0f;
constexpr float FLIP_SIDE_IMPULSE_MAX_SPEED_SCALE = 1.9f;
constexpr float FLIP_BACKWARD_IMPULSE_MAX_SPEED_SCALE = 2.5f;
constexpr float FLIP_BACKWARD_IMPULSE_SCALE_X = 16.0f / 15.0f;
constexpr float AIR_TORQUE_X = 130.0f, AIR_TORQUE_Y = 95.0f, AIR_TORQUE_Z = 400.0f;   // V3: consts.rs:199 (car::air_control)
constexpr float AIR_DAMPING_X = 30.0f, AIR_DAMPING_Y = 20.0f, AIR_DAMPING_Z = 50.0f;
constexpr float AIR_TORQUE_APPLY_SCALE = 2.0f * PI_F / 65536.0f * 1000.0f;
constexpr float AUTOFLIP_IMPULSE = 200.0f;             // V3: consts.rs:209 (car::autoflip)
constexpr float AUTOFLIP_ANG_VEL_DELTA = 1000.0f;
constexpr float AUTOFLIP_TIME = 0.4f;
constexpr float AUTOFLIP_NORM_Z_THRESH = 0.707106781186547524400844362104849039f;  // FRAC_1_SQRT_2
constexpr float AUTOFLIP_ROLL_THRESH = 2.8f;
constexpr float AUTOROLL_FORCE = 100.0f;               // V3: consts.rs:219
constexpr float AUTOROLL_TORQUE = 80.0f;
constexpr float RESPAWN_Z = 36.0f;                     // V3: consts.rs:230 (car::spawn)
constexpr float RESPAWN_TIME = 3.0f;                   // V3: consts.rs:278 (car::spawn)
constexpr float SOCCAR_GOAL_HEIGHT = 642.775f;         // V3: consts.rs:366 (goal)
constexpr float SOCCAR_GOAL_HALF_WIDTH = 892.755f;     // V3: consts.rs:367
// V3: car_body_config.rs:6, 17 (OCTANE): the hitbox's full size and its offset from the car's origin (uu)
constexpr float OCTANE_HITBOX_SIZE_X = 120.507f, OCTANE_HITBOX_SIZE_Y = 86.6994f, OCTANE_HITBOX_SIZE_Z = 38.6591f;
constexpr float OCTANE_HITBOX_OFFSET_X = 13.8757f, OCTANE_HITBOX_OFFSET_Y = 0.0f, OCTANE_HITBOX_OFFSET_Z = 20.755f;
constexpr float SUSPENSION_STIFFNESS = 500.0f;         // V3: consts.rs:400 (bullet_vehicle)
constexpr float WHEELS_DAMPING_COMPRESSION = 25.0f;
constexpr float WHEELS_DAMPING_RELAXATION = 40.0f;
constexpr float MAX_SUSPENSION_TRAVEL = 12.0f;
constexpr float SUSPENSION_SUBTRACTION = 0.05f;
constexpr float WHEEL_PUSHBACK_ERP = 0.1f;             // V3: contact_solver_info.rs:3

}  // namespace c
}  // namespace lfsim
