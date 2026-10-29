// V3 bullet/dynamics/discrete_dynamics_world.rs: phases P2 and P6.
#pragma once
#include "sim/rigid_body.cuh"

namespace lfsim {

constexpr float TICK_TIME = 1.0f / 120.0f;  // sim/consts.rs:45 TICK_TIME = 1 / TICK_RATE (f32)

// V3: discrete_dynamics_world.rs:139 apply_gravity
LF_HD inline void ddw_apply_gravity(ArenaState* s, const StaticData* sd, float time_step) {
  Vec3A impulse = sd->gravity_bt * time_step;
  for (int i = 0; i < NUM_DYN; ++i)
    if (rb_is_active(s->bodies[i])) rb_add_linear_impulse_accum(s->bodies[i], impulse);
}

// V3: discrete_dynamics_world.rs:148 predict_unconstraint_motion
LF_HD inline void ddw_predict_unconstraint_motion(ArenaState* s, const StaticData* sd, float time_step) {
  for (int i = 0; i < NUM_DYN; ++i) {
    Body& b = s->bodies[i];
    if (b.activation == ACT_SLEEPING) {  // :152-155
      b.interp_world_trans = b.world_trans;
      continue;
    }
    rb_apply_damping(b, sd->bodies[sd->dyn_body[i]], time_step);
    b.interp_world_trans = rb_predict_integration_trans(b, time_step);
  }
}

// V3: discrete_dynamics_world.rs:170 integrate_trans_internal (:182 integrate_trans: the dynamic list is never empty here)
LF_HD inline void ddw_integrate_trans(ArenaState* s, const StaticData* sd, float time_step) {
  for (int i = 0; i < NUM_DYN; ++i) {
    Body& b = s->bodies[i];
    if (!rb_is_active(b)) continue;
    Affine3A predicted = rb_predict_integration_trans(b, time_step);
    rb_set_center_of_mass_trans(b, sd->bodies[sd->dyn_body[i]], predicted);
  }
}

// V3: discrete_dynamics_world.rs:188 update_activation_state
LF_HD inline void ddw_update_activation_state(ArenaState* s, const StaticData* sd, float time_step) {
  for (int i = 0; i < NUM_DYN; ++i) rb_update_activation_state(s->bodies[i], sd->bodies[sd->dyn_body[i]], time_step);
}

// V3: discrete_dynamics_world.rs:255 trace_p2 (step_simulation :247, :215)
LF_HD inline void ddw_phase_p2(ArenaState* s, const StaticData* sd) {
  ddw_apply_gravity(s, sd, TICK_TIME);
  ddw_predict_unconstraint_motion(s, sd, TICK_TIME);
}

// V3: discrete_dynamics_world.rs:301 trace_p6 (:238-239)
LF_HD inline void ddw_phase_p6(ArenaState* s, const StaticData* sd) {
  ddw_integrate_trans(s, sd, TICK_TIME);
  ddw_update_activation_state(s, sd, TICK_TIME);
}

}  // namespace lfsim
