// P5 (phase index 6): V3's SeqImpulseConstraintSolver::solve_group.
//
// V3 sources (rev ad8b38c, paths under rocketsim/src/bullet/):
//   dynamics/constraint_solver/seq_impulse_constraint_solver.rs   setup, special rows, iterations, finish
//   dynamics/discrete_dynamics_world.rs:232-237 and :152-160 (the call: time_step = TICK_TIME = 1/120)
//   linear_math/transform_util.rs:8-37 integrate_trans, integrate_trans_no_rot
//   dynamics/rigid_body.rs:266-274 set_world_trans
//
// Differences from V3, each identical in the Soccar 1v1 arena:
//   * skip_separated_special_rows is false (it is set only for a ball-only arena), so the *_one_dynamic paths and the
//     "first accumulator" shortcut are not ported.
//   * Warm start is omitted (solver_constraint.cuh header explains why it is the identity).
//   * companion_id lives in `companion[d]` for the three dynamic bodies; a static body never reads its companion_id before the
//     solver assigns it (get_or_init_solver_body returns the fixed id if it exists, else makes it).
//   * The branch `rb.inv_mass == 0.0` of get_or_init_solver_body (a dynamic body with no mass) cannot happen (debug_assert_ne
//     in setup_solver_bodies); a dynamic body that is not active falls into the same new-body path as in V3.
//   * Rows are local arrays capped at MAX_SOLVER_ROWS; a row that would not fit is dropped and overflow[OVF_ACTIVE] is bumped
//     (ArenaState has no solver counter; the layout is not mine to change). The cap is never reached in the 40M-tick sizing.
//   * The assert!(distance_1 <= contact_processing_threshold) of the point loop (a V3 panic) is not ported.
//   * Impulses are not written back to manifold points: V3's solve_group_finish does not.
#pragma once
#include "common/libm.cuh"
#include "sim/arena_state.cuh"
#include "sim/simulation_island_manager.cuh"
#include "sim/solver_constraint.cuh"

namespace lfsim {

constexpr uint32_t FLAG_NO_ANGULAR_MOTION = 1u << 5;  // rigid_body.rs:56 CollisionFlags::NoAngularMotion
constexpr int MAX_SPECIAL = 4;                         // accumulators: one per dynamic body (3)

// seq_impulse_constraint_solver.rs:20-27 SpecialAccumulator (the SpecialSample is added directly)
struct SpecialAccumulator {
  int obj_idx;  // dynamic index of the body (V3 keys on the world index; the map is one-to-one)
  Vec3A total_normal;
  float total_lever_len, total_friction, total_restitution;
  uint32_t count;
};

// The local state of one solve (V3's SeqImpulseConstraintSolver fields).
struct SolverCtx {
  SolverBody bodies[MAX_SOLVER_BODIES];
  SolverConstraint contact[MAX_SOLVER_ROWS];
  SolverConstraint friction[MAX_SOLVER_ROWS];
  SpecialAccumulator special[MAX_SPECIAL];
  int n_bodies, n_contact, n_friction, n_special;
  int fixed_body_id;
  int companion[NUM_DYN];
};

LF_HDI bool body_is_active(const Body& b) { return b.activation != 1 && b.activation != 2; }  // rigid_body.rs:292 (Sleeping, DisableSimulation)

// seq_impulse_constraint_solver.rs:111-145 get_or_init_solver_body for the world body w (dynamic index d, -1 if static)
LF_HDI int get_or_init_solver_body(SolverCtx& cx, ArenaState* s, const StaticData* sd, int d, uint32_t w) {
  if (d < 0) {  // is_static_obj
    if (cx.fixed_body_id >= 0) return cx.fixed_body_id;
    int id = cx.n_bodies;
    cx.fixed_body_id = id;
    cx.bodies[cx.n_bodies++] = solver_body_default();
    return id;
  }
  if (cx.companion[d] >= 0) return cx.companion[d];
  int id = cx.n_bodies;  // inv_mass != 0 (debug_assert in setup_solver_bodies)
  cx.companion[d] = id;
  cx.bodies[cx.n_bodies++] = solver_body_new(s->bodies[d], sd->bodies[w], d);
  return id;
}

// seq_impulse_constraint_solver.rs:321-352 setup_solver_bodies
LF_HDI void setup_solver_bodies(SolverCtx& cx, ArenaState* s, const StaticData* sd) {
  cx.fixed_body_id = NO_BODY;
  for (int d = 0; d < NUM_DYN; ++d) {
    cx.companion[d] = NO_BODY;
    if (!body_is_active(s->bodies[d])) continue;
    cx.companion[d] = cx.n_bodies;
    cx.bodies[cx.n_bodies++] = solver_body_new(s->bodies[d], sd->bodies[sd->dyn_body[d]], d);
  }
}

// seq_impulse_constraint_solver.rs:37-61 accumulate_special_sample: first-seen order, one entry per body
LF_HDI void accumulate_special_sample(SolverCtx& cx, int obj, Vec3A normal, float lever_len, float friction, float restitution) {
  for (int i = 0; i < cx.n_special; ++i) {
    SpecialAccumulator& a = cx.special[i];
    if (a.obj_idx == obj) {
      a.total_normal += normal;
      a.total_lever_len = lf_add(a.total_lever_len, lever_len);
      a.total_friction = lf_add(a.total_friction, friction);
      a.total_restitution = lf_add(a.total_restitution, restitution);
      a.count += 1;
      return;
    }
  }
  cx.special[cx.n_special++] = SpecialAccumulator{obj, normal, lever_len, friction, restitution, 1u};
}

// :362-435 push_special_synthetic, and :354-360 emit_special_synthetics (in first-seen order)
LF_HDI void push_special_synthetic(SolverCtx& cx, ArenaState* s, const StaticData* sd, const SpecialAccumulator& acc, float time_step) {
  float num_samples = lf_u2f(acc.count);
  Vec3A mean_normal = normalize_or_zero(acc.total_normal / num_samples);
  float mean_lever_len = lf_div(acc.total_lever_len, num_samples);
  float mean_friction = lf_div(acc.total_friction, num_samples);
  float mean_restitution = lf_div(acc.total_restitution, num_samples);

  int d = acc.obj_idx;
  int id_a = cx.companion[d];
  int id_b;
  if (cx.fixed_body_id >= 0) {
    id_b = cx.fixed_body_id;
  } else {
    id_b = cx.n_bodies;
    cx.fixed_body_id = id_b;
    cx.bodies[cx.n_bodies++] = solver_body_default();
  }
  const SolverBody& sa = cx.bodies[id_a];
  const SolverBody& sb = cx.bodies[id_b];

  Vec3A rel_pos1 = mean_normal * lf_neg(mean_lever_len);  // a radial lever
  Vec3A rel_pos2 = {0.0f, 0.0f, 0.0f};
  ContactPoint cp = {mean_normal, mean_friction, mean_restitution, 0.0f};

  RbSide s0 = {sa.has_orig ? &s->bodies[d] : nullptr, &sd->bodies[sd->dyn_body[d]]};
  RbSide s1 = {nullptr, nullptr};
  int friction_idx = cx.n_contact;
  if (cx.n_contact >= MAX_SOLVER_ROWS || cx.n_friction >= MAX_SOLVER_ROWS) { s->overflow[OVF_ACTIVE]++; return; }
  cx.contact[cx.n_contact++] = get_contact_constraint(id_a, id_b, sa, sb, s0, s1, rel_pos1, rel_pos2, cp, friction_idx, time_step);
  Vec3A lat = calc_lat_friction_dir(cp, sa, sb, rel_pos1, rel_pos2);
  cx.friction[cx.n_friction++] = get_friction_constraint(id_a, id_b, sa, sb, s0, s1, rel_pos1, rel_pos2, cp.combined_friction, lat, friction_idx);
}

// :188-319 solve_group_setup
LF_HDI void solve_group_setup(SolverCtx& cx, ArenaState* s, const StaticData* sd, float time_step) {
  setup_solver_bodies(cx, s, sd);
  cx.n_special = 0;

  for (uint32_t ai = 0; ai < s->n_active; ++ai) {
    const Manifold& m = s->manifolds[s->active[ai]];
    int d0 = dyn_index_of(sd, m.body0_idx), d1 = dyn_index_of(sd, m.body1_idx);

    int id_a = get_or_init_solver_body(cx, s, sd, d0, m.body0_idx);
    int id_b = get_or_init_solver_body(cx, s, sd, d1, m.body1_idx);
    // body0.companion_id = Some(id_a) ... : for a dynamic body that is the value it already has

    Vec3A trans0 = d0 >= 0 ? s->bodies[d0].world_trans.translation : sd->bodies[m.body0_idx].pose.translation;
    Vec3A trans1 = d1 >= 0 ? s->bodies[d1].world_trans.translation : sd->bodies[m.body1_idx].pose.translation;

    for (uint32_t pi = 0; pi < m.point_count; ++pi) {
      const ManifoldPoint& cp = m.points[pi];
      Vec3A rel_pos1 = cp.pos_world_on_a - trans0;
      Vec3A rel_pos2 = cp.pos_world_on_b - trans1;
      ContactPoint c = {cp.normal_world_on_b, cp.combined_friction, cp.combined_restitution, cp.distance_1};

      // rb0 / rb1: Some(body) only when the solver body has an original body
      RbSide s0 = {cx.bodies[id_a].has_orig ? &s->bodies[d0] : nullptr, d0 >= 0 ? &sd->bodies[m.body0_idx] : nullptr};
      RbSide s1 = {cx.bodies[id_b].has_orig ? &s->bodies[d1] : nullptr, d1 >= 0 ? &sd->bodies[m.body1_idx] : nullptr};

      if (cp.is_special) {
        if (cx.n_contact >= MAX_SOLVER_ROWS) { s->overflow[OVF_ACTIVE]++; }
        else cx.contact[cx.n_contact++] = get_split_only_contact_constraint(id_a, id_b, s0, s1, rel_pos1, rel_pos2, c, time_step);

        // special_dynamic_side (:63-79)
        int obj = -1;
        Vec3A lever = {0.0f, 0.0f, 0.0f};
        if (d0 >= 0) { obj = d0; lever = rel_pos1; }
        else if (d1 >= 0) { obj = d1; lever = rel_pos2; }
        if (obj >= 0) accumulate_special_sample(cx, obj, cp.normal_world_on_b, length(lever), cp.combined_friction, cp.combined_restitution);
        continue;
      }

      int friction_idx = cx.n_contact;
      Vec3A lat = calc_lat_friction_dir(c, cx.bodies[id_a], cx.bodies[id_b], rel_pos1, rel_pos2);
      if (cx.n_contact >= MAX_SOLVER_ROWS || cx.n_friction >= MAX_SOLVER_ROWS) { s->overflow[OVF_ACTIVE]++; continue; }
      cx.contact[cx.n_contact++] = get_contact_constraint(id_a, id_b, cx.bodies[id_a], cx.bodies[id_b], s0, s1, rel_pos1, rel_pos2, c, friction_idx, time_step);
      cx.friction[cx.n_friction++] = get_friction_constraint(id_a, id_b, cx.bodies[id_a], cx.bodies[id_b], s0, s1, rel_pos1, rel_pos2, cp.combined_friction, lat, friction_idx);
    }
  }

  // :301 active_manifold_idcs.clear()
  s->n_active = 0;
  for (int i = 0; i < MAX_ACTIVE; ++i) s->active[i] = 0;

  for (int i = 0; i < cx.n_special; ++i) {  // :303-305 emit_special_synthetics
    SpecialAccumulator acc = cx.special[i];
    push_special_synthetic(cx, s, sd, acc, time_step);
  }
}

// :467-506 solve_group_split_impulse_iterations
LF_HDI void solve_group_split_impulse_iterations(SolverCtx& cx) {
  bool should_run[MAX_SOLVER_ROWS];
  int remaining = cx.n_contact;
  for (int i = 0; i < cx.n_contact; ++i) should_run[i] = true;

  for (int it = 0; it < NUM_ITERATIONS; ++it) {
    if (remaining == 0) break;
    for (int i = 0; i < cx.n_contact; ++i) {
      if (!should_run[i]) continue;
      SolverConstraint& c = cx.contact[i];
      float residual = resolve_split_penetration_impulse(c, cx.bodies[c.solver_body_id_a], cx.bodies[c.solver_body_id_b]);
      if (lf_mul(residual, residual) == 0.0f) {
        should_run[i] = false;
        --remaining;
      }
    }
  }
}

// :537-585 solve_single_iteration
LF_HDI float solve_single_iteration(SolverCtx& cx) {
  float least_squares_residual = 0.0f;

  for (int i = 0; i < cx.n_contact; ++i) {
    SolverConstraint& c = cx.contact[i];
    if (c.is_split_only) continue;
    float residual = resolve_single_constraint_row(c, cx.bodies[c.solver_body_id_a], cx.bodies[c.solver_body_id_b], false);
    least_squares_residual = rs_max(lf_mul(residual, residual), least_squares_residual);
  }

  for (int i = 0; i < cx.n_friction; ++i) {
    SolverConstraint& c = cx.friction[i];
    float total_impulse = cx.contact[c.friction_idx].applied_impulse;
    if (total_impulse <= 0.0f) continue;

    float limit = lf_mul(c.friction, total_impulse);
    c.lower_limit = lf_neg(limit);
    c.upper_limit = limit;

    float residual = resolve_single_constraint_row(c, cx.bodies[c.solver_body_id_a], cx.bodies[c.solver_body_id_b], true);
    least_squares_residual = rs_max(lf_mul(residual, residual), least_squares_residual);
  }
  return least_squares_residual;
}

// :587-596 solve_group_iterations
LF_HDI void solve_group_iterations(SolverCtx& cx) {
  solve_group_split_impulse_iterations(cx);
  for (int it = 0; it < NUM_ITERATIONS; ++it) {
    float least_squares_residual = solve_single_iteration(cx);
    if (least_squares_residual == 0.0f) break;
  }
}

// linear_math/transform_util.rs:8-10 integrate_trans_no_rot and :12-37 integrate_trans.
// (P6 ports the same V3 function under its own name; this copy is private to the solver.)
LF_HDI void solver_integrate_trans(Affine3A* cur_trans, Quat* cur_rot, Vec3A lin_vel, Vec3A ang_vel, float time_step) {
  cur_trans->translation += lin_vel * time_step;  // integrate_trans_no_rot

  float angle = length(ang_vel);
  const float angular_motion_threshold = 0.78539816339744830962f;  // FRAC_PI_4
  if (lf_mul(angle, time_step) > angular_motion_threshold) angle = lf_div(angular_motion_threshold, time_step);

  float half_angle = lf_mul(lf_mul(angle, time_step), 0.5f);
  float scale;
  if (angle < 0.001f) {
    // 0.5 * dt - dt*dt*dt*0.020833334 * angle * angle, left to right
    float t = lf_mul(lf_mul(lf_mul(lf_mul(lf_mul(time_step, time_step), time_step), 0.020833334f), angle), angle);
    scale = lf_sub(lf_mul(0.5f, time_step), t);
  } else {
    scale = lf_div(lf::sin(half_angle), angle);
  }
  Vec3A axis = ang_vel * scale;
  Quat dorn = {axis.x, axis.y, axis.z, lf::cos(half_angle)};
  *cur_rot = normalize(dorn * *cur_rot);
  cur_trans->matrix3 = mat3a_from_quat(*cur_rot);
}

// :598-638 solve_group_finish (no write-back to the manifold points; the pools are local arrays)
LF_HDI void solve_group_finish(SolverCtx& cx, ArenaState* s, float time_step) {
  for (int i = 0; i < cx.n_bodies; ++i) {
    SolverBody& sv = cx.bodies[i];
    if (!sv.has_orig) continue;
    Body& body = s->bodies[sv.orig];

    sv.lin_vel += sv.delta_lin_vel;
    sv.ang_vel += sv.delta_ang_vel;

    if (length_squared(sv.push_vel) != 0.0f || length_squared(sv.turn_vel) != 0.0f) {
      Affine3A world_trans = body.world_trans;
      if (body.collision_flags & FLAG_NO_ANGULAR_MOTION) {
        world_trans.translation += sv.push_vel * time_step;
      } else {
        Quat world_rot = body.world_quat;
        solver_integrate_trans(&world_trans, &world_rot, sv.push_vel, sv.turn_vel * SPLIT_IMPULSE_TURN_ERP, time_step);
      }
      // rigid_body.rs:266-274 set_world_trans
      if ((body.collision_flags & FLAG_NO_ANGULAR_MOTION) == 0) body.world_quat = quat_from_mat3a(world_trans.matrix3);
      body.world_trans = world_trans;
    }

    Vec3A lin_vel = sv.lin_vel + sv.external_force_impulse;
    Vec3A ang_vel = sv.ang_vel + sv.external_torque_impulse;
    bool was_sleeping = body.activation == ACT_SLEEPING;
    body.lin_vel = lin_vel;
    body.ang_vel = ang_vel;
    // :603-611: a solve that leaves an asleep body moving wakes it into the timer (Vec3A != compares each lane exactly)
    if (was_sleeping && (lin_vel.x != 0.0f || lin_vel.y != 0.0f || lin_vel.z != 0.0f || ang_vel.x != 0.0f || ang_vel.y != 0.0f ||
                         ang_vel.z != 0.0f)) {
      rb_set_activation_state(body, ACT_WANTS_DEACTIVATION);
      body.deactivation_time = 0.0f;
    }
  }
}

// :147-186 solve_group, called from discrete_dynamics_world.rs:152 with time_step = TICK_TIME
LF_HD inline void phase_p5(ArenaState* s, const StaticData* sd) {
  const float time_step = lf_div(1.0f, 120.0f);  // sim/consts.rs:45 TICK_TIME = 1.0 / TICK_RATE
  SolverCtx cx;
  cx.n_bodies = cx.n_contact = cx.n_friction = cx.n_special = 0;
  cx.fixed_body_id = NO_BODY;
  solve_group_setup(cx, s, sd, time_step);
  solve_group_iterations(cx);
  solve_group_finish(cx, s, time_step);
}

}  // namespace lfsim
