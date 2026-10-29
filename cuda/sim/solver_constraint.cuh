// SolverBody and SolverConstraint rows of V3's sequential impulse solver, P5).
//
// V3 sources (rev ad8b38c, paths under rocketsim/src/bullet/):
//   dynamics/constraint_solver/solver_constraint.rs   rows: setup and the resolve_* functions
//   dynamics/constraint_solver/solver_body.rs         SolverBody
//   dynamics/constraint_solver/contact_solver_info.rs constants
//   collision/narrowphase/manifold_point.rs:37        calc_lat_friction_dir;  linear_math/util.rs:11 plane_space_1
//
// Warm start: V3 passes apply_warmstart = true to setup_contact_constraint, but the point's applied_impulse is always 0 (V3 never
// writes solver impulses back), so `applied_impulse = 0 * 0.85 = +0` and internal_apply_impulse adds
// (n * inv_mass) * +-0 and ang * +-0 to deltas that are +0: x + (+-0) is +0 for x = +0, so the deltas stay +0 and no bit changes
// (provided the products are finite, which holds for any finite state). It is therefore omitted.
// The one_dynamic variants (skip_separated_special_rows) are ball-only arenas: not Soccar 1v1, not ported.
#pragma once
#include "sim/arena_state.cuh"

namespace lfsim {

using namespace glam;

// contact_solver_info.rs
constexpr float SOR = 1.0f;
constexpr float ERP_2 = 0.8f;
constexpr float SPLIT_IMPULSE_PENETRATION_THRESHOLD = 1e30f;
constexpr float SPLIT_IMPULSE_TURN_ERP = 0.1f;
constexpr int NUM_ITERATIONS = 10;

constexpr int MAX_SOLVER_ROWS = 32;  // measured maximum 14 normal rows and 10 friction rows
constexpr int MAX_SOLVER_BODIES = 8;  // measured 4: three dynamic bodies and the fixed one
constexpr int NO_BODY = -1;

// solver_body.rs:7-20. original_body is `has_orig` plus `orig` (the dynamic index 0..2).
struct SolverBody {
  Vec3A delta_lin_vel, delta_ang_vel, inv_mass, push_vel, turn_vel, lin_vel, ang_vel, external_force_impulse, external_torque_impulse;
  int has_orig, orig;
};

// solver_body.rs:22-34 SolverBody::DEFAULT
LF_HDI SolverBody solver_body_default() {
  Vec3A z = {0.0f, 0.0f, 0.0f};
  return SolverBody{z, z, z, z, z, z, z, z, z, 0, 0};
}

// solver_body.rs:36-49 SolverBody::new(rb)
LF_HDI SolverBody solver_body_new(const Body& rb, const BodyConst& c, int d) {
  Vec3A z = {0.0f, 0.0f, 0.0f};
  return SolverBody{z, z, c.inv_mass_splat, z, z, rb.lin_vel, rb.ang_vel, rb.accum_lin_vel, rb.accum_ang_vel, 1, d};
}

// solver_body.rs:61-64
LF_HDI Vec3A solver_vel_in_local_point_no_delta(const SolverBody& b, Vec3A rel_pos) {
  return b.lin_vel + b.external_force_impulse + cross(b.ang_vel + b.external_torque_impulse, rel_pos);
}

// solver_constraint.rs:8-11 bullet_dot: (x*x' , y*y', z*z') summed as x + (y + z), not glam's dot order.
LF_HDI float bullet_dot(Vec3A a, Vec3A b) {
  Vec3A r = a * b;
  return lf_add(r.x, lf_add(r.y, r.z));
}

// solver_constraint.rs:13-27
struct SolverConstraint {
  Vec3A rel_pos1_cross_normal, contact_normal_1, rel_pos2_cross_normal, contact_normal_2, angular_component_a, angular_component_b;
  float applied_push_impulse, applied_impulse, friction, jac_diag_ab_inv, rhs, lower_limit, upper_limit, rhs_penetration;
  int friction_idx, solver_body_id_a, solver_body_id_b, is_split_only;
};

// What a row setup reads of a manifold point (ManifoldPoint fields: manifold_point.rs:8-18) and of a synthetic one.
struct ContactPoint {
  Vec3A normal_world_on_b;
  float combined_friction, combined_restitution, distance_1;
};

// The dynamic body behind a row side (rb0 / rb1 of V3: Some only for a solver body with an original body).
struct RbSide {
  const Body* rb;       // null for a static / fixed side
  const BodyConst* c;
};

// linear_math/util.rs:11-21 plane_space_1.  FRAC_1_SQRT_2 = 0.70710677f.
LF_HDI Vec3A plane_space_1(Vec3A n) {
  if (lf_abs(n.z) > 0.70710678118654752440f) {
    float a = lf_add(lf_mul(n.y, n.y), lf_mul(n.z, n.z));
    float k = lf_div(1.0f, lf_sqrt(a));
    return {0.0f, lf_mul(lf_neg(n.z), k), lf_mul(n.y, k)};
  }
  float a = lf_add(lf_mul(n.x, n.x), lf_mul(n.y, n.y));
  float k = lf_div(1.0f, lf_sqrt(a));
  return {lf_mul(lf_neg(n.y), k), lf_mul(n.x, k), 0.0f};
}

// manifold_point.rs:37-58 calc_lat_friction_dir
LF_HDI Vec3A calc_lat_friction_dir(const ContactPoint& cp, const SolverBody& a, const SolverBody& b, Vec3A rel_pos1, Vec3A rel_pos2) {
  Vec3A vel1 = solver_vel_in_local_point_no_delta(a, rel_pos1);
  Vec3A vel2 = solver_vel_in_local_point_no_delta(b, rel_pos2);
  Vec3A vel = vel1 - vel2;
  float rel_vel = dot(cp.normal_world_on_b, vel);
  Vec3A lat = vel - cp.normal_world_on_b * rel_vel;
  float lat_rel_vel = length_squared(lat);
  if (lat_rel_vel > 1.1920929e-7f) return lat * lf_div(1.0f, lf_sqrt(lat_rel_vel));  // f32::EPSILON
  return plane_space_1(cp.normal_world_on_b);
}

// solver_constraint.rs:35-37 / :69-98 get_contact_constraint's struct literal, with Default::default() zeros elsewhere
LF_HDI SolverConstraint row_init(int id_a, int id_b, int friction_idx, float friction, float lower, float upper) {
  SolverConstraint r;
  Vec3A z = {0.0f, 0.0f, 0.0f};
  r.rel_pos1_cross_normal = r.contact_normal_1 = r.rel_pos2_cross_normal = r.contact_normal_2 = r.angular_component_a = r.angular_component_b = z;
  r.applied_push_impulse = r.applied_impulse = r.jac_diag_ab_inv = r.rhs = r.rhs_penetration = 0.0f;
  r.friction = friction;
  r.lower_limit = lower;
  r.upper_limit = upper;
  r.friction_idx = friction_idx;
  r.solver_body_id_a = id_a;
  r.solver_body_id_b = id_b;
  r.is_split_only = 0;
  return r;
}

// solver_constraint.rs:69-98 get_contact_constraint + :190-289 setup_contact_constraint (warm start omitted: see the header)
LF_HDI SolverConstraint get_contact_constraint(int id_a, int id_b, const SolverBody& sa, const SolverBody& sb, RbSide s0, RbSide s1,
                                               Vec3A rel_pos1, Vec3A rel_pos2, const ContactPoint& cp, int friction_idx, float time_step) {
  SolverConstraint r = row_init(id_a, id_b, friction_idx, cp.combined_friction, 0.0f, 1e10f);
  float inv_time_step = lf_div(1.0f, time_step);
  float erp = ERP_2;
  Vec3A n = cp.normal_world_on_b;
  r.applied_impulse = 0.0f;  // cp.applied_impulse (0) * WARMSTARTING_FACTOR

  float denom0 = 0.0f, vel_1_dot_n = 0.0f;
  Vec3A vel0 = {0.0f, 0.0f, 0.0f};
  if (s0.rb) {
    Vec3A torque_axis = cross(rel_pos1, n);
    r.angular_component_a = mul_transpose_vec3a(s0.rb->inv_inertia_tensor_world, torque_axis);
    Vec3A vec = cross(r.angular_component_a, rel_pos1);
    denom0 = lf_add(s0.c->inv_mass, dot(n, vec));
    vel0 = s0.rb->lin_vel + cross(s0.rb->ang_vel, rel_pos1);  // RigidBody::get_vel_in_local_point
    r.contact_normal_1 = n;
    r.rel_pos1_cross_normal = torque_axis;
    vel_1_dot_n = lf_add(dot(r.contact_normal_1, sa.lin_vel + sa.external_force_impulse),
                         dot(r.rel_pos1_cross_normal, sa.ang_vel + sa.external_torque_impulse));
  }
  float denom1 = 0.0f, vel_2_dot_n = 0.0f;
  Vec3A vel1 = {0.0f, 0.0f, 0.0f};
  if (s1.rb) {
    Vec3A torque_axis = cross(rel_pos2, n);
    r.angular_component_b = mul_transpose_vec3a(s1.rb->inv_inertia_tensor_world, -torque_axis);
    Vec3A vec = cross(-r.angular_component_b, rel_pos2);
    denom1 = lf_add(s1.c->inv_mass, dot(n, vec));
    vel1 = s1.rb->lin_vel + cross(s1.rb->ang_vel, rel_pos2);
    r.contact_normal_2 = -n;
    r.rel_pos2_cross_normal = -torque_axis;
    vel_2_dot_n = lf_add(dot(r.contact_normal_2, sb.lin_vel + sb.external_force_impulse),
                         dot(r.rel_pos2_cross_normal, sb.ang_vel + sb.external_torque_impulse));
  }

  r.jac_diag_ab_inv = lf_div(SOR, lf_add(denom0, denom1));

  Vec3A vel = vel0 - vel1;
  float rel_vel = dot(n, vel);
  float restitution = rs_max(lf_mul(cp.combined_restitution, lf_neg(rel_vel)), 0.0f);  // restitution_curve :186

  float penetration = cp.distance_1;
  float positional_error = penetration > 0.0f ? 0.0f : lf_mul(lf_mul(lf_neg(penetration), erp), inv_time_step);

  rel_vel = lf_add(vel_1_dot_n, vel_2_dot_n);
  float vel_error = lf_sub(restitution, rel_vel);

  float penetration_impulse = lf_mul(positional_error, r.jac_diag_ab_inv);
  float vel_impulse = lf_mul(vel_error, r.jac_diag_ab_inv);

  if (penetration > SPLIT_IMPULSE_PENETRATION_THRESHOLD) {
    r.rhs = lf_add(penetration_impulse, vel_impulse);
    r.rhs_penetration = 0.0f;
  } else {
    r.rhs = vel_impulse;
    r.rhs_penetration = penetration_impulse;
  }
  return r;
}

// solver_constraint.rs:100-172 get_split_only_contact_constraint + setup_split_contact_constraint
LF_HDI SolverConstraint get_split_only_contact_constraint(int id_a, int id_b, RbSide s0, RbSide s1, Vec3A rel_pos1, Vec3A rel_pos2,
                                                          const ContactPoint& cp, float time_step) {
  SolverConstraint r = row_init(id_a, id_b, -1, cp.combined_friction, 0.0f, 1e10f);  // NO_FRICTION_PARENT
  r.is_split_only = 1;
  Vec3A n = cp.normal_world_on_b;

  float denom0 = 0.0f;
  if (s0.rb) {
    Vec3A torque_axis = cross(rel_pos1, n);
    r.angular_component_a = mul_transpose_vec3a(s0.rb->inv_inertia_tensor_world, torque_axis);
    Vec3A vec = cross(r.angular_component_a, rel_pos1);
    denom0 = lf_add(s0.c->inv_mass, dot(n, vec));
    r.contact_normal_1 = n;
    r.rel_pos1_cross_normal = torque_axis;
  }
  float denom1 = 0.0f;
  if (s1.rb) {
    Vec3A torque_axis = cross(rel_pos2, n);
    r.angular_component_b = mul_transpose_vec3a(s1.rb->inv_inertia_tensor_world, -torque_axis);
    Vec3A vec = cross(-r.angular_component_b, rel_pos2);
    denom1 = lf_add(s1.c->inv_mass, dot(n, vec));
    r.contact_normal_2 = -n;
    r.rel_pos2_cross_normal = -torque_axis;
  }

  r.jac_diag_ab_inv = lf_div(SOR, lf_add(denom0, denom1));

  float inv_time_step = lf_div(1.0f, time_step);
  float penetration = cp.distance_1;
  float positional_error = penetration > 0.0f ? 0.0f : lf_mul(lf_mul(lf_neg(penetration), ERP_2), inv_time_step);
  float penetration_impulse = lf_mul(positional_error, r.jac_diag_ab_inv);
  r.rhs_penetration = penetration > SPLIT_IMPULSE_PENETRATION_THRESHOLD ? 0.0f : penetration_impulse;
  return r;
}

// solver_constraint.rs:39-67 get_friction_constraint + :291-340 setup_friction_constraint
LF_HDI SolverConstraint get_friction_constraint(int id_a, int id_b, const SolverBody& sa, const SolverBody& sb, RbSide s0, RbSide s1,
                                                Vec3A rel_pos1, Vec3A rel_pos2, float friction, Vec3A normal_axis, int friction_idx) {
  SolverConstraint r = row_init(id_a, id_b, friction_idx, friction, lf_neg(friction), friction);
  float vel_1_dot_n = 0.0f, denom1 = 0.0f;
  if (s0.rb) {
    r.contact_normal_1 = normal_axis;
    r.rel_pos1_cross_normal = cross(rel_pos1, r.contact_normal_1);
    r.angular_component_a = mul_transpose_vec3a(s0.rb->inv_inertia_tensor_world, r.rel_pos1_cross_normal);
    Vec3A vec = cross(r.angular_component_a, rel_pos1);
    denom1 = lf_add(s0.c->inv_mass, dot(normal_axis, vec));
    vel_1_dot_n = lf_add(dot(r.contact_normal_1, sa.lin_vel + sa.external_force_impulse), dot(r.rel_pos1_cross_normal, sa.ang_vel));
  }
  float vel_2_dot_n = 0.0f, denom2 = 0.0f;
  if (s1.rb) {
    r.contact_normal_2 = -normal_axis;
    r.rel_pos2_cross_normal = cross(rel_pos2, r.contact_normal_2);
    r.angular_component_b = mul_transpose_vec3a(s1.rb->inv_inertia_tensor_world, r.rel_pos2_cross_normal);
    Vec3A vec = cross(-r.angular_component_b, rel_pos2);
    denom2 = lf_add(s1.c->inv_mass, dot(normal_axis, vec));
    vel_2_dot_n = lf_add(dot(r.contact_normal_2, sb.lin_vel + sb.external_force_impulse), dot(r.rel_pos2_cross_normal, sb.ang_vel));
  }
  r.jac_diag_ab_inv = lf_div(SOR, lf_add(denom1, denom2));
  float rel_vel = lf_add(vel_1_dot_n, vel_2_dot_n);
  float vel_error = lf_neg(rel_vel);
  r.rhs = lf_mul(vel_error, r.jac_diag_ab_inv);
  return r;
}

// solver_constraint.rs:518-565 resolve_split_penetration_impulse
LF_HDI float resolve_split_penetration_impulse(SolverConstraint& c, SolverBody& body_a, SolverBody& body_b) {
  if (c.rhs_penetration == 0.0f) return 0.0f;

  float delta_impulse = c.rhs_penetration;

  float delta_vel_1_dot_n = body_a.has_orig ? lf_add(bullet_dot(c.contact_normal_1, body_a.push_vel), bullet_dot(c.rel_pos1_cross_normal, body_a.turn_vel)) : 0.0f;
  float delta_vel_2_dot_n = body_b.has_orig ? lf_add(bullet_dot(c.contact_normal_2, body_b.push_vel), bullet_dot(c.rel_pos2_cross_normal, body_b.turn_vel)) : 0.0f;

  delta_impulse = lf_sub(delta_impulse, lf_mul(delta_vel_1_dot_n, c.jac_diag_ab_inv));
  delta_impulse = lf_sub(delta_impulse, lf_mul(delta_vel_2_dot_n, c.jac_diag_ab_inv));

  float sum = lf_add(c.applied_push_impulse, delta_impulse);
  if (sum < c.lower_limit) {
    delta_impulse = lf_sub(c.lower_limit, c.applied_push_impulse);
    c.applied_push_impulse = c.lower_limit;
  } else {
    c.applied_push_impulse = sum;
  }

  if (body_a.has_orig) {
    body_a.push_vel = body_a.push_vel + c.contact_normal_1 * body_a.inv_mass * delta_impulse;
    body_a.turn_vel = body_a.turn_vel + c.angular_component_a * delta_impulse;
  }
  if (body_b.has_orig) {
    body_b.push_vel = body_b.push_vel + c.contact_normal_2 * body_b.inv_mass * delta_impulse;
    body_b.turn_vel = body_b.turn_vel + c.angular_component_b * delta_impulse;
  }
  return lf_div(delta_impulse, c.jac_diag_ab_inv);
}

// solver_constraint.rs:427-472 (generic: both limits, friction rows) and :474-516 (lower limit only, normal rows).
// `two_sided` selects the generic one.
LF_HDI float resolve_single_constraint_row(SolverConstraint& c, SolverBody& body_a, SolverBody& body_b, bool two_sided) {
  float delta_impulse = c.rhs;

  float delta_vel_1_dot_n = body_a.has_orig ? lf_add(bullet_dot(c.contact_normal_1, body_a.delta_lin_vel), bullet_dot(c.rel_pos1_cross_normal, body_a.delta_ang_vel)) : 0.0f;
  float delta_vel_2_dot_n = body_b.has_orig ? lf_add(bullet_dot(c.contact_normal_2, body_b.delta_lin_vel), bullet_dot(c.rel_pos2_cross_normal, body_b.delta_ang_vel)) : 0.0f;

  delta_impulse = lf_sub(delta_impulse, lf_mul(delta_vel_1_dot_n, c.jac_diag_ab_inv));
  delta_impulse = lf_sub(delta_impulse, lf_mul(delta_vel_2_dot_n, c.jac_diag_ab_inv));

  float sum = lf_add(c.applied_impulse, delta_impulse);
  if (sum < c.lower_limit) {
    delta_impulse = lf_sub(c.lower_limit, c.applied_impulse);
    c.applied_impulse = c.lower_limit;
  } else if (two_sided && sum > c.upper_limit) {
    delta_impulse = lf_sub(c.upper_limit, c.applied_impulse);
    c.applied_impulse = c.upper_limit;
  } else {
    c.applied_impulse = sum;
  }

  if (body_a.has_orig) {
    body_a.delta_lin_vel = body_a.delta_lin_vel + c.contact_normal_1 * body_a.inv_mass * delta_impulse;
    body_a.delta_ang_vel = body_a.delta_ang_vel + c.angular_component_a * delta_impulse;
  }
  if (body_b.has_orig) {
    body_b.delta_lin_vel = body_b.delta_lin_vel + c.contact_normal_2 * body_b.inv_mass * delta_impulse;
    body_b.delta_ang_vel = body_b.delta_ang_vel + c.angular_component_b * delta_impulse;
  }
  return lf_div(delta_impulse, c.jac_diag_ab_inv);
}

}  // namespace lfsim
