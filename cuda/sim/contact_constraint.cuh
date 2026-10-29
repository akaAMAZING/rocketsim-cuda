// V3 bullet/dynamics/constraint_solver/contact_constraint.rs: the two single-body resolutions the wheels use.
#pragma once
#include "sim/consts.cuh"
#include "sim/linear_piece_curve.cuh"
#include "sim/rigid_body_p1.cuh"

namespace lfsim {
using namespace glam;

// V3: contact_constraint.rs:6 resolve_single_collision
LF_HDI float resolve_single_collision(const RbView& body1, const RbView& body2, Vec3A contact_pos_world, Vec3A contact_normal_on_b,
                                      float time_step, float distance) {
  Vec3A rel_pos1 = contact_pos_world - body1.world_trans.translation;
  Vec3A rel_pos2 = contact_pos_world - body2.world_trans.translation;

  Vec3A vel1 = rb_get_vel_in_local_point_p1(body1, rel_pos1);
  Vec3A vel2 = rb_get_vel_in_local_point_p1(body2, rel_pos2);
  Vec3A vel = vel1 - vel2;
  float rel_vel = dot(contact_normal_on_b, vel);

  float positional_error = lf_div(lf_mul(c::WHEEL_PUSHBACK_ERP, lf_neg(distance)), time_step);
  float vel_error = lf_neg(rel_vel);
  float denom0 = rb_compute_impulse_denominator_p1(body1, contact_pos_world, contact_normal_on_b);
  float denom1 = rb_compute_impulse_denominator_p1(body2, contact_pos_world, contact_normal_on_b);
  float jac_diag_ab_inv = lf_div(1.0f, lf_add(denom0, denom1));

  float penetration_impulse = lf_mul(positional_error, jac_diag_ab_inv);
  float vel_impulse = lf_mul(vel_error, jac_diag_ab_inv);

  float normal_impulse = lf_add(penetration_impulse, vel_impulse);
  return rs_max(normal_impulse, 0.0f);
}

// V3: contact_constraint.rs:35 resolve_single_bilateral_fake_ground. `inv_inertia_local` is the chassis' BodyConst field.
LF_HDI float resolve_single_bilateral_fake_ground(const RbView& body1, Vec3A inv_inertia_local, Vec3A pos, Vec3A normal) {
  const float CONTACT_DAMPING = -0.2f;
  Vec3A rel_pos1 = pos - body1.world_trans.translation;
  Vec3A vel1 = rb_get_vel_in_local_point_p1(body1, rel_pos1);

  Vec3A a_j = mul_transpose_vec3a(body1.world_trans.matrix3, cross(rel_pos1, normal));
  float jac_diag_ab = lf_add(body1.inv_mass, dot(inv_inertia_local * a_j, a_j));
  float jac_diag_ab_inv = lf_div(1.0f, jac_diag_ab);
  float rel_vel = dot(normal, vel1);

  return lf_mul(lf_mul(CONTACT_DAMPING, rel_vel), jac_diag_ab_inv);
}

}  // namespace lfsim
