// V3 bullet/dynamics/rigid_body.rs: the rigid-body operations P1 uses, on the three dynamic Bodies of ArenaState and
// on the static bodies (a read-only view built from StaticData). The P2/P6 operations of the same file are in
// rigid_body.cuh; the names here carry the suffix p1 so the two files can sit side by side.
#pragma once
#include "sim/arena_state.cuh"
#include "sim/rigid_body.cuh"
#include "sim/consts.cuh"
#include "sim/linear_piece_curve.cuh"
#include "common/libm.cuh"

namespace lfsim {
using namespace glam;

constexpr uint32_t RAY_NO_BODY = 0xFFFFFFFFu;

// The dynamic slot (0 ball, 1 car 0, 2 car 1) of body idx `idx`, or -1 for a static body. V3: discrete_dynamics_world.rs:27.
LF_HDI int dyn_slot_p1(const StaticData* sd, uint32_t idx) {
  for (int d = 0; d < NUM_DYN; ++d)
    if (sd->dyn_body[d] == idx) return d;
  return -1;
}

// A read-only view of one body at the moment of a call: what the vehicle code reads of a chassis or of the body a wheel hit.
struct RbView {
  Affine3A world_trans;
  Vec3A lin_vel, ang_vel;
  Mat3A inv_inertia_tensor_world;
  float inv_mass;
  uint32_t collision_flags;
  uint32_t user_idx;
};

// A static body is never moved: its pose is in StaticData, its velocities and inverse inertia are zero (rigid_body.rs:175-236:
// mass 0 gives inv_mass 0 and inv_inertia_local 0, so get_inertia_tensor is a zero matrix; its sign pattern cannot reach a result,
// compute_impulse_denominator adds the term to +0.0).
LF_HDI RbView rb_view_p1(const ArenaState* s, const StaticData* sd, uint32_t idx) {
  RbView v;
  int d = dyn_slot_p1(sd, idx);
  const BodyConst& c = sd->bodies[idx];
  v.user_idx = c.user_idx;
  v.inv_mass = c.inv_mass;
  if (d >= 0) {
    const Body& b = s->bodies[d];
    v.world_trans = b.world_trans;
    v.lin_vel = b.lin_vel;
    v.ang_vel = b.ang_vel;
    v.inv_inertia_tensor_world = b.inv_inertia_tensor_world;
    v.collision_flags = b.collision_flags;
  } else {
    v.world_trans = c.pose;
    v.lin_vel = splat3(0.0f);
    v.ang_vel = splat3(0.0f);
    v.inv_inertia_tensor_world = {splat3(0.0f), splat3(0.0f), splat3(0.0f)};
    v.collision_flags = CF_STATIC;
  }
  return v;
}

LF_HDI bool rb_is_static_p1(const RbView& v) { return (v.collision_flags & CF_STATIC) != 0; }            // V3: rigid_body.rs:288
LF_HDI bool rb_has_contact_response_p1(const RbView& v) { return (v.collision_flags & CF_NO_CONTACT_RESPONSE) == 0; }  // :299

// V3: rigid_body.rs:417 get_vel_in_local_point
LF_HDI Vec3A rb_get_vel_in_local_point_p1(const RbView& v, Vec3A rel_pos) { return v.lin_vel + cross(v.ang_vel, rel_pos); }

// V3: rigid_body.rs:452 compute_impulse_denominator
LF_HDI float rb_compute_impulse_denominator_p1(const RbView& v, Vec3A pos, Vec3A normal) {
  Vec3A r0 = pos - v.world_trans.translation;
  Vec3A c0 = cross(r0, normal);
  Vec3A vec = cross(mul_transpose_vec3a(v.inv_inertia_tensor_world, c0), r0);
  return lf_add(v.inv_mass, dot(normal, vec));
}

// V3: rigid_body.rs:444 get_mass
LF_HDI float rb_get_mass_p1(const BodyConst& c) { return c.inv_mass == 0.0f ? 0.0f : lf_div(1.0f, c.inv_mass); }

enum ImpulseKind { IMPULSE_LINEAR, IMPULSE_LINEAR_REL_POS, IMPULSE_ANGULAR };

// V3: rigid_body.rs:353 add_impulse (the compiler folds the constant kind/massed/accum arguments in V3; here they are values).
LF_HDI void rb_add_impulse_p1(Body& b, const BodyConst& c, ImpulseKind kind, Vec3A v, Vec3A rel_pos, bool massed, bool accum) {
  Vec3A lin_impulse = splat3(0.0f);
  Vec3A ang_impulse = splat3(0.0f);
  float massed_scaler = massed ? c.inv_mass : 1.0f;
  switch (kind) {
    case IMPULSE_LINEAR:
      lin_impulse = v * massed_scaler;
      break;
    case IMPULSE_LINEAR_REL_POS:
      lin_impulse = v * massed_scaler;
      ang_impulse = b.inv_inertia_tensor_world * cross(rel_pos, v);
      if (!massed) ang_impulse *= c.mass;
      break;
    case IMPULSE_ANGULAR:
      ang_impulse = v * massed_scaler;
      break;
  }
  if (accum) {
    b.accum_lin_vel += lin_impulse;
    b.accum_ang_vel += ang_impulse;
  } else {
    b.lin_vel += lin_impulse;
    b.ang_vel += ang_impulse;
  }
}

}  // namespace lfsim
