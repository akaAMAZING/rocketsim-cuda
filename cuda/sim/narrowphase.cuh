// Narrowphase context and body access shared by the P3b files.
// V3 reads bodies through `&RigidBody`; here a body is its index in collision_objs (0-15 meshes, 16-19 planes, then the ball
// and the two cars at StaticData.dyn_body), and its fields come from StaticData (constants) or ArenaState.bodies (state).
#pragma once
#include "sim/arena_state.cuh"
#include "sim/bullet_consts.cuh"
#include "common/libm.cuh"

namespace lfsim {
using namespace glam;

// V3: sim/user_info_types.rs:8-18 (derive Ord: declaration order).
enum UserInfo : uint32_t { USER_NONE = 0, USER_CAR = 1, USER_BALL = 2 };
// shape kinds of BodyConst.shape_kind.
enum ShapeKind : uint32_t { SHAPE_MESH = 0, SHAPE_PLANE = 1, SHAPE_SPHERE = 2, SHAPE_COMPOUND = 3 };

constexpr float LF_EPSILON = 1.1920928955078125e-07f;  // f32::EPSILON
constexpr float LF_F32_MAX = 3.4028234663852886e+38f;  // f32::MAX
constexpr float LF_TAU = 6.28318530717958647692f;      // f32::consts::TAU

// The state P3b works on: the arena, the static data, and a counter of pair algorithms that are not ported yet (a state
// that bumps it is blocked: the phase returns PHASE_UNIMPLEMENTED for it).
struct NpCtx {
  ArenaState* s;
  const StaticData* sd;
  uint32_t unimplemented;
};

// The contact a mesh job (the ball or a car against a triangle mesh: sphere_concave_collision_alg.cuh, compound_collision_alg.cuh)
// reports for one leaf, and where it goes: manifold_add_contact_point(c, target, body0, body1, normal, point, depth, idx).
struct MeshContact {
  Manifold* target;
  uint32_t body0, body1;
  Vec3A normal, point;
  float depth;
  int32_t idx;
};

// Dynamic slot (index into ArenaState.bodies) of a body index, or -1 for a static body.
LF_HDI int dyn_slot(const NpCtx& c, uint32_t b) {
  for (int d = 0; d < NUM_DYN; ++d)
    if (c.sd->dyn_body[d] == b) return d;
  return -1;
}
LF_HDI const BodyConst& body_const(const NpCtx& c, uint32_t b) { return c.sd->bodies[b]; }
// rigid_body.rs:130 world_trans (a static body's is its pose).
LF_HDI Affine3A body_trans(const NpCtx& c, uint32_t b) {
  int d = dyn_slot(c, b);
  return d >= 0 ? c.s->bodies[d].world_trans : c.sd->bodies[b].pose;
}
// rigid_body.rs:144 collision_flags (a static body's is CF_STATIC; the ball and the cars carry CF_CUSTOM_MATERIAL, set at
// construction: sim/ball/base.rs:96, sim/car/base.rs:145).
LF_HDI uint32_t body_flags(const NpCtx& c, uint32_t b) {
  int d = dyn_slot(c, b);
  return d >= 0 ? c.s->bodies[d].collision_flags : CF_STATIC;
}
LF_HDI bool body_is_static(const NpCtx& c, uint32_t b) { return (body_flags(c, b) & CF_STATIC) != 0; }               // :288
LF_HDI bool body_has_contact_response(const NpCtx& c, uint32_t b) { return (body_flags(c, b) & CF_NO_CONTACT_RESPONSE) == 0; }  // :299
// A static body is added asleep (discrete_dynamics_world.rs:166-167 add_rigid_body_default), so it reads inactive.
LF_HDI bool body_is_active(const NpCtx& c, uint32_t b) {                                                            // :300
  int d = dyn_slot(c, b);
  return d >= 0 && c.s->bodies[d].activation != 1 && c.s->bodies[d].activation != 2;
}

// ---- small helpers over the glam layer -----------------------------------------------------------------------------------
LF_HDI Vec3A add_s(Vec3A a, float s) { return a + splat3(s); }  // Vec3A + f32
LF_HDI bool aabb_intersects(const Aabb& a, const Aabb& b) {     // shared/aabb.rs:43
  return glam::all(cmple(a.min, b.max)) && glam::all(cmpge(a.max, b.min));
}
LF_HDI Mat3A mat3a_abs(const Mat3A& m) { return {glam::abs(m.x_axis), glam::abs(m.y_axis), glam::abs(m.z_axis)}; }
// shared/aabb.rs:60 from_half_extents_transform
LF_HDI Aabb aabb_from_half_extents_transform(Vec3A half_extents, float margin, const Affine3A& t) {
  Vec3A half_with_margin = add_s(half_extents, margin);
  Mat3A abs_b = mat3a_abs(t.matrix3);
  Vec3A center = t.translation;
  Vec3A extent = abs_b * half_with_margin;
  return {center - extent, center + extent};
}
// sphere_shape.rs:36-44 get_aabb(center); SPHERE_RADIUS_MARGIN is sphere_shape.rs:9
LF_HDI Aabb sphere_aabb(float radius, Vec3A center) {
  float margin = lf_add(radius, 0.08f);
  Vec3A extent = splat3(margin);
  return {center - extent, center + extent};
}
// collision_shape.rs:38 get_aabb(t) of body b (a plane's is its cache; a mesh's the identity box shifted by t)
LF_HDI Aabb body_raw_aabb(const NpCtx& c, uint32_t b) {
  const BodyConst& bc = body_const(c, b);
  Affine3A t = body_trans(c, b);
  switch (bc.shape_kind) {
    case SHAPE_SPHERE: return sphere_aabb(bc.sphere_margin, t.translation);
    case SHAPE_PLANE: return c.sd->planes[b - NUM_MESHES].aabb_cache;
    case SHAPE_MESH: {
      const Aabb& id = c.sd->meshes[b].aabb_ident_cache;
      return {id.min + t.translation, id.max + t.translation};
    }
    default: {  // compound: compound_shape.rs:30-42
      const Aabb& la = bc.compound_local_aabb;
      Vec3A local_half_extents = 0.5f * (la.max - la.min);
      Vec3A local_center = 0.5f * (la.max + la.min);
      Mat3A abs_b = mat3a_abs(t.matrix3);
      Vec3A center = transform_point3a(t, local_center);
      Vec3A extent = abs_b * local_half_extents;
      return {center - extent, center + extent};
    }
  }
}
// compound_collision_alg.rs:100 child_world_aabb: the car's box in the world.
LF_HDI Affine3A compound_child_world_trans(const NpCtx& c, uint32_t b) { return body_trans(c, b) * c.sd->bodies[b].child_trans; }
LF_HDI Aabb compound_child_world_aabb(const NpCtx& c, uint32_t b) {
  const BodyConst& bc = body_const(c, b);
  return aabb_from_half_extents_transform(bc.box_half_extents, bc.box_margin, compound_child_world_trans(c, b));
}

}  // namespace lfsim
