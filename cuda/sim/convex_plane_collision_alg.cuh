// V3: bullet/collision/dispatch/convex_plane_collision_alg.rs and the RigidBodyWrapper it takes (collision_obj_wrapper.rs):
// the ball (a sphere at the body's own transform) or a car's box child (at the child's world transform) against one of the
// four planes.
#pragma once
#include "sim/narrowphase.cuh"
#include "sim/persistent_manifold.cuh"

namespace lfsim {

// V3: collision_obj_wrapper.rs:24 get_aabb
LF_HDI Aabb convex_wrapper_aabb(const NpCtx& c, uint32_t obj, const Affine3A& world_trans, bool box_child) {
  const BodyConst& bc = body_const(c, obj);
  if (box_child) return aabb_from_half_extents_transform(bc.box_half_extents, bc.box_margin, world_trans);  // box_shape.rs:30
  return sphere_aabb(bc.sphere_margin, world_trans.translation);                                           // collision_shape.rs:40
}

// V3: collision_obj_wrapper.rs:32 local_get_supporting_vertex (box_shape.rs:60 or sphere_shape.rs:62)
LF_HDI Vec3A convex_wrapper_supporting_vertex(const NpCtx& c, uint32_t obj, bool box_child, Vec3A vec) {
  const BodyConst& bc = body_const(c, obj);
  if (box_child) {
    BVec3A neg = cmplt(vec, splat3(0.0f));
    Vec3A support_sign = select(neg, splat3(-1.0f), splat3(1.0f));
    return add_s(bc.box_half_extents, bc.box_margin) * support_sign;
  }
  Vec3A n;
  try_normalize(vec, &n);  // V3 unwraps (a zero vector cannot occur: the plane normal is a unit axis)
  return splat3(bc.sphere_margin) * n;
}

// V3: convex_plane_collision_alg.rs:12 process_collision(convex_obj wrapper, plane_obj, plane_shape, cb, out). `obj` is the
// body (the ball, or the car whose box child is the convex), `world_trans` the wrapper's transform, `box_child` its
// child_shape_override. Returns whether V3's `out` is Some, with the manifold in *out.
LF_HDI bool convex_plane_process_collision(NpCtx& c, uint32_t obj, uint32_t plane_obj, const Affine3A& world_trans, bool box_child,
                                           Manifold* out) {
  const Plane& plane_shape = c.sd->planes[plane_obj - NUM_MESHES];
  Aabb convex_aabb = convex_wrapper_aabb(c, obj, world_trans, box_child);
  if (!aabb_intersects(convex_aabb, plane_shape.aabb_cache)) return false;

  Vec3A plane_normal = plane_shape.plane_normal;

  Affine3A plane_trans = body_trans(c, plane_obj);

  Mat3A plane_in_convex = transpose(world_trans.matrix3);
  Affine3A convex_in_plane_trans = {world_trans.matrix3, world_trans.translation - plane_trans.translation};

  Vec3A vtx = convex_wrapper_supporting_vertex(c, obj, box_child, plane_in_convex * (-plane_normal));
  Vec3A vtx_in_plane = transform_point3a(convex_in_plane_trans, vtx);
  float distance = dot(plane_normal, vtx_in_plane);

  Manifold manifold = manifold_new(obj, plane_obj);
  if (distance >= manifold_breaking_threshold(c, manifold)) return false;

  Vec3A vtx_in_plane_projected = vtx_in_plane - splat3(distance) * plane_normal;
  Vec3A vtx_in_plane_world = vtx_in_plane_projected + plane_trans.translation;
  Vec3A normal_on_surface_b = plane_normal;

  manifold_add_contact_point(c, &manifold, obj, plane_obj, normal_on_surface_b, vtx_in_plane_world, distance, -1);

  manifold_refresh_contact_points(c, &manifold, obj, plane_obj);
  *out = manifold;
  return true;
}

}  // namespace lfsim
