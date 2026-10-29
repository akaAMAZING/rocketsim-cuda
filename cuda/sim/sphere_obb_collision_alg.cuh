// V3: bullet/collision/dispatch/sphere_obb_collision_alg.rs.
#pragma once
#include "sim/narrowphase.cuh"
#include "sim/persistent_manifold.cuh"

namespace lfsim {

LF_HDI void sphere_obb_set_at(Vec3A* v, int i, float f) {
  if (i == 0) v->x = f;
  else if (i == 1) v->y = f;
  else v->z = f;
}

// V3: sphere_obb_collision_alg.rs:12 get_sphere_penetration -> (min_dist, closest, normal); the `test_face` closure inlined
// in V3's call order.
LF_HDI void sphere_obb_get_sphere_penetration(Vec3A box_extents, Vec3A sphere_from_local, float* out_dist, Vec3A* out_closest,
                                              Vec3A* out_normal) {
  float min_dist = lf_sub(box_extents.x, sphere_from_local.x);
  Vec3A closest = sphere_from_local;
  closest.x = box_extents.x;
  Vec3A normal = {1.0f, 0.0f, 0.0f};

  const int axes[5] = {0, 1, 1, 2, 2};
  const float signs[5] = {-1.0f, 1.0f, -1.0f, 1.0f, -1.0f};
  for (int t = 0; t < 5; ++t) {
    int axis = axes[t];
    float sign = signs[t];
    float face_dist = sign < 0.0f ? lf_add(at(box_extents, axis), at(sphere_from_local, axis))
                                  : lf_sub(at(box_extents, axis), at(sphere_from_local, axis));
    if (face_dist < min_dist) {
      min_dist = face_dist;
      closest = sphere_from_local;
      sphere_obb_set_at(&closest, axis, lf_mul(at(box_extents, axis), sign));
      normal = {0.0f, 0.0f, 0.0f};
      sphere_obb_set_at(&normal, axis, sign);
    }
  }
  *out_dist = min_dist;
  *out_closest = closest;
  *out_normal = normal;
}

// V3: sphere_obb_collision_alg.rs:39 process_collision(sphere_obj, sphere_shape, obb_obj, obb_shape, cb, out); the sphere is
// first, the compound is the car. Returns whether V3 leaves `*out = Some(manifold)`.
LF_HDI bool alg_sphere_obb(NpCtx& c, uint32_t sphere_obj, uint32_t compound_obj, Manifold* out) {
  Affine3A sphere_trans = body_trans(c, sphere_obj);
  float radius = body_const(c, sphere_obj).sphere_margin;  // SphereShape::get_radius = margin
  Aabb aabb_1 = sphere_aabb(radius, sphere_trans.translation);

  Affine3A org_trans = body_trans(c, compound_obj);
  Aabb aabb_2 = body_raw_aabb(c, compound_obj);
  if (!aabb_intersects(aabb_1, aabb_2)) return false;

  Affine3A new_child_world_trans = compound_child_world_trans(c, compound_obj);
  (void)org_trans;

  const BodyConst& box = body_const(c, compound_obj);
  Vec3A box_extents = box.box_half_extents;

  Vec3A sphere_from_local = affine_inv_x_form(new_child_world_trans, sphere_trans.translation);

  Vec3A closest = clamp(sphere_from_local, -box_extents, box_extents);
  Vec3A delta = sphere_from_local - closest;
  float dist_sq = length_squared(delta);

  float box_margin = box.box_margin;

  Manifold manifold = manifold_new(sphere_obj, compound_obj);
  float intersection_dist = lf_add(radius, box_margin);
  float contact_dist = lf_add(intersection_dist, manifold_breaking_threshold(c, manifold));
  if (dist_sq > lf_mul(contact_dist, contact_dist)) return false;

  float dist;
  if (dist_sq > LF_EPSILON) {
    dist = lf_sqrt(dist_sq);
    delta = delta / dist;
  } else {
    sphere_obb_get_sphere_penetration(box_extents, sphere_from_local, &dist, &closest, &delta);
    dist = lf_mul(dist, -1.0f);
  }

  Vec3A normal_on_box = transform_vector3a(new_child_world_trans, delta);
  Vec3A point_on_box = transform_point3a(new_child_world_trans, closest);
  float depth = lf_sub(dist, intersection_dist);

  Vec3A point_on_box_plus_margin = point_on_box + normal_on_box * box_margin;
  manifold_add_contact_point(c, &manifold, sphere_obj, compound_obj, normal_on_box, point_on_box_plus_margin, depth, -1);
  manifold_refresh_contact_points(c, &manifold, sphere_obj, compound_obj);

  if (manifold.point_count == 0) return false;
  *out = manifold;
  return true;
}

}  // namespace lfsim
