// P3a: CollisionWorld::update_aabbs with the shape AABBs it calls.
#pragma once
#include "sim/grid_broadphase.cuh"
#include "sim/bullet_consts.cuh"

namespace lfsim {
using namespace glam;

constexpr float SPHERE_RADIUS_MARGIN = 0.08f;        // V3: shapes/sphere_shape.rs:11

// V3: shapes/sphere_shape.rs:39 SphereShape::get_aabb (get_margin() is the radius)
LF_HDI Aabb sphere_get_aabb(const BodyConst& b, Vec3A center) {
  float margin = lf_add(b.sphere_margin, SPHERE_RADIUS_MARGIN);
  Vec3A extent = splat3(margin);
  return Aabb{center - extent, center + extent};
}

// V3: shapes/compound_shape.rs:29 CompoundShape::get_aabb
LF_HDI Aabb compound_get_aabb(const BodyConst& b, const Affine3A& t) {
  Vec3A local_half_extents = 0.5f * (b.compound_local_aabb.max - b.compound_local_aabb.min);
  Vec3A local_center = 0.5f * (b.compound_local_aabb.max + b.compound_local_aabb.min);
  Mat3A abs_b = Mat3A{abs(t.matrix3.x_axis), abs(t.matrix3.y_axis), abs(t.matrix3.z_axis)};
  Vec3A center = transform_point3a(t, local_center);
  Vec3A extent = abs_b * local_half_extents;
  return Aabb{center - extent, center + extent};
}

// V3: shapes/collision_shape.rs:38 CollisionShapes::get_aabb (the dynamic shapes: sphere 2, compound 3)
LF_HDI Aabb shape_get_aabb(const BodyConst& b, const Affine3A& t) {
  return b.shape_kind == 2 ? sphere_get_aabb(b, t.translation) : compound_get_aabb(b, t);
}

// V3: collision_world.rs:66 update_aabbs: the dynamic bodies in collision_objs order (static ones keep their creation AABB)
LF_HDI void update_aabbs(ArenaState* s, const StaticData* sd) {
  const Vec3A cbt = splat3(CONTACT_BREAKING_THRESHOLD);
  for (uint32_t d = 0; d < (uint32_t)NUM_DYN; ++d) {
    const BodyConst& bc = sd->bodies[sd->dyn_body[d]];
    const Body& b = s->bodies[d];
    if ((b.collision_flags & 1u) != 0) continue;  // is_static_obj
    Aabb aabb = shape_get_aabb(bc, b.world_trans);
    aabb.min -= cbt;
    aabb.max += cbt;
    Aabb aabb2 = shape_get_aabb(bc, b.interp_world_trans);
    aabb2.min -= cbt;
    aabb2.max += cbt;
    aabb.min = min(aabb.min, aabb2.min);  // aabb += aabb2
    aabb.max = max(aabb.max, aabb2.max);
    grid_set_aabb_dynamic(s, sd, d, aabb);
  }
}

}  // namespace lfsim
