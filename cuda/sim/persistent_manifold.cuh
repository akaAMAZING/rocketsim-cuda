// V3: bullet/collision/narrowphase/persistent_manifold.rs and manifold_point.rs. A `Manifold` (arena_state.cuh) is V3's
// PersistentManifold: its point cache is `points[0..point_count]`, slots from point_count on are kept zero (V3's dump writes
// them as default points). The two thresholds are the min over the two bodies' (derived, not stored). V3 passes the two
// RigidBody refs; here the callers pass the two body indices (the manifold's body0/body1, or an algorithm's own order).
#pragma once
#include "sim/narrowphase.cuh"
#include "sim/arena_contact_tracker.cuh"

namespace lfsim {

// V3: persistent_manifold.rs:57 new(body0, body1): an empty manifold in body order (the two thresholds are derived).
LF_HDI Manifold manifold_new(uint32_t body0, uint32_t body1) {
  Manifold m = {};
  m.body0_idx = body0;
  m.body1_idx = body1;
  return m;
}

// V3: persistent_manifold.rs:60-63 contact_breaking_threshold: min over the two bodies (shapes' cached thresholds)
LF_HDI float manifold_breaking_threshold(const NpCtx& c, const Manifold& m) {
  return rs_min(body_const(c, m.body0_idx).contact_breaking_threshold, body_const(c, m.body1_idx).contact_breaking_threshold);
}

// ArrayVec::swap_remove(i) on the point cache; the vacated slot is zeroed (a default point in V3's dump).
LF_HDI void manifold_swap_remove(Manifold* m, uint32_t i) {
  uint32_t last = m->point_count - 1;
  if (i != last) m->points[i] = m->points[last];
  ManifoldPoint z = {};
  m->points[last] = z;
  m->point_count = last;
}

// V3: persistent_manifold.rs:83 calculate_combined_friction
LF_HDI float manifold_combined_friction(const NpCtx& c, uint32_t b0, uint32_t b1) {
  float f0 = body_const(c, b0).friction, f1 = body_const(c, b1).friction;
  if (body_is_static(c, b0) || body_is_static(c, b1)) return rs_min(f0, f1);
  return lf_mul(f0, f1);
}

// V3: persistent_manifold.rs:91 calculate_combined_restitution
LF_HDI float manifold_combined_restitution(const NpCtx& c, uint32_t b0, uint32_t b1) {
  float r0 = body_const(c, b0).restitution, r1 = body_const(c, b1).restitution;
  if (body_is_static(c, b0) || body_is_static(c, b1)) return rs_max(r0, r1);
  return lf_mul(r0, r1);
}

// V3: persistent_manifold.rs:99 get_res
LF_HDI float manifold_get_res(Vec3A new_contact_local, Vec3A point1, Vec3A point2, Vec3A point3) {
  return length_squared(cross(new_contact_local - point1, point2 - point3));
}

// V3: persistent_manifold.rs:107-141 get_res_0..3 (the three other points, in V3's order)
LF_HDI float manifold_get_res_n(const Manifold& m, int n, Vec3A p) {
  const ManifoldPoint* pc = m.points;
  switch (n) {
    case 0: return manifold_get_res(p, pc[1].local_point_a, pc[3].local_point_a, pc[2].local_point_a);
    case 1: return manifold_get_res(p, pc[0].local_point_a, pc[3].local_point_a, pc[2].local_point_a);
    case 2: return manifold_get_res(p, pc[0].local_point_a, pc[3].local_point_a, pc[1].local_point_a);
    default: return manifold_get_res(p, pc[0].local_point_a, pc[2].local_point_a, pc[1].local_point_a);
  }
}

// V3: persistent_manifold.rs:143 sort_cached_points: the slot a full cache replaces (closestAxis4: first maximum wins)
LF_HDI uint32_t manifold_sort_cached_points(const Manifold& m, const ManifoldPoint& new_contact) {
  uint32_t max_penetration_idx = MAX_POINTS;
  float max_penetration = new_contact.distance_1;
  for (uint32_t i = 0; i < m.point_count; ++i) {
    if (m.points[i].distance_1 < max_penetration) {
      max_penetration_idx = i;
      max_penetration = m.points[i].distance_1;
    }
  }

  Vec3A p = new_contact.local_point_a;
  Vec4 res;
  res.x = max_penetration_idx == 0 ? 0.0f : manifold_get_res_n(m, 0, p);
  res.y = max_penetration_idx == 1 ? 0.0f : manifold_get_res_n(m, 1, p);
  res.z = max_penetration_idx == 2 ? 0.0f : manifold_get_res_n(m, 2, p);
  res.w = max_penetration_idx == 3 ? 0.0f : manifold_get_res_n(m, 3, p);

  uint32_t biggest_area = 0;
  float max_area = res.x;
  if (res.y > max_area) {
    biggest_area = 1;
    max_area = res.y;
  }
  if (res.z > max_area) {
    biggest_area = 2;
    max_area = res.z;
  }
  if (res.w > max_area) {
    biggest_area = 3;
  }
  return biggest_area;
}

// V3: persistent_manifold.rs:213 get_cache_entry: nearest cached point within the breaking threshold, or -1
LF_HDI int32_t manifold_get_cache_entry(const NpCtx& c, const Manifold& m, const ManifoldPoint& new_contact) {
  float cbt = manifold_breaking_threshold(c, m);
  float threshold_sq = lf_mul(cbt, cbt);
  float shortest_dist = threshold_sq;
  int32_t nearest_point = -1;
  for (uint32_t index = 0; index < m.point_count; ++index) {
    float distance_sq = length_squared(m.points[index].local_point_a - new_contact.local_point_a);
    if (distance_sq < shortest_dist) {
      shortest_dist = distance_sq;
      nearest_point = (int32_t)index;
    }
  }
  return nearest_point;
}

// V3: persistent_manifold.rs:228 replace_contact_point (applied_impulse is always 0 and not stored)
LF_HDI void manifold_replace_contact_point(Manifold* m, uint32_t index, const ManifoldPoint& contact) { m->points[index] = contact; }

// V3: persistent_manifold.rs:234 add_manifold_point
LF_HDI uint32_t manifold_add_manifold_point(Manifold* m, const ManifoldPoint& contact) {
  uint32_t num_points = m->point_count;
  if (num_points == (uint32_t)MAX_POINTS) {
    uint32_t idx = manifold_sort_cached_points(*m, contact);
    m->points[idx] = contact;
    return idx;
  }
  m->points[num_points] = contact;
  m->point_count = num_points + 1;
  return num_points;
}

// V3: persistent_manifold.rs:283 add_contact_without_callback
LF_HDI uint32_t manifold_add_contact_without_callback(const NpCtx& c, Manifold* m, const ManifoldPoint& contact) {
  int32_t insert_idx = manifold_get_cache_entry(c, *m, contact);
  if (insert_idx >= 0) {
    manifold_replace_contact_point(m, (uint32_t)insert_idx, contact);
    return (uint32_t)insert_idx;
  }
  return manifold_add_manifold_point(m, contact);
}

// V3: persistent_manifold.rs:243 add_contact_point(body0, body1, normal_on_b_in_world, point_in_world, depth, idx_1, callback).
// `idx_1` is the triangle index or -1 for None.
LF_HDI void manifold_add_contact_point(NpCtx& c, Manifold* m, uint32_t body0, uint32_t body1, Vec3A normal_on_b_in_world,
                                       Vec3A point_in_world, float depth, int32_t idx_1) {
  if (depth > manifold_breaking_threshold(c, *m)) return;

  Vec3A point_a = point_in_world + normal_on_b_in_world * depth;
  Vec3A local_a = affine_inv_x_form(body_trans(c, body0), point_a);
  Vec3A local_b = affine_inv_x_form(body_trans(c, body1), point_in_world);

  // ManifoldPoint::new(point_a, point_b, normal, distance) (manifold_point.rs:21), zero elsewhere
  ManifoldPoint new_pt = {};
  new_pt.local_point_a = local_a;
  new_pt.local_point_b = local_b;
  new_pt.normal_world_on_b = normal_on_b_in_world;
  new_pt.distance_1 = depth;
  new_pt.pos_world_on_a = point_a;
  new_pt.pos_world_on_b = point_in_world;

  new_pt.combined_friction = manifold_combined_friction(c, body0, body1);
  new_pt.combined_restitution = manifold_combined_restitution(c, body0, body1);

  uint32_t insert_idx = manifold_add_contact_without_callback(c, m, new_pt);

  if ((body_flags(c, body0) & CF_CUSTOM_MATERIAL) != 0 || (body_flags(c, body1) & CF_CUSTOM_MATERIAL) != 0) {
    tracker_callback(c, &m->points[insert_idx], body0, body1, idx_1);
  }
}

// V3: persistent_manifold.rs:296 merge_contact_points(other)
LF_HDI void manifold_merge_contact_points(const NpCtx& c, Manifold* m, const Manifold& other) {
  if (m->point_count == 0) {
    for (uint32_t i = 0; i < other.point_count; ++i) {
      m->points[m->point_count] = other.points[i];
      m->point_count += 1;
    }
  } else {
    for (uint32_t i = 0; i < other.point_count; ++i) manifold_add_contact_without_callback(c, m, other.points[i]);
  }
}

// V3: persistent_manifold.rs:312 refresh_contact_points(body0, body1): re-project carried points and cull separated ones
LF_HDI void manifold_refresh_contact_points(const NpCtx& c, Manifold* m, uint32_t body0, uint32_t body1) {
  if (m->point_count == 0) return;

  Affine3A tr_a = body_trans(c, body0);
  Affine3A tr_b = body_trans(c, body1);

  for (uint32_t i = 0; i < m->point_count; ++i) {
    ManifoldPoint& mp = m->points[i];
    mp.pos_world_on_a = transform_point3a(tr_a, mp.local_point_a);
    mp.pos_world_on_b = transform_point3a(tr_b, mp.local_point_b);
    mp.distance_1 = dot(mp.pos_world_on_a - mp.pos_world_on_b, mp.normal_world_on_b);
  }

  float cbt = manifold_breaking_threshold(c, *m);
  float contact_breaking_threshold_sq = lf_mul(cbt, cbt);

  for (int32_t i = (int32_t)m->point_count - 1; i >= 0; --i) {
    ManifoldPoint point = m->points[i];
    if (point.distance_1 > cbt) {
      // contact becomes invalid when signed distance exceeds margin (projected on contact normal direction)
      manifold_swap_remove(m, (uint32_t)i);
      continue;
    }

    Vec3A projected_point = point.pos_world_on_a - point.normal_world_on_b * point.distance_1;
    Vec3A projected_difference = point.pos_world_on_b - projected_point;
    float distance_2d = dot(projected_difference, projected_difference);
    if (distance_2d > contact_breaking_threshold_sq) {
      // contact also becomes invalid when relative movement orthogonal to normal exceeds margin
      manifold_swap_remove(m, (uint32_t)i);
    }
  }
}

}  // namespace lfsim
