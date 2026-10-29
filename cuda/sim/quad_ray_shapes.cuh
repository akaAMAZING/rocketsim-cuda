// The quad-ray tests of the four Soccar shapes: the car's compound box (compound_shape.rs), the ball's sphere
// (sphere_shape.rs), the static planes (static_plane_shape.rs) and the triangle meshes through the wide BVH (shared/bvh.rs
// report_quad_ray_overlapping_node, bvh_triangle_mesh_shape.rs, tri_bvh_util.rs). Each takes the body's local rays in
// `QuadRayInfo` and reports into the bridge; the order of reports is V3's.
#pragma once
#include "sim/quad_ray_callbacks.cuh"
#include "common/libm.cuh"

namespace lfsim {
using namespace glam;

// ---- compound (car) -------------------------------------------------------------------------------------------------------------

// V3: compound_shape.rs:217 solve_rounded_box. Returns true with t_hit and the normal when there is a hit.
LF_HDI bool solve_rounded_box(Vec3A source_l, Vec3A dir_l, float dist, Vec3A inner_half, float margin, float* t_out, Vec3A* n_out) {
  float points[8];
  for (int i = 0; i < 8; ++i) points[i] = 0.0f;
  points[0] = 0.0f;
  points[1] = dist;
  int count = 2;

  for (int axis = 0; axis < 3; ++axis) {
    float d = at(dir_l, axis);
    if (d == 0.0f) continue;
    float s = at(source_l, axis);
    float h = at(inner_half, axis);
    for (int bi = 0; bi < 2; ++bi) {
      float bound = bi == 0 ? lf_neg(h) : h;
      float t = lf_div(lf_sub(bound, s), d);
      if (t > 0.0f && t < dist && count < 8) {
        points[count] = t;
        count += 1;
      }
    }
  }

  // Deterministic insertion sort, no allocation.
  for (int i = 1; i < count; ++i) {
    float key = points[i];
    int j = i;
    while (j > 0 && points[j - 1] > key) {
      points[j] = points[j - 1];
      j -= 1;
    }
    points[j] = key;
  }

  float margin_sq = lf_mul(margin, margin);

  for (int i = 0; i < count - 1; ++i) {
    float t0 = points[i];
    float t1 = points[i + 1];
    if (t1 <= t0) continue;

    float mid = lf_add(t0, lf_mul(lf_sub(t1, t0), 0.5f));
    Vec3A p_mid = source_l + dir_l * mid;

    float a = 0.0f, b = 0.0f, c = 0.0f;
    uint32_t outside = 0;
    for (int axis = 0; axis < 3; ++axis) {
      float pm = at(p_mid, axis);
      float h = at(inner_half, axis);
      float bound;
      if (pm < lf_neg(h)) {
        bound = lf_neg(h);
      } else if (pm > h) {
        bound = h;
      } else {
        continue;
      }
      outside += 1;

      float o = lf_sub(at(source_l, axis), bound);
      float d = at(dir_l, axis);

      a = lf_add(a, lf_mul(d, d));
      b = lf_add(b, lf_mul(lf_mul(2.0f, d), o));
      c = lf_add(c, lf_mul(o, o));
    }
    if (outside == 0) continue;

    float cc = lf_sub(c, margin_sq);
    float disc = lf_sub(lf_mul(b, b), lf_mul(lf_mul(4.0f, a), cc));
    if (disc < 0.0f) {
      float scale = lf_add(lf_mul(b, b), lf_abs(lf_mul(lf_mul(4.0f, a), cc)));
      if (disc > lf_mul(lf_neg(c::F32_EPSILON), scale)) {
        disc = 0.0f;
      } else {
        continue;
      }
    }

    float sqrt_d = lf_sqrt(disc);
    float t_hit;
    if (disc == 0.0f) {
      t_hit = lf_div(lf_neg(b), lf_mul(2.0f, a));
    } else {
      float q = lf_mul(-0.5f, b >= 0.0f ? lf_add(b, sqrt_d) : lf_sub(b, sqrt_d));
      if (q == 0.0f) {
        t_hit = lf_div(lf_neg(b), lf_mul(2.0f, a));
      } else {
        float r0 = lf_div(q, a);
        float r1 = lf_div(cc, q);
        t_hit = r0 <= r1 ? r0 : r1;
      }
    }

    if (t_hit <= 0.0f || t_hit > dist || t_hit < t0 || t_hit > t1) continue;

    Vec3A hit = source_l + dir_l * t_hit;
    Vec3A clamped = glam::clamp(hit, -inner_half, inner_half);
    Vec3A n = hit - clamped;
    float len_sq = length_squared(n);
    if (len_sq <= lf_mul(c::F32_EPSILON, c::F32_EPSILON)) continue;

    // Entering hit must oppose the ray; keep tangency (dot == 0).
    Vec3A n_norm = n / lf_sqrt(len_sq);
    if (dot(n_norm, dir_l) > 0.0f) continue;

    *t_out = t_hit;
    *n_out = n_norm;
    return true;
  }
  return false;
}

// V3: compound_shape.rs:193 analytic_rounded_hit
LF_HDI bool analytic_rounded_hit(const BodyConst& bc, Vec3A ray_source, Vec3A dir, float dist, float* t_out, Vec3A* n_out) {
  Vec3A source_l = ray_source - bc.child_trans.translation;
  Vec3A dir_l = dir;
  float t_hit;
  Vec3A normal_c;
  if (!solve_rounded_box(source_l, dir_l, dist, bc.box_half_extents, bc.box_margin, &t_hit, &normal_c)) return false;
  // Entering hit must oppose the ray. Allow exact tangency (dot == 0).
  normal_c = normal_c / length(normal_c);
  if (dot(normal_c, dir) > 0.0f) return false;
  *t_out = t_hit;
  *n_out = normal_c;
  return true;
}

// V3: compound_shape.rs:85 internal_perform_raycast
LF_HDI void compound_internal_perform_raycast(const BodyConst& bc, QuadRayBridge* br, Vec3A ray_source, Vec3A ray_target, int ray_idx) {
  const Aabb& local_aabb = bc.compound_local_aabb;
  Vec3A delta = ray_target - ray_source;
  float dist = length(delta);
  Vec3A dir = delta / dist;

  // implementation of the slab method to handle `dir` potentially having elements that are `0`
  float tenter = 0.0f;
  float texit = dist;
  int hit_axis = 0;

  Vec3A inv = 1.0f / dir;
  Vec3A t1 = (local_aabb.min - ray_source) * inv;
  Vec3A t2 = (local_aabb.max - ray_source) * inv;

  Vec3A tmin = glam::min(t1, t2);
  Vec3A tmax = glam::max(t1, t2);

  // is_negative_bitmask: the sign bits of the three lanes
  uint32_t is_neg = (lf_bits(tmax.x) >> 31) | ((lf_bits(tmax.y) >> 31) << 1) | ((lf_bits(tmax.z) >> 31) << 2);
  BVec3A fin = glam::is_finite_mask(inv);
  uint32_t is_finite[3] = {fin.x, fin.y, fin.z};
  for (int axis = 0; axis < 3; ++axis) {
    if (!is_finite[axis]) {
      float origin = at(ray_source, axis);
      float mn = at(local_aabb.min, axis);
      float mx = at(local_aabb.max, axis);

      // parallel - if the origin not within slab, no hit
      if (mn > origin || origin > mx) return;

      // Axis does not clip the interval
      continue;
    }

    if (is_neg & (1u << axis)) return;

    texit = rs_min(texit, at(tmax, axis));
    if (at(tmin, axis) > tenter) {
      tenter = at(tmin, axis);
      hit_axis = axis;
    }

    if (tenter > texit) return;
  }

  if (0.0f >= tenter || tenter > dist) return;

  Vec3A inner_half = bc.box_half_extents;

  // Pure-face fast path for translation-only child transform.
  {
    Vec3A entry = ray_source + dir * tenter;
    Vec3A local = entry - bc.child_trans.translation;
    bool inside_face = true;
    for (int axis = 0; axis < 3; ++axis) {
      if (axis == hit_axis) continue;
      if (lf_abs(at(local, axis)) > at(inner_half, axis)) {
        inside_face = false;
        break;
      }
    }

    if (inside_face) {
      Vec3A hit_normal = splat3(0.0f);
      float sg = lf_neg(rs_signum(at(dir, hit_axis)));
      if (hit_axis == 0) hit_normal.x = sg; else if (hit_axis == 1) hit_normal.y = sg; else hit_normal.z = sg;

      if (length_squared(hit_normal) > 0.0f) {
        float hit_fraction = lf_div(rs_max(tenter, 0.0f), dist);
        bridge_report_hit(br, hit_normal, hit_fraction, ray_idx);
      }
      return;
    }
  }

  // Edge or corner entry: exact rounded box solve.
  float t_hit;
  Vec3A normal_compound;
  if (!analytic_rounded_hit(bc, ray_source, dir, dist, &t_hit, &normal_compound)) return;

  if (0.0f >= t_hit || t_hit > dist) return;

  float hit_fraction = lf_div(t_hit, dist);
  bridge_report_hit(br, normal_compound, hit_fraction, ray_idx);
}

// V3: compound_shape.rs:53 perform_quad_raycast
LF_HDI void compound_perform_quad_raycast(const BodyConst& bc, QuadRayBridge* br, const QuadRayInfo& ray_info) {
  const Aabb& box_aabb = bc.compound_local_aabb;
  if (!aabb_intersects(ray_info.aabb, box_aabb)) return;

  float origins[3][4], inv_dirs[3][4];
  quad_ray_calc_pos_dir(ray_info, origins, inv_dirs);
  uint32_t mask = intersect_quad_ray_aabb(origins, inv_dirs, box_aabb, br->hit_fraction);

  for (int i = 0; i < 4; ++i) {
    if ((mask & (1u << i)) == 0) continue;
    compound_internal_perform_raycast(bc, br, ray_info.src[i], ray_info.tgt[i], i);
  }
}

// ---- sphere (ball) ---------------------------------------------------------------------------------------------------------------

// V3: sphere_shape.rs:57 perform_quad_raycast. `radius` is SphereShape::get_radius (the shape's margin).
LF_HDI void sphere_perform_quad_raycast(float radius, QuadRayBridge* br, const QuadRayInfo& ray_info) {
  float sx[4], sy[4], sz[4], tx[4], ty[4], tz[4];
  for (int l = 0; l < 4; ++l) {
    sx[l] = ray_info.src[l].x; sy[l] = ray_info.src[l].y; sz[l] = ray_info.src[l].z;
    tx[l] = ray_info.tgt[l].x; ty[l] = ray_info.tgt[l].y; tz[l] = ray_info.tgt[l].z;
  }

  float radius_sq = lf_mul(radius, radius);
  bool within[4];
  bool any_within = false;
  for (int l = 0; l < 4; ++l) {
    float amin_x = sse_min(sx[l], tx[l]), amin_y = sse_min(sy[l], ty[l]), amin_z = sse_min(sz[l], tz[l]);
    float amax_x = sse_max(sx[l], tx[l]), amax_y = sse_max(sy[l], ty[l]), amax_z = sse_max(sz[l], tz[l]);
    // Vec4::ZERO.clamp(min, max) = ZERO.max(min).min(max)
    float cx = sse_min(sse_max(0.0f, amin_x), amax_x);
    float cy = sse_min(sse_max(0.0f, amin_y), amax_y);
    float cz = sse_min(sse_max(0.0f, amin_z), amax_z);
    float closest_len_sq = lf_add(lf_add(lf_mul(cx, cx), lf_mul(cy, cy)), lf_mul(cz, cz));
    within[l] = closest_len_sq <= radius_sq;
    any_within = any_within || within[l];
  }
  if (!any_within) return;

  float dist[4], dirx[4], diry[4], dirz[4];
  float disc[4];
  bool valid[4];
  bool any_valid = false;
  for (int l = 0; l < 4; ++l) {
    float dx = lf_sub(tx[l], sx[l]), dy = lf_sub(ty[l], sy[l]), dz = lf_sub(tz[l], sz[l]);
    dist[l] = lf_sqrt(lf_add(lf_add(lf_mul(dx, dx), lf_mul(dy, dy)), lf_mul(dz, dz)));
    dirx[l] = lf_div(dx, dist[l]);
    diry[l] = lf_div(dy, dist[l]);
    dirz[l] = lf_div(dz, dist[l]);
    float b = lf_mul(2.0f, lf_add(lf_add(lf_mul(sx[l], dirx[l]), lf_mul(sy[l], diry[l])), lf_mul(sz[l], dirz[l])));
    float c = lf_sub(lf_add(lf_add(lf_mul(sx[l], sx[l]), lf_mul(sy[l], sy[l])), lf_mul(sz[l], sz[l])), radius_sq);
    disc[l] = lf_sub(lf_mul(b, b), lf_mul(4.0f, c));
    valid[l] = (disc[l] >= 0.0f) && within[l];
    any_valid = any_valid || valid[l];
  }
  if (!any_valid) return;

  float t[4];
  bool nearest[4];
  bool any_nearest = false;
  for (int l = 0; l < 4; ++l) {
    float b = lf_mul(2.0f, lf_add(lf_add(lf_mul(sx[l], dirx[l]), lf_mul(sy[l], diry[l])), lf_mul(sz[l], dirz[l])));
    float sqrt_disc = lf_sqrt(disc[l]);
    float t0 = lf_div(lf_sub(lf_neg(b), sqrt_disc), 2.0f);
    float t1 = lf_div(lf_add(lf_neg(b), sqrt_disc), 2.0f);

    bool swap_mask = t0 > t1;
    t0 = swap_mask ? t1 : t0;
    t1 = swap_mask ? t0 : t1;  // sic: V3 selects with the already-updated t0

    bool t0_nearest = (t0 >= 0.0f) && (t0 <= dist[l]);
    bool t1_nearest = (t1 >= 0.0f) && (t1 <= dist[l]) && !t0_nearest;
    nearest[l] = valid[l] && (t0_nearest || t1_nearest);
    any_nearest = any_nearest || nearest[l];
    t[l] = t0_nearest ? t0 : t1;
  }
  if (!any_nearest) return;

  for (int l = 0; l < 4; ++l) {
    if (!nearest[l]) continue;
    float hpx = lf_add(sx[l], lf_mul(dirx[l], t[l]));
    float hpy = lf_add(sy[l], lf_mul(diry[l], t[l]));
    float hpz = lf_add(sz[l], lf_mul(dirz[l], t[l]));
    Vec3A normal = {lf_div(hpx, radius), lf_div(hpy, radius), lf_div(hpz, radius)};
    float hit_fraction = lf_div(t[l], dist[l]);
    bridge_report_hit(br, normal, hit_fraction, l);
  }
}

// ---- static plane ----------------------------------------------------------------------------------------------------------------

// V3: static_plane_shape.rs:82 perform_quad_raycast
LF_HDI void plane_perform_quad_raycast(const Plane& pl, QuadRayBridge* br, const QuadRayInfo& ray_info) {
  const Aabb& plane = pl.aabb_ident_cache;
  if (!aabb_intersects(ray_info.aabb, plane)) return;

  float sx[4], sy[4], sz[4], dx[4], dy[4], dz[4];
  float origins[3][4], inv_delta[3][4];
  for (int l = 0; l < 4; ++l) {
    sx[l] = ray_info.src[l].x; sy[l] = ray_info.src[l].y; sz[l] = ray_info.src[l].z;
    dx[l] = lf_sub(ray_info.tgt[l].x, sx[l]);
    dy[l] = lf_sub(ray_info.tgt[l].y, sy[l]);
    dz[l] = lf_sub(ray_info.tgt[l].z, sz[l]);
    origins[0][l] = sx[l]; origins[1][l] = sy[l]; origins[2][l] = sz[l];
    inv_delta[0][l] = lf_div(1.0f, dx[l]);
    inv_delta[1][l] = lf_div(1.0f, dy[l]);
    inv_delta[2][l] = lf_div(1.0f, dz[l]);
  }

  uint32_t ray_mask = intersect_quad_ray_aabb(origins, inv_delta, plane, br->hit_fraction);
  if (ray_mask == 0) return;

  Vec3A n = pl.plane_normal;
  float dist[4], dir_align[4], t[4];
  uint32_t align_mask = 0, t_mask = 0;
  for (int l = 0; l < 4; ++l) {
    dist[l] = lf_sqrt(lf_add(lf_add(lf_mul(dx[l], dx[l]), lf_mul(dy[l], dy[l])), lf_mul(dz[l], dz[l])));
    float inv_dist = lf_div(1.0f, dist[l]);
    float dirx = lf_mul(dx[l], inv_dist), diry = lf_mul(dy[l], inv_dist), dirz = lf_mul(dz[l], inv_dist);
    dir_align[l] = lf_add(lf_add(lf_mul(dirx, n.x), lf_mul(diry, n.y)), lf_mul(dirz, n.z));
    if (lf_abs(dir_align[l]) >= c::F32_EPSILON) align_mask |= 1u << l;
  }
  if (align_mask == 0) return;

  for (int l = 0; l < 4; ++l) {
    float normal_start = lf_add(lf_add(lf_mul(sx[l], n.x), lf_mul(sy[l], n.y)), lf_mul(sz[l], n.z));
    t[l] = lf_div(lf_neg(normal_start), dir_align[l]);
    if (t[l] >= 0.0f && t[l] < dist[l]) t_mask |= 1u << l;
  }

  uint32_t hit_mask = ray_mask & (align_mask & t_mask);
  if (hit_mask == 0) return;

  for (int l = 0; l < 4; ++l) {
    if ((hit_mask & (1u << l)) == 0) continue;
    float hit_fraction = lf_div(t[l], dist[l]);
    bridge_report_hit(br, n, hit_fraction, l);
  }
}

// ---- triangle mesh through the wide BVH -----------------------------------------------------------------------------------------------

// V3: bvh_triangle_mesh_shape.rs:60 perform_quad_raycast over shared/bvh.rs:275 report_quad_ray_overlapping_node, with
// tri_bvh_util.rs:47 QuadRayNodeOverlapCallback::process_node (the triangle of a leaf goes to the bridge). Node and leaf indices
// are local to the mesh.
LF_HDI void mesh_perform_quad_raycast(const StaticData* sd, const MeshInfo& mesh, QuadRayBridge* br, QuadRayInfo* ray_info) {
  const uint32_t LEAF_BIT = 1u << 31;
  if (!aabb_intersects(ray_info->aabb, mesh.root_aabb)) return;

  float origins[3][4], inv_dirs[3][4];
  quad_ray_calc_pos_dir(*ray_info, origins, inv_dirs);
  uint32_t stack[128];  // TRAVERSAL_STACK_SIZE
  stack[0] = 0;         // WideChild::branch(0)
  int stack_len = 1;
  while (stack_len != 0) {
    stack_len -= 1;
    uint32_t work = stack[stack_len];
    if (work & LEAF_BIT) {
      const Leaf& leaf = sd->leaves[mesh.leaf_first + (work & ~LEAF_BIT)];
      uint32_t mask = intersect_quad_ray_aabb(origins, inv_dirs, leaf.bounds, ray_info->lambda_max);
      if (mask != 0) bridge_process_node(br, sd->tris[mesh.tri_first + leaf.leaf_idx], mask, ray_info->lambda_max);
      continue;
    }

    const WideNode& node = sd->nodes[mesh.node_first + work];
    uint32_t mask = wide_node_intersection_mask(node, ray_info->aabb);
    for (int lane = (int)node.child_count - 1; lane >= 0; --lane) {
      if (mask & (1u << lane)) {
        stack[stack_len] = node.children[lane];
        stack_len += 1;
      }
    }
  }
}

}  // namespace lfsim
