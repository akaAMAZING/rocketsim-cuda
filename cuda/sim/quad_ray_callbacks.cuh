// V3 bullet/collision/dispatch/quad_ray_callbacks.rs: ClosestQuadRayResultCallback and the per-body bridge that turns a
// shape's local hit into a world result. The closest hit per ray wins by a strict fraction comparison, so the order in which
// V3 offers candidates (grid_broadphase ray_test, then each shape) decides ties.
#pragma once
#include "sim/arena_state.cuh"
#include "sim/consts.cuh"
#include "sim/linear_piece_curve.cuh"
#include "sim/quad_ray.cuh"
#include "sim/rigid_body_p1.cuh"

namespace lfsim {
using namespace glam;

constexpr uint32_t FILTER_DEFAULT = 1u, FILTER_DROPSHOT_FLOOR = 16u, FILTER_ALL = 0xFFu;  // broadphase CollisionFilterGroups

// ClosestQuadRayResultCallback with its QuadRayResultCallbackBase (quad_ray_callbacks.rs:13-23, :58-65). The rays are the world
// rays; hit_point_world / hit_normal_world are the results.
struct ClosestQuadRayCb {
  float closest_hit_fraction[4];
  uint32_t collision_obj_idx[4];  // RAY_NO_BODY = None
  uint32_t ignore_obj_world_idx;
  uint32_t collision_filter_group, collision_filter_mask;
  float min_hit_dist_sq;
  Vec3A ray_from_world[4], ray_to_world[4];
  Vec3A hit_normal_world[4], hit_point_world[4];
};

// V3: quad_ray_callbacks.rs:68 ClosestQuadRayResultCallback::new (+ Default, :30)
LF_HDI void closest_cb_new(ClosestQuadRayCb* cb, const Vec3A* from, const Vec3A* to, uint32_t ignore_obj) {
  for (int i = 0; i < 4; ++i) {
    cb->closest_hit_fraction[i] = 1.0f;
    cb->collision_obj_idx[i] = RAY_NO_BODY;
    cb->ray_from_world[i] = from[i];
    cb->ray_to_world[i] = to[i];
    cb->hit_normal_world[i] = splat3(0.0f);
    cb->hit_point_world[i] = splat3(0.0f);
  }
  cb->ignore_obj_world_idx = ignore_obj;
  cb->collision_filter_group = FILTER_DEFAULT;
  cb->collision_filter_mask = FILTER_ALL;
  cb->min_hit_dist_sq = 0.0f;
}

// V3: quad_ray_callbacks.rs:47 needs_collision
LF_HDI bool closest_cb_needs_collision(const ClosestQuadRayCb& cb, uint32_t client_obj_idx, uint32_t proxy_group, uint32_t proxy_mask) {
  if (cb.ignore_obj_world_idx == client_obj_idx) return false;
  return (proxy_group & cb.collision_filter_mask) != 0 && (cb.collision_filter_group & proxy_mask) != 0;
}

// V3: bullet/linear_math/util.rs:6 interpolate_3
LF_HDI Vec3A interpolate_3(Vec3A v0, Vec3A v1, float rt) {
  float s = lf_sub(1.0f, rt);
  return s * v0 + rt * v1;
}

// V3: quad_ray_callbacks.rs:92 add_single_result
LF_HDI void closest_cb_add_single_result(ClosestQuadRayCb* cb, uint32_t obj_idx, Vec3A hit_normal_world, float hit_fraction, int ray_idx) {
  if (hit_fraction >= cb->closest_hit_fraction[ray_idx]) return;
  Vec3A candidate = interpolate_3(cb->ray_from_world[ray_idx], cb->ray_to_world[ray_idx], hit_fraction);
  if (cb->min_hit_dist_sq > 0.0f && length_squared(candidate - cb->ray_from_world[ray_idx]) < cb->min_hit_dist_sq) return;
  cb->closest_hit_fraction[ray_idx] = hit_fraction;
  cb->hit_normal_world[ray_idx] = hit_normal_world;
  cb->collision_obj_idx[ray_idx] = obj_idx;
  cb->hit_point_world[ray_idx] = candidate;
}

// BridgeTriQuadRayCallback (quad_ray_callbacks.rs:115-123): the local rays of one body, the hit fractions as the callback held
// them when the body's test began, the body's rotation (local normal to world normal) and the callback.
struct QuadRayBridge {
  Vec3A to[4], from[4];  // local rays
  float hit_fraction[4];
  Mat3A obj_rot;
  uint32_t obj_idx;
  ClosestQuadRayCb* cb;
};

// V3: quad_ray_callbacks.rs:168 internal_report_hit
LF_HDI void bridge_internal_report_hit(QuadRayBridge* br, Vec3A hit_normal_local, float hit_fraction, int ray_idx) {
  Vec3A hit_normal_world = br->obj_rot * hit_normal_local;
  closest_cb_add_single_result(br->cb, br->obj_idx, hit_normal_world, hit_fraction, ray_idx);
}

// V3: quad_ray_callbacks.rs:179 report_hit
LF_HDI void bridge_report_hit(QuadRayBridge* br, Vec3A hit_normal_local, float hit_fraction, int ray_idx) {
  if (hit_fraction >= br->hit_fraction[ray_idx]) return;
  bridge_internal_report_hit(br, hit_normal_local, hit_fraction, ray_idx);
}

// V3: quad_ray_callbacks.rs:187 process_triangle. `tri` is the triangle (points p0 p1 p2, normal) in the body's local frame.
LF_HDI void bridge_process_triangle(QuadRayBridge* br, const Tri& tri, float* lambda_max, int ray_idx) {
  const float EDGE_TOLERANCE = -0.0001f;
  float dist = dot(tri.p0, tri.normal);
  float dist_a = lf_sub(dot(tri.normal, br->from[ray_idx]), dist);
  float dist_b = lf_sub(dot(tri.normal, br->to[ray_idx]), dist);
  if (lf_mul(dist_a, dist_b) >= 0.0f) return;  // same sign

  float proj_length = lf_sub(dist_a, dist_b);
  float distance = lf_div(dist_a, proj_length);
  float hit_fraction = br->hit_fraction[ray_idx];
  if (distance >= hit_fraction) {
    *lambda_max = rs_min(*lambda_max, hit_fraction);
    return;
  }

  Vec3A point = lerp(br->from[ray_idx], br->to[ray_idx], distance);
  Vec3A v0p = tri.p0 - point;
  Vec3A v1p = tri.p1 - point;
  Vec3A cp0 = cross(v0p, v1p);
  if (dot(cp0, tri.normal) < EDGE_TOLERANCE) return;

  Vec3A v2p = tri.p2 - point;
  Vec3A cp1 = cross(v1p, v2p);
  if (dot(cp1, tri.normal) < EDGE_TOLERANCE) return;

  Vec3A cp2 = cross(v2p, v0p);
  if (dot(cp2, tri.normal) < EDGE_TOLERANCE) return;

  float prev_fraction = br->cb->closest_hit_fraction[ray_idx];
  if (dist_a <= 0.0f) {
    bridge_internal_report_hit(br, -tri.normal, distance, ray_idx);
  } else {
    bridge_internal_report_hit(br, tri.normal, distance, ray_idx);
  }

  if (br->cb->closest_hit_fraction[ray_idx] < prev_fraction) *lambda_max = br->cb->closest_hit_fraction[ray_idx];
}

// V3: quad_ray_callbacks.rs:238 process_node (ProcessQuadRayTriangle)
LF_HDI void bridge_process_node(QuadRayBridge* br, const Tri& tri, uint32_t active_mask, float lambda_max[4]) {
  for (int i = 0; i < 4; ++i)
    if ((active_mask & (1u << i)) != 0) bridge_process_triangle(br, tri, &lambda_max[i], i);
}

}  // namespace lfsim
