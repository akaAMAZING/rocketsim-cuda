// P3a: Bullet's 3-axis sweep and prune as V3 wrote it: u16-quantised edges, incremental edge sorts, pair add/remove.
// The axis is a run-time argument here where V3 has a const generic; UPDATE_OVERLAPS is always true on the set_aabb path
// (the add_handle path, false on axes 0 and 1, is construction and not in a tick).
#pragma once
#include "sim/arena_state.cuh"
#include "sim/overlapping_pair_cache.cuh"
#include "sim/collision_dispatcher_release.cuh"

namespace lfsim {
using namespace glam;

LF_HDI bool edge_is_max(Edge e) { return (e.pos & 1) != 0; }  // V3: axis_sweep.rs:24
LF_HDI void edge_swap(Edge* e, uint32_t i, uint32_t j) { Edge t = e[i]; e[i] = e[j]; e[j] = t; }
LF_HDI uint16_t& he_min(ArenaState* s, uint32_t h, int axis) { return s->handle_edges[h].min[axis]; }
LF_HDI uint16_t& he_max(ArenaState* s, uint32_t h, int axis) { return s->handle_edges[h].max[axis]; }

// V3: axis_sweep.rs:310 quantize_point
LF_HDI void quantize_point(const StaticData* sd, Vec3A point, uint16_t is_max, uint16_t out[3]) {
  Vec3A scaled = (point - sd->sap_world_aabb_min) * sd->sap_quantize;
  Vec3A clamped = select(cmpge(scaled, splat3(0.0f)), min(scaled, splat3(lf_u2f(sd->sap_sentinel))), splat3(0.0f));
  out[0] = (uint16_t)(((uint16_t)lf_f2i(clamped.x) & sd->sap_handle_mask) | is_max);
  out[1] = (uint16_t)(((uint16_t)lf_f2i(clamped.y) & sd->sap_handle_mask) | is_max);
  out[2] = (uint16_t)(((uint16_t)lf_f2i(clamped.z) & sd->sap_handle_mask) | is_max);
}

// V3: axis_sweep.rs:326 test_overlap_2d (compares sorted edge indices; touching counts)
LF_HDI bool test_overlap_2d(const ArenaState* s, uint32_t ha, uint32_t hb, int axis) {
  int axis1 = (1 << axis) & 3;
  int axis2 = (1 << axis1) & 3;
  const HandleEdges& a = s->handle_edges[ha];
  const HandleEdges& b = s->handle_edges[hb];
  return !(a.max[axis1] < b.min[axis1] || b.max[axis1] < a.min[axis1] || a.max[axis2] < b.min[axis2] || b.max[axis2] < a.min[axis2]);
}

// V3: axis_sweep.rs:342 add_pair
LF_HDI void sap_add_pair(ArenaState* s, const StaticData* sd, uint32_t ha, uint32_t hb) { add_overlapping_pair(s, sd, ha, hb); }

// V3: axis_sweep.rs:350 remove_pair (the manifold is released at the removal site)
LF_HDI void sap_remove_pair(ArenaState* s, const StaticData* sd, uint32_t ha, uint32_t hb) {
  uint32_t ua = sd->proxies[ha].unique_id, ub = sd->proxies[hb].unique_id;
  uint32_t id0 = ua < ub ? ua : ub, id1 = ua < ub ? ub : ua;
  Pair p;
  if (remove_overlapping_pair(s, sd, id0, id1, &p))
    p3a_release_manifold(s, sd->proxies[p.proxy0].client_obj_idx, sd->proxies[p.proxy1].client_obj_idx);
}

// V3: axis_sweep.rs:366 sort_min_down (a min edge moving down only adds overlaps)
LF_HDI void sort_min_down(ArenaState* s, const StaticData* sd, int axis, uint32_t edge) {
  Edge* e = s->edges[axis];
  uint32_t index = edge;
  uint32_t moving = e[index].handle;
  while (index > 0 && e[index].pos < e[index - 1].pos) {
    Edge previous = e[index - 1];
    uint32_t ph = previous.handle;
    if (edge_is_max(previous)) {
      if (test_overlap_2d(s, moving, ph, axis)) sap_add_pair(s, sd, moving, ph);
      he_max(s, ph, axis) = (uint16_t)(he_max(s, ph, axis) + 1);
    } else {
      he_min(s, ph, axis) = (uint16_t)(he_min(s, ph, axis) + 1);
    }
    he_min(s, moving, axis) = (uint16_t)(he_min(s, moving, axis) - 1);
    edge_swap(e, index, index - 1);
    index -= 1;
  }
}

// V3: axis_sweep.rs:393 sort_max_up (a max edge moving up only adds overlaps)
LF_HDI void sort_max_up(ArenaState* s, const StaticData* sd, int axis, uint32_t edge) {
  Edge* e = s->edges[axis];
  uint32_t index = edge;
  uint32_t moving = e[index].handle;
  const uint32_t len = NUM_EDGES;
  while (index + 1 < len && e[index + 1].handle != 0 && e[index].pos >= e[index + 1].pos) {
    Edge next = e[index + 1];
    uint32_t nh = next.handle;
    if (!edge_is_max(next)) {
      if (test_overlap_2d(s, moving, nh, axis)) sap_add_pair(s, sd, moving, nh);
      he_min(s, nh, axis) = (uint16_t)(he_min(s, nh, axis) - 1);
    } else {
      he_max(s, nh, axis) = (uint16_t)(he_max(s, nh, axis) - 1);
    }
    he_max(s, moving, axis) = (uint16_t)(he_max(s, moving, axis) + 1);
    edge_swap(e, index, index + 1);
    index += 1;
  }
}

// V3: axis_sweep.rs:424 sort_min_up (a min edge moving up only removes overlaps)
LF_HDI void sort_min_up(ArenaState* s, const StaticData* sd, int axis, uint32_t edge) {
  Edge* e = s->edges[axis];
  uint32_t index = edge;
  uint32_t moving = e[index].handle;
  const uint32_t len = NUM_EDGES;
  while (index + 1 < len && e[index + 1].handle != 0 && e[index].pos >= e[index + 1].pos) {
    Edge next = e[index + 1];
    uint32_t nh = next.handle;
    if (edge_is_max(next)) {
      if (test_overlap_2d(s, moving, nh, axis)) sap_remove_pair(s, sd, moving, nh);
      he_max(s, nh, axis) = (uint16_t)(he_max(s, nh, axis) - 1);
    } else {
      he_min(s, nh, axis) = (uint16_t)(he_min(s, nh, axis) - 1);
    }
    he_min(s, moving, axis) = (uint16_t)(he_min(s, moving, axis) + 1);
    edge_swap(e, index, index + 1);
    index += 1;
  }
}

// V3: axis_sweep.rs:458 sort_max_down (a max edge moving down only removes overlaps)
LF_HDI void sort_max_down(ArenaState* s, const StaticData* sd, int axis, uint32_t edge) {
  Edge* e = s->edges[axis];
  uint32_t index = edge;
  uint32_t moving = e[index].handle;
  while (index > 0 && e[index].pos < e[index - 1].pos) {
    Edge previous = e[index - 1];
    uint32_t ph = previous.handle;
    if (!edge_is_max(previous)) {
      if (test_overlap_2d(s, moving, ph, axis)) sap_remove_pair(s, sd, moving, ph);
      he_min(s, ph, axis) = (uint16_t)(he_min(s, ph, axis) + 1);
    } else {
      he_max(s, ph, axis) = (uint16_t)(he_max(s, ph, axis) + 1);
    }
    he_max(s, moving, axis) = (uint16_t)(he_max(s, moving, axis) - 1);
    edge_swap(e, index, index - 1);
    index -= 1;
  }
}

// V3: axis_sweep.rs:216 update_axis
LF_HDI void update_axis(ArenaState* s, const StaticData* sd, int axis, uint32_t handle, uint16_t qmin, uint16_t qmax) {
  Edge* e = s->edges[axis];
  uint32_t min_edge = s->handle_edges[handle].min[axis];
  uint32_t max_edge = s->handle_edges[handle].max[axis];
  int32_t dmin = (int32_t)qmin - (int32_t)e[min_edge].pos;
  int32_t dmax = (int32_t)qmax - (int32_t)e[max_edge].pos;
  if (dmin == 0 && dmax == 0) return;
  e[min_edge].pos = qmin;
  e[max_edge].pos = qmax;
  // expand first (add overlaps), then shrink (remove overlaps)
  if (dmin < 0) sort_min_down(s, sd, axis, min_edge);
  if (dmax > 0) sort_max_up(s, sd, axis, max_edge);
  if (dmin > 0) sort_min_up(s, sd, axis, min_edge);
  if (dmax < 0) sort_max_down(s, sd, axis, max_edge);
}

// V3: axis_sweep.rs:140 AxisSweep3::set_aabb (the handle's proxy aabb is dyn_proxy[d].aabb, stored by the caller)
LF_HDI void axis_sweep_set_aabb(ArenaState* s, const StaticData* sd, uint32_t handle, Aabb aabb) {
  uint16_t qmin[3], qmax[3];
  quantize_point(sd, aabb.min, 0, qmin);
  quantize_point(sd, aabb.max, 1, qmax);
  update_axis(s, sd, 0, handle, qmin[0], qmax[0]);
  update_axis(s, sd, 1, handle, qmin[1], qmax[1]);
  update_axis(s, sd, 2, handle, qmin[2], qmax[2]);
}

}  // namespace lfsim
