// The wheel ray query: vehicle/raycaster.rs cast_rays over discrete_dynamics_world.rs ray_test, the ray grid of
// grid_broadphase.rs ray_test (candidate order: the cells of the four ray sources in ray order, each cell's static handles
// then its dynamic handles) and collision_world.rs quad_ray_test (world rays into the body's frame, then the shape's test).
#pragma once
#include "sim/quad_ray_shapes.cuh"

namespace lfsim {
using namespace glam;

// V3: vehicle/raycaster.rs:21 WHEEL_RAY_MIN_HIT_DIST_SQ
constexpr float WHEEL_RAY_MIN_HIT_DIST_SQ = 0.116684f;

struct VehicleRaycasterResult {
  uint32_t some;
  Vec3A hit_point_in_world, hit_normal_in_world;
  uint32_t rigid_body_idx;
};

// V3: collision_world.rs:130 quad_ray_test
LF_HDI void collision_world_quad_ray_test(const ArenaState* s, const StaticData* sd, const Vec3A* ray_from, const Vec3A* ray_to,
                                          uint32_t obj_idx, ClosestQuadRayCb* cb) {
  RbView co = rb_view_p1(s, sd, obj_idx);
  const BodyConst& bc = sd->bodies[obj_idx];
  const Affine3A& co_trans = co.world_trans;
  Vec3A from_l[4], to_l[4];
  bool identity = co_trans.matrix3.x_axis.x == 1.0f && co_trans.matrix3.x_axis.y == 0.0f && co_trans.matrix3.x_axis.z == 0.0f &&
                  co_trans.matrix3.y_axis.x == 0.0f && co_trans.matrix3.y_axis.y == 1.0f && co_trans.matrix3.y_axis.z == 0.0f &&
                  co_trans.matrix3.z_axis.x == 0.0f && co_trans.matrix3.z_axis.y == 0.0f && co_trans.matrix3.z_axis.z == 1.0f;
  if (identity) {
    // Fast path for translation-only bodies: local = world - translation.
    Vec3A t = co_trans.translation;
    for (int i = 0; i < 4; ++i) {
      from_l[i] = ray_from[i] - t;
      to_l[i] = ray_to[i] - t;
    }
  } else {
    Affine3A world_to_co = affine_transpose(co_trans);
    for (int i = 0; i < 4; ++i) {
      from_l[i] = transform_point3a(world_to_co, ray_from[i]);
      to_l[i] = transform_point3a(world_to_co, ray_to[i]);
    }
  }

  QuadRayBridge br;
  for (int i = 0; i < 4; ++i) {
    br.from[i] = from_l[i];
    br.to[i] = to_l[i];
    br.hit_fraction[i] = cb->closest_hit_fraction[i];
  }
  br.obj_rot = co_trans.matrix3;
  br.obj_idx = obj_idx;
  br.cb = cb;

  QuadRayInfo ray_info;
  quad_ray_info_new(&ray_info, from_l, to_l);
  for (int i = 0; i < 4; ++i) ray_info.lambda_max[i] = br.hit_fraction[i];

  switch (bc.shape_kind) {
    case 3: compound_perform_quad_raycast(bc, &br, ray_info); break;
    case 2: sphere_perform_quad_raycast(bc.sphere_margin, &br, ray_info); break;
    case 1:
      for (int p = 0; p < NUM_PLANES; ++p)
        if (sd->planes[p].body == obj_idx) plane_perform_quad_raycast(sd->planes[p], &br, ray_info);
      break;
    default:
      for (int m = 0; m < NUM_MESHES; ++m)
        if (sd->meshes[m].body == obj_idx) mesh_perform_quad_raycast(sd, sd->meshes[m], &br, &ray_info);
      break;
  }
}

// V3: grid_broadphase.rs:58 get_cell_indices + :72 get_cell_idx + :76 cell_indices_to_idx
LF_HDI uint32_t grid_get_cell_idx(const StaticData* sd, Vec3A pos) {
  Vec3A cell_idx_f = (pos - sd->grid_min_pos) / sd->grid_cell_size;
  int32_t ix = lf_f2i(cell_idx_f.x), iy = lf_f2i(cell_idx_f.y), iz = lf_f2i(cell_idx_f.z);  // to_int_unchecked
  ix = ix > 0 ? ix : 0;                                                                     // .max(IVec3::ZERO)
  iy = iy > 0 ? iy : 0;
  iz = iz > 0 ? iz : 0;
  uint32_t x = (uint32_t)ix, y = (uint32_t)iy, z = (uint32_t)iz;                            // as_usizevec3
  uint32_t nx = sd->grid_num_cells[0], ny = sd->grid_num_cells[1], nz = sd->grid_num_cells[2];
  x = x < nx - 1 ? x : nx - 1;                                                              // .min(num_cells - 1)
  y = y < ny - 1 ? y : ny - 1;
  z = z < nz - 1 ? z : nz - 1;
  return x * ny * nz + y * nz + z;
}

// The proxy (BroadphaseProxy) of handle h: its box is state for a dynamic body, constant for a static one.
LF_HDI Aabb proxy_aabb(const ArenaState* s, const StaticData* sd, uint32_t h) {
  for (int d = 0; d < NUM_DYN; ++d)
    if (sd->dyn_handle[d] == h) return s->dyn_proxy[d].aabb;
  return sd->proxies[h].aabb;
}

// V3: quad_ray_callbacks.rs:141 QuadRayCallback::process
LF_HDI void quad_ray_callback_process(const ArenaState* s, const StaticData* sd, uint32_t h, ClosestQuadRayCb* cb) {
  const ProxyConst& proxy = sd->proxies[h];
  if (closest_cb_needs_collision(*cb, proxy.client_obj_idx, proxy.filter_group, proxy.filter_mask)) {
    collision_world_quad_ray_test(s, sd, cb->ray_from_world, cb->ray_to_world, proxy.client_obj_idx, cb);
  }
}

// V3: grid_broadphase.rs:339 ray_test. A bitset of seen handles stands for both V3 paths: the grid keeps unique entries per cell
// (a static handle is pushed only if absent, a dynamic one is added once per cell of its block and removed first when it moves),
// so the single-cell path V3 runs without the set visits the same handles in the same order.
LF_HDI void grid_ray_test(const ArenaState* s, const StaticData* sd, ClosestQuadRayCb* cb) {
  const Vec3A* ray_from = cb->ray_from_world;
  const Vec3A* ray_to = cb->ray_to_world;
  Aabb ray_aabb;
  ray_aabb.min = glam::min(ray_from[0], ray_to[0]);
  ray_aabb.max = glam::max(ray_from[0], ray_to[0]);
  for (int i = 1; i < 4; ++i) {  // bounds.combine(&Aabb::new(from.min(to), from.max(to)))
    ray_aabb.min = glam::min(ray_aabb.min, glam::min(ray_from[i], ray_to[i]));
    ray_aabb.max = glam::max(ray_aabb.max, glam::max(ray_from[i], ray_to[i]));
  }

  // Collect one source cell per ray.
  uint32_t cell_idxs[4] = {0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu};
  int num_cells = 0;
  for (int r = 0; r < 4; ++r) {
    uint32_t idx = grid_get_cell_idx(sd, ray_from[r]);
    bool found = false;
    for (int k = 0; k < num_cells; ++k) found = found || cell_idxs[k] == idx;
    if (!found) {
      cell_idxs[num_cells] = idx;
      num_cells += 1;
    }
  }

  uint32_t seen = 0;  // NUM_HANDLES = 24 handles fit one word
  for (int c = 0; c < num_cells; ++c) {
    uint32_t cell_idx = cell_idxs[c];
    // static_handles first, then dyn_handles
    const CellRange& cr = sd->cells[cell_idx];
    for (uint32_t e = 0; e < cr.count; ++e) {
      uint32_t h = sd->cell_handles[cr.first + e];
      if (seen & (1u << h)) continue;
      seen |= 1u << h;
      if (aabb_intersects(ray_aabb, proxy_aabb(s, sd, h))) quad_ray_callback_process(s, sd, h, cb);
    }
    for (uint32_t e = 0; e < s->n_dyn_cells; ++e) {
      if (s->dyn_cells[e].cell != cell_idx) continue;
      uint32_t h = s->dyn_cells[e].handle;
      if (seen & (1u << h)) continue;
      seen |= 1u << h;
      if (aabb_intersects(ray_aabb, proxy_aabb(s, sd, h))) quad_ray_callback_process(s, sd, h, cb);
    }
  }
}

// V3: vehicle/raycaster.rs:35 cast_rays (the chassis is ignored; the callback's group gains DropshotFloor and min_hit_dist_sq is
// WHEEL_RAY_MIN_HIT_DIST_SQ). A hit body without contact response (a demolished car) turns the ray into no hit.
LF_HDI void raycaster_cast_rays(const ArenaState* s, const StaticData* sd, const Vec3A* from, const Vec3A* to, uint32_t ignore_body_idx,
                                VehicleRaycasterResult out[4]) {
  ClosestQuadRayCb cb;
  closest_cb_new(&cb, from, to, ignore_body_idx);
  cb.collision_filter_group |= FILTER_DROPSHOT_FLOOR;
  cb.min_hit_dist_sq = WHEEL_RAY_MIN_HIT_DIST_SQ;
  grid_ray_test(s, sd, &cb);

  for (int i = 0; i < 4; ++i) {
    out[i].some = 0;
    out[i].hit_point_in_world = splat3(0.0f);
    out[i].hit_normal_in_world = splat3(0.0f);
    out[i].rigid_body_idx = 0;
    if (cb.collision_obj_idx[i] != RAY_NO_BODY) {
      uint32_t co_idx = cb.collision_obj_idx[i];
      RbView rb = rb_view_p1(s, sd, co_idx);
      if (rb_has_contact_response_p1(rb)) {
        out[i].some = 1;
        out[i].rigid_body_idx = co_idx;
        out[i].hit_point_in_world = cb.hit_point_world[i];
        out[i].hit_normal_in_world = normalize_or_zero(cb.hit_normal_world[i]);
      }
    }
  }
}

}  // namespace lfsim
