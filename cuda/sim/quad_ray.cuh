// V3 shared/quad_ray.rs: the four-ray packet, its union AABB, the inverse directions and the slab test. Vec4 lanes are
// float[4] arrays; each lane computes what the SSE lane computes (min/max keep the _mm_min_ps / _mm_max_ps operand order).
#pragma once
#include "sim/arena_state.cuh"
#include "sim/consts.cuh"
#include "sim/bvh.cuh"
#include "sim/linear_piece_curve.cuh"

namespace lfsim {
using namespace glam;

struct QuadRayInfo {
  Vec3A src[4], tgt[4];  // ray_sources, ray_targets
  float lambda_max[4];
  Aabb aabb;
};

// V3: shared/quad_ray.rs:18 QuadRayInfo::new (lambda_max = ONE)
LF_HDI void quad_ray_info_new(QuadRayInfo* r, const Vec3A* src, const Vec3A* tgt) {
  for (int i = 0; i < 4; ++i) {
    r->src[i] = src[i];
    r->tgt[i] = tgt[i];
    r->lambda_max[i] = 1.0f;
  }
  // Union AABB of the 4 segments: Aabb::new(min, max) then `aabb += ...` (min.min(rhs.min), max.max(rhs.max)).
  r->aabb.min = glam::min(src[0], tgt[0]);
  r->aabb.max = glam::max(src[0], tgt[0]);
  for (int i = 1; i < 4; ++i) {
    r->aabb.min = glam::min(r->aabb.min, glam::min(src[i], tgt[i]));
    r->aabb.max = glam::max(r->aabb.max, glam::max(src[i], tgt[i]));
  }
}

// V3: shared/quad_ray.rs:40 calc_pos_dir. origins[axis][lane], inv_dirs[axis][lane].
LF_HDI void quad_ray_calc_pos_dir(const QuadRayInfo& r, float origins[3][4], float inv_dirs[3][4]) {
  for (int l = 0; l < 4; ++l) {
    origins[0][l] = r.src[l].x;
    origins[1][l] = r.src[l].y;
    origins[2][l] = r.src[l].z;
    float inv0 = lf_div(1.0f, lf_sub(r.tgt[l].x, r.src[l].x));
    float inv1 = lf_div(1.0f, lf_sub(r.tgt[l].y, r.src[l].y));
    float inv2 = lf_div(1.0f, lf_sub(r.tgt[l].z, r.src[l].z));
    // Vec4::select(inv.is_finite_mask(), inv, Vec4::MAX)
    inv_dirs[0][l] = lf_isfinite(inv0) ? inv0 : c::F32_MAX;
    inv_dirs[1][l] = lf_isfinite(inv1) ? inv1 : c::F32_MAX;
    inv_dirs[2][l] = lf_isfinite(inv2) ? inv2 : c::F32_MAX;
  }
}

// V3: shared/quad_ray.rs:101 intersect_quad_ray_aabb: bit i of the result is set when ray i overlaps the box.
LF_HDI uint32_t intersect_quad_ray_aabb(const float origins[3][4], const float inv_dir[3][4], const Aabb& bounds,
                                        const float lambda_max[4]) {
  uint32_t mask = 0;
  for (int l = 0; l < 4; ++l) {
    float t0x = lf_mul(lf_sub(bounds.min.x, origins[0][l]), inv_dir[0][l]);
    float t1x = lf_mul(lf_sub(bounds.max.x, origins[0][l]), inv_dir[0][l]);
    float tminx = sse_min(t0x, t1x), tmaxx = sse_max(t0x, t1x);

    float t0y = lf_mul(lf_sub(bounds.min.y, origins[1][l]), inv_dir[1][l]);
    float t1y = lf_mul(lf_sub(bounds.max.y, origins[1][l]), inv_dir[1][l]);
    float tminy = sse_min(t0y, t1y), tmaxy = sse_max(t0y, t1y);

    float t0z = lf_mul(lf_sub(bounds.min.z, origins[2][l]), inv_dir[2][l]);
    float t1z = lf_mul(lf_sub(bounds.max.z, origins[2][l]), inv_dir[2][l]);
    float tminz = sse_min(t0z, t1z), tmaxz = sse_max(t0z, t1z);

    float t_enter = sse_max(sse_max(tminx, tminy), tminz);
    float t_exit = sse_min(sse_min(tmaxx, tmaxy), tmaxz);

    bool cond1 = t_enter <= t_exit;
    bool cond2 = t_exit >= 0.0f;
    bool cond3 = t_enter <= lambda_max[l];
    if (cond1 && cond2 && cond3) mask |= 1u << l;
  }
  return mask;
}

}  // namespace lfsim
