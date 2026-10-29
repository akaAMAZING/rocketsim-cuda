// V3: bullet/collision/dispatch/internal_edge_utility.rs (adjust_internal_edge_contacts and its helpers), with the
// TriangleInfoMap constants (triangle_info_map.rs:46-52). Runs inside the contact-added callback, on a manifold point just
// stored in a manifold, for contacts against a triangle mesh (the ball's and the car's, through the same function).
#pragma once
#include "sim/narrowphase.cuh"
#include "sim/triangle_shape.cuh"

namespace lfsim {

// triangle_info_map.rs:7-14 TriInfoFlag
constexpr uint32_t TIF_V0V1_CONVEX = 1u, TIF_V1V2_CONVEX = 1u << 1, TIF_V2V0_CONVEX = 1u << 2;
constexpr uint32_t TIF_V0V1_SWAP_NORMAL_B = 1u << 3, TIF_V1V2_SWAP_NORMAL_B = 1u << 4, TIF_V2V0_SWAP_NORMAL_B = 1u << 5;
// triangle_info_map.rs:46-52
constexpr float TIM_CONVEX_EPSILON = 0.0f;
constexpr float TIM_EDGE_DISTANCE_THRESHOLD = 0.1f;
constexpr float TIM_MAX_EDGE_ANGLE_THRESHOLD = LF_TAU;

// V3: internal_edge_utility.rs:24 get_angle
LF_HDI float ieu_get_angle(Vec3A edge_a, Vec3A normal_a, Vec3A normal_b) {
  return lf::atan2(dot(normal_b, edge_a), dot(normal_b, normal_a));
}

// V3: internal_edge_utility.rs:176 EDGE_T_MIN / EDGE_T_MAX (exact bits of 1e-05 and 0.99999)
LF_HDI float ieu_edge_t_min() { return lf_from_bits(0x3727c5acu); }
LF_HDI float ieu_edge_t_max() { return lf_from_bits(0x3f7fff58u); }

// V3: internal_edge_utility.rs:188 nearst_point_in_line_segment
LF_HDI Vec3A ieu_nearest_point_in_line_segment(Vec3A point, Vec3A line0, Vec3A line1) {
  Vec3A line_delta = line1 - line0;

  if (length_squared(line_delta) < lf_mul(LF_EPSILON, LF_EPSILON)) {
    return line0;
  }
  float delta = lf_div(dot(point - line0, line_delta), dot(line_delta, line_delta));
  return line0 + line_delta * rs_clamp(delta, ieu_edge_t_min(), ieu_edge_t_max());
}

// V3: internal_edge_utility.rs:199 clamp_normal. Returns false for None.
LF_HDI bool ieu_clamp_normal(Vec3A edge, Vec3A tri_normal_org, Vec3A local_contact_normal_on_b, float corrected_edge_angle,
                             Vec3A* out) {
  Vec3A edge_cross = normalize(cross(edge, tri_normal_org));
  float cur_angle = ieu_get_angle(edge_cross, tri_normal_org, local_contact_normal_on_b);

  if ((corrected_edge_angle < 0.0f && cur_angle < corrected_edge_angle) ||
      (corrected_edge_angle >= 0.0f && cur_angle > corrected_edge_angle)) {
    float diff_angle = lf_sub(corrected_edge_angle, cur_angle);
    // QuatExt::from_axis_angle_simd (quat_ext.rs:9)
    float s, cc;
    lf::sin_cos(lf_mul(diff_angle, 0.5f), &s, &cc);
    Vec3A v = normalize(edge) * splat3(s);
    Quat rotation = quat_from_xyzw(v.x, v.y, v.z, cc);
    *out = mul_vec3a(rotation, local_contact_normal_on_b);
    return true;
  }
  return false;
}

enum BestEdge : int { BEST_NONE = 0, BEST_X = 1, BEST_Y = 2, BEST_Z = 3 };

// V3: internal_edge_utility.rs:227 select_best_edge
LF_HDI void ieu_select_best_edge(int edge, float distance, float angle, uint32_t flags, int* best_edge, float* dist_to_best_edge) {
  bool should_select;
  if (edge == BEST_X) {
    should_select = distance < *dist_to_best_edge;
  } else if (edge == BEST_Y) {
    should_select = distance < *dist_to_best_edge ||
                    (distance == *dist_to_best_edge && angle != 0.0f && (flags & TIF_V1V2_CONVEX) == 0);
  } else {
    should_select = distance < *dist_to_best_edge || (distance == *dist_to_best_edge && angle != 0.0f);
  }

  if (should_select) {
    *best_edge = edge;
    if (edge == BEST_X || edge == BEST_Y) *dist_to_best_edge = distance;
  }
}

// One edge of adjust_internal_edge_contacts (the three blocks of internal_edge_utility.rs:297-335, 337-381, 383-427 differ in
// the edge, its angle and flags, and in whether the local normal is recomputed after the back-facing test: the X block keeps
// the one taken at the top, the Y and Z blocks recompute it from the point's current normal).
LF_HDI void ieu_edge_block(ManifoldPoint* cp, const Affine3A& mesh_trans, const Tri& tri, int edge_no, float angle, uint32_t flags,
                           uint32_t convex_flag, uint32_t swap_flag, Vec3A local_contact_normal_on_b, bool recompute_local_normal,
                           bool* concave_edge_hit) {
  if (angle == 0.0f) {
    *concave_edge_hit = true;
    return;
  }
  bool is_edge_convex = (flags & convex_flag) != 0;
  float swap_factor = lf_sub(lf_mul(is_edge_convex ? 1.0f : 0.0f, 2.0f), 1.0f);

  Vec3A edge = -tri_edge(tri, edge_no);
  Vec3A n_a = splat3(swap_factor) * tri.normal;
  float s, cc;
  lf::sin_cos(lf_mul(angle, 0.5f), &s, &cc);
  Vec3A axis_s = normalize(edge) * splat3(s);
  Quat orn = quat_from_xyzw(axis_s.x, axis_s.y, axis_s.z, cc);
  Vec3A computed_normal_b = mul_vec3a(orn, tri.normal);
  if ((flags & swap_flag) != 0) computed_normal_b *= -1.0f;
  Vec3A n_b = splat3(swap_factor) * computed_normal_b;

  float n_dot_a = dot(local_contact_normal_on_b, n_a);
  float n_dot_b = dot(local_contact_normal_on_b, n_b);
  bool back_facing_normal = n_dot_a < TIM_CONVEX_EPSILON && n_dot_b < TIM_CONVEX_EPSILON;

  if (back_facing_normal) {
    *concave_edge_hit = true;
    return;
  }
  Vec3A local_normal = local_contact_normal_on_b;
  if (recompute_local_normal) local_normal = mul_transpose_vec3a(mesh_trans.matrix3, cp->normal_world_on_b);

  Vec3A clamped_local_normal;
  if (ieu_clamp_normal(edge, splat3(swap_factor) * tri.normal, local_normal, angle, &clamped_local_normal) &&
      dot(clamped_local_normal, tri.normal) > 0.0f) {
    Vec3A new_normal = mesh_trans.matrix3 * clamped_local_normal;
    cp->normal_world_on_b = new_normal;
    cp->pos_world_on_b = cp->pos_world_on_a - new_normal * cp->distance_1;
    cp->local_point_b = affine_inv_x_form(mesh_trans, cp->pos_world_on_b);
  }
}

// V3: internal_edge_utility.rs:259 adjust_internal_edge_contacts(cp, tri_mesh_col_obj, idx): `mesh_body` is the mesh's body
// index, `idx` the mesh-local triangle index.
LF_HDI void adjust_internal_edge_contacts(const NpCtx& c, ManifoldPoint* cp, uint32_t mesh_body, uint32_t idx) {
  const MeshInfo& mesh = c.sd->meshes[mesh_body];
  const TriInfo& info = c.sd->infos[mesh.tri_first + idx];
  const Tri& tri = c.sd->tris[mesh.tri_first + idx];
  Affine3A mesh_trans = c.sd->bodies[mesh_body].pose;

  Vec3A nearest = ieu_nearest_point_in_line_segment(cp->local_point_b, tri.p0, tri.p1);
  Vec3A contact = cp->local_point_b;

  Vec3A local_contact_normal_on_b = mul_transpose_vec3a(mesh_trans.matrix3, cp->normal_world_on_b);

  int best_edge = BEST_NONE;
  float dists_x = LF_F32_MAX, dists_y = LF_F32_MAX, dists_z = LF_F32_MAX;
  float dist_to_best_edge = LF_F32_MAX;

  if (lf_abs(info.edge_v0_v1_angle) < TIM_MAX_EDGE_ANGLE_THRESHOLD) {
    dists_x = length(contact - nearest);
    ieu_select_best_edge(BEST_X, dists_x, info.edge_v0_v1_angle, info.flags, &best_edge, &dist_to_best_edge);
  }

  if (lf_abs(info.edge_v1_v2_angle) < TIM_MAX_EDGE_ANGLE_THRESHOLD) {
    Vec3A nearest_y = ieu_nearest_point_in_line_segment(cp->local_point_b, tri.p1, tri.p2);
    dists_y = length(contact - nearest_y);
    ieu_select_best_edge(BEST_Y, dists_y, info.edge_v1_v2_angle, info.flags, &best_edge, &dist_to_best_edge);
  }

  if (lf_abs(info.edge_v2_v0_angle) < TIM_MAX_EDGE_ANGLE_THRESHOLD) {
    Vec3A nearest_z = ieu_nearest_point_in_line_segment(cp->local_point_b, tri.p2, tri.p0);
    dists_z = length(contact - nearest_z);
    ieu_select_best_edge(BEST_Z, dists_z, info.edge_v2_v0_angle, info.flags, &best_edge, &dist_to_best_edge);
  }

  bool processed_edge = false;
  bool concave_edge_hit = false;

  if (lf_abs(info.edge_v0_v1_angle) < TIM_MAX_EDGE_ANGLE_THRESHOLD && dists_x < TIM_EDGE_DISTANCE_THRESHOLD &&
      (best_edge == BEST_X || (dists_x < LF_EPSILON && info.edge_v0_v1_angle != 0.0f))) {
    processed_edge = true;
    ieu_edge_block(cp, mesh_trans, tri, 0, info.edge_v0_v1_angle, info.flags, TIF_V0V1_CONVEX, TIF_V0V1_SWAP_NORMAL_B,
                   local_contact_normal_on_b, false, &concave_edge_hit);
  }

  if (lf_abs(info.edge_v1_v2_angle) < TIM_MAX_EDGE_ANGLE_THRESHOLD && dists_y < TIM_EDGE_DISTANCE_THRESHOLD &&
      (best_edge == BEST_Y || (dists_y < LF_EPSILON && info.edge_v1_v2_angle != 0.0f))) {
    processed_edge = true;
    ieu_edge_block(cp, mesh_trans, tri, 1, info.edge_v1_v2_angle, info.flags, TIF_V1V2_CONVEX, TIF_V1V2_SWAP_NORMAL_B,
                   local_contact_normal_on_b, true, &concave_edge_hit);
  }

  if (lf_abs(info.edge_v2_v0_angle) < TIM_MAX_EDGE_ANGLE_THRESHOLD && dists_z < TIM_EDGE_DISTANCE_THRESHOLD &&
      (best_edge == BEST_Z || (dists_z < LF_EPSILON && info.edge_v2_v0_angle != 0.0f))) {
    processed_edge = true;
    ieu_edge_block(cp, mesh_trans, tri, 2, info.edge_v2_v0_angle, info.flags, TIF_V2V0_CONVEX, TIF_V2V0_SWAP_NORMAL_B,
                   local_contact_normal_on_b, true, &concave_edge_hit);
  }

  if (!processed_edge) return;
  if (!concave_edge_hit) return;

  float d = dot(tri.normal, local_contact_normal_on_b);
  if (d < 0.0f) return;

  cp->normal_world_on_b = mesh_trans.matrix3 * tri.normal;
  cp->pos_world_on_b = cp->pos_world_on_a - cp->normal_world_on_b * cp->distance_1;
  cp->local_point_b = affine_inv_x_form(mesh_trans, cp->pos_world_on_b);
}

}  // namespace lfsim
