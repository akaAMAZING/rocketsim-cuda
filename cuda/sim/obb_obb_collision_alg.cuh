// V3: bullet/collision/dispatch/obb_obb_collision_alg.rs and box_box_detector.rs. GJK (convex_convex_collision_alg.rs) is
// not ported: Soccar 1v1 has no ConvexHull shape, rule 5).
#pragma once
#include "sim/narrowphase.cuh"
#include "sim/persistent_manifold.cuh"
#include "sim/consts.cuh"

namespace lfsim {

struct BbObb {  // V3: linear_math/obb.rs:3
  Vec3A center;
  Mat3A axis;
  Vec3A extent;
};
struct BbHit {  // V3: box_box_detector.rs:178
  float depth;
  Vec3A normal;
  uint32_t axis_idx;
};
struct BbPoly {  // ArrayVec<[f32; 2], 16>
  float p[16][2];
  uint32_t n;
};

// V3: box_box_detector.rs:20
LF_HDI float bb_line_closest_approach(Vec3A pa, Vec3A ua, Vec3A pb, Vec3A ub) {
  Vec3A p = pb - pa;
  float uaub = dot(ua, ub);
  float q1 = dot(ua, p);
  float q2 = lf_neg(dot(ub, p));
  float d = lf_sub(1.0f, lf_mul(uaub, uaub));
  if (d <= 0.0001f) return 0.0f;
  return lf_div(lf_add(lf_mul(uaub, q1), q2), d);
}

// V3: box_box_detector.rs:38 intersect_rect_quad2
LF_HDI BbPoly bb_intersect_rect_quad2(const float h[2], const float poly[4][2]) {
  BbPoly q;
  q.n = 4;
  for (int i = 0; i < 4; ++i) {
    q.p[i][0] = poly[i][0];
    q.p[i][1] = poly[i][1];
  }
  for (int dir = 0; dir < 2; ++dir) {
    for (int si = 0; si < 2; ++si) {
      float sign = si == 0 ? -1.0f : 1.0f;
      BbPoly r;
      r.n = 0;
      float clip_val = lf_mul(sign, h[dir]);
      uint32_t len = q.n;
      for (uint32_t i = 0; i < len; ++i) {
        const float* cur = q.p[i];
        const float* next = q.p[(i + 1) % len];
        float cur_val = cur[dir];
        float next_val = next[dir];
        bool inside_cur = lf_mul(sign, cur_val) < h[dir];
        if (inside_cur) {
          r.p[r.n][0] = cur[0];
          r.p[r.n][1] = cur[1];
          r.n++;
        }
        bool inside_next = lf_mul(sign, next_val) < h[dir];
        if (inside_cur != inside_next) {
          float p = lf_add(cur[1 - dir],
                           lf_mul(lf_div(lf_sub(next[1 - dir], cur[1 - dir]), lf_sub(next_val, cur_val)), lf_sub(clip_val, cur_val)));
          float p1 = p, p2 = clip_val;
          if (dir == 0) {
            float t = p1;
            p1 = p2;
            p2 = t;
          }
          r.p[r.n][0] = p1;
          r.p[r.n][1] = p2;
          r.n++;
        }
      }
      q = r;
    }
  }
  return q;
}

// V3: box_box_detector.rs:88 cull_points2: `res` gets m indices.
LF_HDI void bb_cull_points2(const float p[][2], uint32_t n, uint32_t i0, uint32_t m, uint32_t* res) {
  float cx, cy;
  if (n == 1) {
    cx = p[0][0];
    cy = p[0][1];
  } else if (n == 2) {
    cx = lf_mul(lf_add(p[0][0], p[1][0]), 0.5f);
    cy = lf_mul(lf_add(p[0][1], p[1][1]), 0.5f);
  } else {
    float a_sum = 0.0f, cx_sum = 0.0f, cy_sum = 0.0f;
    for (uint32_t i = 0; i + 1 < n; ++i) {
      float x0 = p[i][0], y0 = p[i][1];
      float x1 = p[i + 1][0], y1 = p[i + 1][1];
      float q = lf_sub(lf_mul(x0, y1), lf_mul(x1, y0));
      a_sum = lf_add(a_sum, q);
      cx_sum = lf_add(cx_sum, lf_mul(q, lf_add(x0, x1)));
      cy_sum = lf_add(cy_sum, lf_mul(q, lf_add(y0, y1)));
    }
    float x_last = p[n - 1][0], y_last = p[n - 1][1];
    float x0 = p[0][0], y0 = p[0][1];
    float q_last = lf_sub(lf_mul(x_last, y0), lf_mul(x0, y_last));
    float area = lf_add(a_sum, q_last);
    float inv = lf_abs(area) > LF_EPSILON ? lf_div(1.0f, lf_mul(3.0f, area)) : 1e18f;
    cx = lf_mul(inv, lf_add(cx_sum, lf_mul(q_last, lf_add(x_last, x0))));
    cy = lf_mul(inv, lf_add(cy_sum, lf_mul(q_last, lf_add(y_last, y0))));
  }

  float angles[8];
  for (uint32_t i = 0; i < n; ++i) {
    float dx = lf_sub(p[i][0], cx);
    float dy = lf_sub(p[i][1], cy);
    angles[i] = lf::atan2(dy, dx);
  }

  bool avail[8] = {false, false, false, false, false, false, false, false};
  for (uint32_t i = 0; i < n; ++i) avail[i] = true;
  avail[i0] = false;
  res[0] = i0;

  for (uint32_t j = 1; j < m; ++j) {
    float target = lf_add(lf_mul(lf_u2f(j), lf_div(LF_TAU, lf_u2f(m))), angles[i0]);
    if (target > c::PI_F) target = lf_sub(target, LF_TAU);

    uint32_t best_idx = i0;
    float best_diff = LF_F32_MAX;
    for (uint32_t i = 0; i < n; ++i) {
      if (avail[i]) {
        float diff = lf_abs(lf_sub(angles[i], target));
        if (diff > c::PI_F) diff = lf_sub(LF_TAU, diff);
        if (diff < best_diff) {
          best_diff = diff;
          best_idx = i;
        }
      }
    }
    avail[best_idx] = false;
    res[j] = best_idx;
  }
}

// V3: box_box_detector.rs:458 box_box_sat; the two `tst` closures are the two methods.
struct BbSat {
  float s, s2;
  bool has_normal_r;
  Vec3A normal_r, normal_c;
  bool invert_normal;
  uint32_t code;

  LF_HDI bool tst_face(float expr1, float expr2, Vec3A n, uint32_t cc) {  // :493
    s2 = lf_sub(lf_abs(expr1), expr2);
    if (s2 > 0.0f) return false;
    if (s2 > s) {
      s = s2;
      has_normal_r = true;
      normal_r = n;
      invert_normal = expr1 < 0.0f;
      code = cc;
    }
    return true;
  }
  LF_HDI bool tst_cross(float expr1, float expr2, Vec3A n, uint32_t cc) {  // :541
    s2 = lf_sub(lf_abs(expr1), expr2);
    if (s2 > LF_EPSILON) return false;
    float l = length(n);
    if (l > LF_EPSILON) {
      s2 = lf_div(s2, l);
      if (lf_mul(s2, 1.05f) > s) {
        s = s2;
        has_normal_r = false;
        normal_c = n / l;
        invert_normal = expr1 < 0.0f;
        code = cc;
      }
    }
    return true;
  }
};

// V3: box_box_detector.rs:458. Returns false for None.
LF_HDI bool bb_box_box_sat(const BbObb& obb1, const Mat3A& r1t, const BbObb& obb2, BbHit* hit) {
  Vec3A p = obb2.center - obb1.center;
  Vec3A pp = r1t * p;

  Mat3A rij = r1t * obb2.axis;
  Vec3A rij_cols[3] = {rij.x_axis, rij.y_axis, rij.z_axis};

  Mat3A q_cols_mat = mat3a_abs(rij);
  Vec3A q_cols[3] = {q_cols_mat.x_axis, q_cols_mat.y_axis, q_cols_mat.z_axis};
  Mat3A q_rows_mat = transpose(q_cols_mat);
  Vec3A q_rows[3] = {q_rows_mat.x_axis, q_rows_mat.y_axis, q_rows_mat.z_axis};

  BbSat st;
  st.s = -INFINITY;
  st.s2 = 0.0f;
  st.has_normal_r = false;
  st.normal_r = splat3(0.0f);
  st.normal_c = splat3(0.0f);
  st.invert_normal = false;
  st.code = 0;

  Vec3A a1[3] = {obb1.axis.x_axis, obb1.axis.y_axis, obb1.axis.z_axis};
  for (int i = 0; i < 3; ++i) {
    if (!st.tst_face(at(pp, i), lf_add(at(obb1.extent, i), dot(obb2.extent, q_rows[i])), a1[i], (uint32_t)(i + 1))) return false;
  }
  Vec3A a2[3] = {obb2.axis.x_axis, obb2.axis.y_axis, obb2.axis.z_axis};
  for (int i = 0; i < 3; ++i) {
    if (!st.tst_face(dot(a2[i], p), lf_add(dot(obb1.extent, q_cols[i]), at(obb2.extent, i)), a2[i], (uint32_t)(i + 4))) return false;
  }

  const Vec3A fudge2 = splat3(1e-5f);
  for (int i = 0; i < 3; ++i) q_cols[i] = q_cols[i] + fudge2;
  q_rows_mat = transpose(mat3a_from_cols(q_cols[0], q_cols[1], q_cols[2]));
  q_rows[0] = q_rows_mat.x_axis;
  q_rows[1] = q_rows_mat.y_axis;
  q_rows[2] = q_rows_mat.z_axis;

  const Vec3A unit[3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  for (int i = 0; i < 3; ++i) {
    int i1 = (i + 1) % 3, i2 = (i + 2) % 3;
    for (int j = 0; j < 3; ++j) {
      int j1 = (j + 1) % 3, j2 = (j + 2) % 3;
      Vec3A n = cross(unit[i], rij_cols[j]);
      float expr2 = lf_add(lf_add(lf_add(lf_mul(at(obb1.extent, i1), at(q_cols[j], i2)), lf_mul(at(obb1.extent, i2), at(q_cols[j], i1))),
                                  lf_mul(at(obb2.extent, j1), at(q_rows[i], j2))),
                           lf_mul(at(obb2.extent, j2), at(q_rows[i], j1)));
      if (!st.tst_cross(dot(pp, n), expr2, n, (uint32_t)(7 + i * 3 + j))) return false;
    }
  }

  if (st.code == 0) return false;

  Vec3A normal = st.has_normal_r ? st.normal_r : obb1.axis * st.normal_c;
  if (st.invert_normal) normal = -normal;

  hit->depth = lf_neg(st.s);
  hit->normal = normal;
  hit->axis_idx = st.code;
  return true;
}

// V3: box_box_detector.rs:226 compute_contact_points. col1/col2 are the compound bodies (manifold body0/body1).
LF_HDI void bb_compute_contact_points(NpCtx& c, BbObb obb1, Mat3A r1t, BbObb obb2, Mat3A r2t, const BbHit& hit, Manifold* manifold,
                                      uint32_t col1, uint32_t col2) {
  Vec3A r1t_axes[3] = {r1t.x_axis, r1t.y_axis, r1t.z_axis};
  Vec3A r2t_axes[3] = {r2t.x_axis, r2t.y_axis, r2t.z_axis};

  if (hit.axis_idx > 6) {
    Vec3A pa = obb1.center;
    for (int i = 0; i < 3; ++i) {
      float sign = lf_sub(lf_mul(2.0f, dot(hit.normal, r1t_axes[i]) > 0.0f ? 1.0f : 0.0f), 1.0f);
      pa = pa + lf_mul(sign, at(obb1.extent, i)) * r1t_axes[i];
    }
    Vec3A pb = obb2.center;
    for (int i = 0; i < 3; ++i) {
      float sign = lf_sub(1.0f, lf_mul(2.0f, dot(hit.normal, r2t_axes[i]) > 0.0f ? 1.0f : 0.0f));
      pb = pb + lf_mul(sign, at(obb2.extent, i)) * r2t_axes[i];
    }
    Vec3A ua = r1t_axes[(hit.axis_idx - 7) / 3];
    Vec3A ub = r2t_axes[(hit.axis_idx - 7) % 3];

    float beta = bb_line_closest_approach(pa, ua, pb, ub);
    manifold_add_contact_point(c, manifold, col1, col2, -hit.normal, pb + ub * beta, lf_neg(hit.depth), -1);
    return;
  }

  if (hit.axis_idx > 3) {
    BbObb to = obb1;
    obb1 = obb2;
    obb2 = to;
    Mat3A tm = r1t;
    r1t = r2t;
    r2t = tm;
    for (int i = 0; i < 3; ++i) {
      Vec3A t = r1t_axes[i];
      r1t_axes[i] = r2t_axes[i];
      r2t_axes[i] = t;
    }
  }

  Vec3A normal_2 = hit.axis_idx <= 3 ? hit.normal : -hit.normal;

  Vec3A nr = mul_transpose_vec3a(r2t, normal_2);
  Vec3A anr = glam::abs(nr);

  int lanr, a1, a2;
  if (anr.y > anr.x) {
    if (anr.y > anr.z) { lanr = 1; a1 = 0; a2 = 2; }
    else { lanr = 2; a1 = 0; a2 = 1; }
  } else if (anr.x > anr.z) {
    lanr = 0; a1 = 1; a2 = 2;
  } else {
    lanr = 2; a1 = 0; a2 = 1;
  }

  float inc = at(nr, lanr) < 0.0f ? at(obb2.extent, lanr) : lf_neg(at(obb2.extent, lanr));
  Vec3A center = (obb2.center - obb1.center) + inc * r2t_axes[lanr];

  uint32_t code_n = hit.axis_idx <= 3 ? hit.axis_idx - 1 : hit.axis_idx - 4;
  int code1, code2;
  if (code_n == 0) { code1 = 1; code2 = 2; }
  else if (code_n == 1) { code1 = 0; code2 = 2; }
  else { code1 = 0; code2 = 1; }

  float c1 = dot(center, r1t_axes[code1]);
  float c2 = dot(center, r1t_axes[code2]);

  Vec4 m = {dot(r1t_axes[code1], r2t_axes[a1]), dot(r1t_axes[code1], r2t_axes[a2]), dot(r1t_axes[code2], r2t_axes[a1]),
            dot(r1t_axes[code2], r2t_axes[a2])};
  Vec4 k = m * Vec4{at(obb2.extent, a1), at(obb2.extent, a2), at(obb2.extent, a1), at(obb2.extent, a2)};

  float quad[4][2] = {{lf_sub(lf_sub(c1, k.x), k.y), lf_sub(lf_sub(c2, k.z), k.w)},
                      {lf_add(lf_sub(c1, k.x), k.y), lf_add(lf_sub(c2, k.z), k.w)},
                      {lf_add(lf_add(c1, k.x), k.y), lf_add(lf_add(c2, k.z), k.w)},
                      {lf_sub(lf_add(c1, k.x), k.y), lf_sub(lf_add(c2, k.z), k.w)}};

  float rect[2] = {at(obb1.extent, code1), at(obb1.extent, code2)};

  BbPoly ret = bb_intersect_rect_quad2(rect, quad);
  if (ret.n == 0) return;

  float det1 = lf_div(1.0f, lf_sub(lf_mul(m.x, m.w), lf_mul(m.y, m.z)));
  m = m * splat4(det1);

  uint32_t cnum = 0;
  Vec3A point[16];
  float dep[16];
  for (uint32_t j = 0; j < ret.n; ++j) {
    float k1 = lf_sub(lf_mul(m.w, lf_sub(ret.p[j][0], c1)), lf_mul(m.y, lf_sub(ret.p[j][1], c2)));
    float k2 = lf_add(lf_mul(lf_neg(m.z), lf_sub(ret.p[j][0], c1)), lf_mul(m.x, lf_sub(ret.p[j][1], c2)));

    point[cnum] = (center + k1 * r2t_axes[a1]) + k2 * r2t_axes[a2];
    dep[cnum] = lf_sub(at(obb1.extent, (int)code_n), dot(normal_2, point[cnum]));

    if (dep[cnum] >= 0.0f) {
      ret.p[cnum][0] = ret.p[j][0];
      ret.p[cnum][1] = ret.p[j][1];
      cnum += 1;
    }
  }
  if (cnum == 0) return;

  uint32_t maxc = cnum < (uint32_t)MAX_POINTS ? cnum : (uint32_t)MAX_POINTS;  // MANIFOLD_CACHE_SIZE.clamp(1, cnum)

  if (cnum <= maxc) {
    if (hit.axis_idx < 4) {
      for (uint32_t i = 0; i < cnum; ++i)
        manifold_add_contact_point(c, manifold, col1, col2, -hit.normal, point[i] + obb1.center, lf_neg(dep[i]), -1);
    } else {
      for (uint32_t i = 0; i < cnum; ++i)
        manifold_add_contact_point(c, manifold, col1, col2, -hit.normal, (point[i] + obb1.center) - hit.normal * dep[i],
                                   lf_neg(dep[i]), -1);
    }
    return;
  }

  uint32_t i1 = 0;
  for (uint32_t i = 0; i < cnum; ++i)
    if (dep[i] > dep[i1]) i1 = i;

  uint32_t iret[MAX_POINTS];
  bb_cull_points2(ret.p, cnum, i1, maxc, iret);

  for (uint32_t t = 0; t < maxc; ++t) {
    uint32_t idx = iret[t];
    Vec3A pos_in_world = point[idx] + obb1.center;
    if (hit.axis_idx < 4) {
      manifold_add_contact_point(c, manifold, col1, col2, -hit.normal, pos_in_world, lf_neg(dep[idx]), -1);
    } else {
      manifold_add_contact_point(c, manifold, col1, col2, -hit.normal, pos_in_world - hit.normal * dep[idx], lf_neg(dep[idx]), -1);
    }
  }
}

// V3: box_box_detector.rs:193 get_closest_points + obb_obb_collision_alg.rs:11 process_collision(compound_a, compound_b), the
// pair in the dispatcher's order. Returns whether V3 leaves `*out = Some(manifold)`.
LF_HDI bool alg_obb_obb(NpCtx& c, uint32_t body_a, uint32_t body_b, Manifold* out) {
  Aabb aabb_0 = body_raw_aabb(c, body_a);
  Aabb aabb_1 = body_raw_aabb(c, body_b);
  if (!aabb_intersects(aabb_0, aabb_1)) return false;

  Affine3A transform_a = compound_child_world_trans(c, body_a);
  Affine3A transform_b = compound_child_world_trans(c, body_b);

  Mat3A axis_a = transform_a.matrix3;
  Mat3A axis_b = transform_b.matrix3;
  Mat3A axis_a_inv = transpose(axis_a);

  const BodyConst& bc1 = body_const(c, body_a);
  const BodyConst& bc2 = body_const(c, body_b);
  Vec3A side1 = add_s(bc1.box_half_extents, bc1.box_margin);
  Vec3A side2 = add_s(bc2.box_half_extents, bc2.box_margin);

  BbObb obb1 = {transform_a.translation, axis_a, side1};
  BbObb obb2 = {transform_b.translation, axis_b, side2};

  BbHit hit;
  if (!bb_box_box_sat(obb1, axis_a_inv, obb2, &hit)) return false;

  Manifold manifold = manifold_new(body_a, body_b);
  bb_compute_contact_points(c, obb1, axis_a, obb2, axis_b, hit, &manifold, body_a, body_b);

  if (manifold.point_count == 0) return false;
  manifold_refresh_contact_points(c, &manifold, body_a, body_b);
  *out = manifold;
  return true;
}

}  // namespace lfsim
