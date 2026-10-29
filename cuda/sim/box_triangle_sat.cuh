// V3: bullet/collision/narrowphase/box_triangle_sat.rs (the car's box against one arena triangle, in the box-local frame).
// Mirrors V3's operation order, axis order and tie rules. Option<T> is `bool` plus an out parameter.
#pragma once
#include "sim/narrowphase.cuh"

namespace lfsim {

// The SAT is large: keep its big functions out of line so the device compiler (cicc) does not inline 47-pair arrays everywhere.
#if defined(__CUDACC__)
#define LF_HDN LF_HD __noinline__ inline
#elif defined(_MSC_VER)
#define LF_HDN __declspec(noinline) inline
#else
#define LF_HDN __attribute__((noinline)) inline
#endif

constexpr float BTS_TIE_EPS = 1e-4f;  // V3: box_triangle_sat.rs:9 TIE_EPS

struct BoxTriangleSatContact {  // V3: :18
  Vec3A point_on_b_world;
  float distance;
};

struct BtsPair {  // V3: CandidatePair :833
  Vec3A point_box, point_tri;
  float distance_sq;
};

// V3: AxisKind :302
enum BtsKindTag : int { BTS_NONE = 0, BTS_TRI_FACE = 1, BTS_BOX_FACE = 2, BTS_EDGE_EDGE = 3 };
struct BtsKind {
  int tag;
  int box_axis, tri_edge;
};

struct BtsSweep {  // V3: SweepResult :311
  bool has_sep;
  float best_gap;
  Vec3A best_sep_axis;
  float best_overlap;
  Vec3A best_pen_axis;
  Vec3A best_pen_raw;
  BtsKind pen_kind;
};

LF_HDI void bts_set(Vec3A* v, int i, float x) {
  if (i == 0) v->x = x;
  else if (i == 1) v->y = x;
  else v->z = x;
}

// V3: :331 orient_pen_axis
LF_HDI Vec3A bts_orient_pen_axis(Vec3A a, Vec3A to_box, Vec3A face_axis, bool has_face_axis) {
  float s = dot(a, to_box);
  if (s < 0.0f) return -a;
  if (s > 1e-12f) return a;
  if (has_face_axis && dot(a, face_axis) > 0.0f) return -a;
  return a;
}

// V3: :525 axis_gap_overlap -> (gap, overlap, side)
LF_HDN void bts_axis_gap_overlap(Vec3A a, const Vec3A q[3], Vec3A half, float* gap, float* overlap, float* side) {
  float r = lf_add(lf_add(lf_mul(half.x, lf_abs(a.x)), lf_mul(half.y, lf_abs(a.y))), lf_mul(half.z, lf_abs(a.z)));
  float d0 = dot(q[0], a);
  float d1 = dot(q[1], a);
  float d2 = dot(q[2], a);
  float min_t = rs_min(rs_min(d0, d1), d2);
  float max_t = rs_max(rs_max(d0, d1), d2);
  if (min_t > r) {
    *gap = lf_sub(min_t, r); *overlap = 0.0f; *side = 1.0f;
  } else if (max_t < lf_neg(r)) {
    *gap = lf_sub(lf_neg(r), max_t); *overlap = 0.0f; *side = -1.0f;
  } else {
    float exit_pos = lf_add(max_t, r);
    float exit_neg = lf_sub(r, min_t);
    *gap = 0.0f; *overlap = rs_max(rs_min(exit_pos, exit_neg), 0.0f); *side = 0.0f;
  }
}

// V3: :341 consider_axis; returns true when the axis proves rejection.
LF_HDN bool bts_consider_axis(BtsSweep* out, Vec3A a, BtsKind kind, const Vec3A q[3], Vec3A half, float margin,
                              float maximum_distance) {
  float gap, overlap, side;
  bts_axis_gap_overlap(a, q, half, &gap, &overlap, &side);
  if (gap > 0.0f) {
    if (lf_sub(gap, margin) >= maximum_distance) return true;
    if (!out->has_sep || gap > out->best_gap) {
      out->has_sep = true;
      out->best_gap = gap;
      out->best_sep_axis = side > 0.0f ? -a : a;
    }
  } else if (!out->has_sep && overlap < out->best_overlap) {
    out->best_overlap = overlap;
    out->pen_kind = kind;
    out->best_pen_raw = a;
  }
  return false;
}

// One box face of the sweep (V3: :373-427, three copies for x, y, z); `axis_unit` is Vec3A::X/Y/Z. Returns true on rejection.
LF_HDI bool bts_box_face(BtsSweep* out, float tmin, float tmax, float h, int axis, Vec3A axis_unit, float margin,
                         float maximum_distance) {
  if (tmin > h) {
    float gap = lf_sub(tmin, h);
    if (lf_sub(gap, margin) >= maximum_distance) return true;
    if (!out->has_sep || gap > out->best_gap) {
      out->has_sep = true;
      out->best_gap = gap;
      out->best_sep_axis = -axis_unit;
    }
  } else if (tmax < lf_neg(h)) {
    float gap = lf_sub(lf_neg(h), tmax);
    if (lf_sub(gap, margin) >= maximum_distance) return true;
    if (!out->has_sep || gap > out->best_gap) {
      out->has_sep = true;
      out->best_gap = gap;
      out->best_sep_axis = axis_unit;
    }
  } else if (!out->has_sep) {
    float overlap = rs_max(rs_min(lf_add(tmax, h), lf_sub(h, tmin)), 0.0f);
    if (overlap < out->best_overlap) {
      out->best_overlap = overlap;
      out->pen_kind = BtsKind{BTS_BOX_FACE, axis, 0};
      out->best_pen_raw = axis_unit;
    }
  }
  return false;
}

// V3: :359 sat_sweep. Returns false for V3's None.
LF_HDN bool bts_sat_sweep(Vec3A half, const Vec3A q[3], Vec3A tri_min, Vec3A tri_max, const Vec3A edges[3], Vec3A face_axis,
                          bool has_face_axis, Vec3A centroid, float margin, float maximum_distance, BtsSweep* out) {
  out->has_sep = false;
  out->best_gap = 0.0f;
  out->best_sep_axis = splat3(0.0f);
  out->best_overlap = LF_F32_MAX;
  out->best_pen_axis = splat3(0.0f);
  out->best_pen_raw = splat3(0.0f);
  out->pen_kind = BtsKind{BTS_NONE, 0, 0};
  Vec3A to_box = -centroid;
  if (bts_box_face(out, tri_min.x, tri_max.x, half.x, 0, vec3a(1.0f, 0.0f, 0.0f), margin, maximum_distance)) return false;
  if (bts_box_face(out, tri_min.y, tri_max.y, half.y, 1, vec3a(0.0f, 1.0f, 0.0f), margin, maximum_distance)) return false;
  if (bts_box_face(out, tri_min.z, tri_max.z, half.z, 2, vec3a(0.0f, 0.0f, 1.0f), margin, maximum_distance)) return false;
  if (has_face_axis && bts_consider_axis(out, face_axis, BtsKind{BTS_TRI_FACE, 0, 0}, q, half, margin, maximum_distance))
    return false;
  for (int bi = 0; bi < 3; ++bi) {
    Vec3A b_axis = bi == 0 ? vec3a(1.0f, 0.0f, 0.0f) : bi == 1 ? vec3a(0.0f, 1.0f, 0.0f) : vec3a(0.0f, 0.0f, 1.0f);
    for (int ei = 0; ei < 3; ++ei) {
      Vec3A c = cross(b_axis, edges[ei]);
      float l2 = length_squared(c);
      if (l2 < 1e-18f) continue;
      if (bts_consider_axis(out, c / lf_sqrt(l2), BtsKind{BTS_EDGE_EDGE, bi, ei}, q, half, margin, maximum_distance)) return false;
    }
  }
  if (!out->has_sep && out->pen_kind.tag != BTS_NONE)
    out->best_pen_axis = bts_orient_pen_axis(out->best_pen_raw, to_box, face_axis, has_face_axis);
  return true;
}

// V3: :554 tri_support_first_max
LF_HDI Vec3A bts_tri_support_first_max(const Vec3A q[3], Vec3A dir) {
  Vec3A best = q[0];
  float best_d = dot(q[0], dir);
  for (int i = 1; i < 3; ++i) {
    float d = dot(q[i], dir);
    if (d > best_d) { best_d = d; best = q[i]; }
  }
  return best;
}

// V3: :607 clip_halfspace
LF_HDN int bts_clip_halfspace(const Vec3A* input, int n_in, Vec3A* output, int axis, float bound, bool keep_above) {
  if (n_in == 0) return 0;
  int out = 0;
  Vec3A prev = input[n_in - 1];
  float pc = at(prev, axis);
  float prev_d = keep_above ? lf_sub(pc, bound) : lf_sub(bound, pc);
  for (int i = 0; i < n_in; ++i) {
    Vec3A v = input[i];
    float c = at(v, axis);
    float d = keep_above ? lf_sub(c, bound) : lf_sub(bound, c);
    if (d >= 0.0f) {
      if (prev_d < 0.0f) {
        float t = lf_div(prev_d, lf_sub(prev_d, d));
        output[out++] = prev + (v - prev) * t;
      }
      output[out++] = v;
    } else if (prev_d >= 0.0f) {
      float t = lf_div(prev_d, lf_sub(prev_d, d));
      output[out++] = prev + (v - prev) * t;
    }
    prev = v;
    prev_d = d;
  }
  return out;
}

// V3: :576 clipped_tri_box_centroid
LF_HDN bool bts_clipped_tri_box_centroid(Vec3A half, const Vec3A q[3], Vec3A* res) {
  Vec3A poly[9], scratch[9];
  for (int i = 0; i < 9; ++i) { poly[i] = splat3(0.0f); scratch[i] = splat3(0.0f); }
  poly[0] = q[0]; poly[1] = q[1]; poly[2] = q[2];
  int len = 3;
  for (int axis = 0; axis < 3; ++axis) {
    float h = at(half, axis);
    len = bts_clip_halfspace(poly, len, scratch, axis, lf_neg(h), true);
    if (len == 0) return false;
    len = bts_clip_halfspace(scratch, len, poly, axis, h, false);
    if (len == 0) return false;
  }
  Vec3A sum = splat3(0.0f);
  for (int i = 0; i < len; ++i) sum += poly[i];
  *res = sum / lf_i2f(len);
  return true;
}

// V3: :649
LF_HDI Vec3A bts_clamp_point_to_aabb(Vec3A p, Vec3A half) {
  return vec3a(rs_clamp(p.x, lf_neg(half.x), half.x), rs_clamp(p.y, lf_neg(half.y), half.y),
               rs_clamp(p.z, lf_neg(half.z), half.z));
}

// V3: :657
LF_HDI Vec3A bts_closest_point_on_segment(Vec3A p, Vec3A a, Vec3A b) {
  Vec3A ab = b - a;
  float denom = dot(ab, ab);
  if (denom < 1e-30f) return a;
  float t = rs_clamp(lf_div(dot(p - a, ab), denom), 0.0f, 1.0f);
  return a + ab * t;
}

// V3: :669 closest_point_on_degenerate_triangle
LF_HDI Vec3A bts_closest_point_on_degenerate_triangle(Vec3A p, const Vec3A q[3]) {
  Vec3A c0 = bts_closest_point_on_segment(p, q[0], q[1]);
  Vec3A c1 = bts_closest_point_on_segment(p, q[1], q[2]);
  Vec3A c2 = bts_closest_point_on_segment(p, q[2], q[0]);
  float d0 = length_squared(p - c0);
  float d1 = length_squared(p - c1);
  float d2 = length_squared(p - c2);
  if (d0 <= d1 && d0 <= d2) return c0;
  if (d1 <= d2) return c1;
  return c2;
}

// V3: :690 segment_segment_closest -> (c1, c2)
LF_HDN void bts_segment_segment_closest(Vec3A p1, Vec3A q1, Vec3A p2, Vec3A q2, Vec3A* o1, Vec3A* o2) {
  Vec3A d1 = q1 - p1;
  Vec3A d2 = q2 - p2;
  Vec3A r = p1 - p2;
  float a = dot(d1, d1);
  float e = dot(d2, d2);
  float f = dot(d2, r);
  const float eps = 1e-30f;
  float s, t;
  if (a <= eps && e <= eps) { *o1 = p1; *o2 = p2; return; }
  if (a <= eps) {
    s = 0.0f;
    t = rs_clamp(lf_div(f, e), 0.0f, 1.0f);
  } else {
    float c = dot(d1, r);
    if (e <= eps) {
      t = 0.0f;
      s = rs_clamp(lf_div(lf_neg(c), a), 0.0f, 1.0f);
    } else {
      float b = dot(d1, d2);
      if (lf_mul(b, b) > lf_mul(lf_mul(0.99900025f, a), e)) {
        float t0 = lf_div(dot(p2 - p1, d1), a);
        float t1 = lf_div(dot(q2 - p1, d1), a);
        float lo = rs_max(0.0f, rs_min(t0, t1));
        float hi = rs_min(1.0f, rs_max(t0, t1));
        if (lo <= hi) {
          float sm = lf_mul(lf_add(lo, hi), 0.5f);
          Vec3A c1 = p1 + d1 * sm;
          float tm = rs_clamp(lf_div(dot(c1 - p2, d2), e), 0.0f, 1.0f);
          *o1 = c1; *o2 = p2 + d2 * tm;
          return;
        }
      }
      float denom = lf_sub(lf_mul(a, e), lf_mul(b, b));
      s = denom > eps ? rs_clamp(lf_div(lf_sub(lf_mul(b, f), lf_mul(c, e)), denom), 0.0f, 1.0f) : 0.0f;
      t = lf_div(lf_add(lf_mul(b, s), f), e);
      if (t < 0.0f) {
        t = 0.0f;
        s = rs_clamp(lf_div(lf_neg(c), a), 0.0f, 1.0f);
      } else if (t > 1.0f) {
        t = 1.0f;
        s = rs_clamp(lf_div(lf_sub(b, c), a), 0.0f, 1.0f);
      }
    }
  }
  *o1 = p1 + d1 * s;
  *o2 = p2 + d2 * t;
}

// V3: :749 box_support_corner
LF_HDI Vec3A bts_box_support_corner(Vec3A half, Vec3A dir) {
  return vec3a(dir.x < 0.0f ? lf_neg(half.x) : half.x, dir.y < 0.0f ? lf_neg(half.y) : half.y,
               dir.z < 0.0f ? lf_neg(half.z) : half.z);
}

// V3: :762 point_in_triangle_eps
LF_HDI bool bts_point_in_triangle_eps(Vec3A p, const Vec3A q[3], Vec3A n) {
  const float EPS = 1e-6f;
  bool pos = false, neg = false;
  for (int i = 0; i < 3; ++i) {
    float s = dot(cross(q[(i + 1) % 3] - q[i], p - q[i]), n);
    if (s > EPS) pos = true;
    else if (s < lf_neg(EPS)) neg = true;
    if (pos && neg) return false;
  }
  return true;
}

// V3: :783 clip_segment_to_box
LF_HDI bool bts_clip_segment_to_box(Vec3A a, Vec3A b, Vec3A half, Vec3A* o0, Vec3A* o1) {
  float t0 = 0.0f, t1 = 1.0f;
  Vec3A d = b - a;
  for (int axis = 0; axis < 3; ++axis) {
    float h = at(half, axis);
    for (int k = 0; k < 2; ++k) {
      bool is_min = k == 0;
      float p = is_min ? at(d, axis) : lf_neg(at(d, axis));
      float qv = is_min ? lf_add(at(a, axis), h) : lf_sub(h, at(a, axis));
      if (lf_abs(p) < LF_EPSILON) {
        if (qv < 0.0f) return false;
      } else {
        float r = lf_div(qv, p);
        if (p < 0.0f) {
          if (r > t1) return false;
          if (r > t0) t0 = r;
        } else {
          if (r < t0) return false;
          if (r < t1) t1 = r;
        }
      }
    }
  }
  if (t0 > t1) return false;
  *o0 = a + d * t0;
  *o1 = a + d * t1;
  return true;
}

// V3: :735 pen_witness_by_kind
LF_HDN bool bts_pen_witness_by_kind(Vec3A half, const Vec3A q[3], BtsKind kind, Vec3A n_local, Vec3A* res) {
  switch (kind.tag) {
    case BTS_TRI_FACE: {
      float plane_dist = dot(n_local, q[0]);
      Vec3A corner = bts_box_support_corner(half, -n_local);
      Vec3A p = corner - n_local * lf_sub(dot(n_local, corner), plane_dist);
      if (bts_point_in_triangle_eps(p, q, n_local)) { *res = p; return true; }
      return false;
    }
    case BTS_BOX_FACE: {
      float deepest = lf_from_bits(0xFF800000u);  // f32::NEG_INFINITY
      for (int i = 0; i < 3; ++i) deepest = rs_max(deepest, dot(n_local, q[i]));
      Vec3A tied[3];
      int n_tied = 0;
      for (int i = 0; i < 3; ++i) {
        if (lf_abs(lf_sub(dot(n_local, q[i]), deepest)) <= BTS_TIE_EPS) tied[n_tied++] = q[i];
      }
      if (n_tied == 0) return false;
      if (n_tied == 1) { *res = tied[0]; return true; }
      if (n_tied == 2) {
        Vec3A a, b;
        if (!bts_clip_segment_to_box(tied[0], tied[1], half, &a, &b)) return false;
        *res = (a + b) * 0.5f;
        return true;
      }
      return bts_clipped_tri_box_centroid(half, q, res);
    }
    case BTS_EDGE_EDGE: {
      Vec3A s = signum(-n_local);
      Vec3A start = half * s;
      Vec3A end = half * s;
      bts_set(&start, kind.box_axis, at(half, kind.box_axis));
      bts_set(&end, kind.box_axis, lf_neg(at(half, kind.box_axis)));
      Vec3A c1, tri_pt;
      bts_segment_segment_closest(start, end, q[kind.tri_edge], q[(kind.tri_edge + 1) % 3], &c1, &tri_pt);
      *res = tri_pt;
      return true;
    }
    default: return false;
  }
}

// V3: :848 closest_point_on_triangle_precomp
LF_HDN Vec3A bts_closest_point_on_triangle_precomp(Vec3A p, const Vec3A q[3], Vec3A ab, Vec3A ac) {
  Vec3A ap = p - q[0];
  float d1 = dot(ab, ap);
  float d2 = dot(ac, ap);
  if (d1 <= 0.0f && d2 <= 0.0f) return q[0];
  Vec3A bp = p - q[1];
  float d3 = dot(ab, bp);
  float d4 = dot(ac, bp);
  if (d3 >= 0.0f && d4 <= d3) return q[1];
  float vc = lf_sub(lf_mul(d1, d4), lf_mul(d3, d2));
  if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
    float v = lf_div(d1, lf_sub(d1, d3));
    return q[0] + ab * v;
  }
  Vec3A cp = p - q[2];
  float d5 = dot(ab, cp);
  float d6 = dot(ac, cp);
  if (d6 >= 0.0f && d5 <= d6) return q[2];
  float vb = lf_sub(lf_mul(d5, d2), lf_mul(d1, d6));
  if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
    float w = lf_div(d2, lf_sub(d2, d6));
    return q[0] + ac * w;
  }
  float va = lf_sub(lf_mul(d3, d6), lf_mul(d5, d4));
  if (va <= 0.0f && lf_sub(d4, d3) >= 0.0f && lf_sub(d5, d6) >= 0.0f) {
    float w = lf_div(lf_sub(d4, d3), lf_add(lf_sub(d4, d3), lf_sub(d5, d6)));
    return q[1] + (q[2] - q[1]) * w;
  }
  float denom = lf_add(lf_add(va, vb), vc);
  if (lf_abs(denom) < 1e-24f) return bts_closest_point_on_degenerate_triangle(p, q);
  float v = lf_div(vb, denom);
  float w = lf_div(vc, denom);
  return q[0] + ab * v + ac * w;
}

// V3: :899 tri_edge_vs_box_edge_precomp -> (point_tri, point_box)
LF_HDN void bts_tri_edge_vs_box_edge_precomp(Vec3A tri_p, Vec3A tri_d, float tri_a, Vec3A box_p0, Vec3A box_p1, Vec3A* o_tri,
                                             Vec3A* o_box) {
  Vec3A d2 = box_p1 - box_p0;
  Vec3A r = tri_p - box_p0;
  float e = dot(d2, d2);
  float f = dot(d2, r);
  const float eps = 1e-30f;
  if (tri_a <= eps) {
    if (e <= eps) { *o_tri = tri_p; *o_box = box_p0; return; }
    float t = rs_clamp(lf_div(f, e), 0.0f, 1.0f);
    *o_tri = tri_p; *o_box = box_p0 + d2 * t;
    return;
  }
  if (e <= eps) {
    float c = dot(tri_d, r);
    float s = rs_clamp(lf_div(lf_neg(c), tri_a), 0.0f, 1.0f);
    *o_tri = tri_p + tri_d * s; *o_box = box_p0;
    return;
  }
  float c = dot(tri_d, r);
  float b = dot(tri_d, d2);
  if (lf_mul(b, b) > lf_mul(lf_mul(0.99900025f, tri_a), e)) {
    float t0 = lf_div(dot(box_p0 - tri_p, tri_d), tri_a);
    float t1 = lf_div(dot(box_p1 - tri_p, tri_d), tri_a);
    float lo = rs_max(0.0f, rs_min(t0, t1));
    float hi = rs_min(1.0f, rs_max(t0, t1));
    if (lo <= hi) {
      float sm = lf_mul(lf_add(lo, hi), 0.5f);
      Vec3A c1 = tri_p + tri_d * sm;
      float tm = rs_clamp(lf_div(dot(c1 - box_p0, d2), e), 0.0f, 1.0f);
      *o_tri = c1; *o_box = box_p0 + d2 * tm;
      return;
    }
  }
  float denom = lf_sub(lf_mul(tri_a, e), lf_mul(b, b));
  float s = denom > eps ? rs_clamp(lf_div(lf_sub(lf_mul(b, f), lf_mul(c, e)), denom), 0.0f, 1.0f) : 0.0f;
  float t = lf_div(lf_add(lf_mul(b, s), f), e);
  if (t < 0.0f) {
    t = 0.0f;
    s = rs_clamp(lf_div(lf_neg(c), tri_a), 0.0f, 1.0f);
  } else if (t > 1.0f) {
    t = 1.0f;
    s = rs_clamp(lf_div(lf_sub(b, c), tri_a), 0.0f, 1.0f);
  }
  *o_tri = tri_p + tri_d * s;
  *o_box = box_p0 + d2 * t;
}

constexpr int BTS_NPAIRS = 47;

// V3: :960 enum_aabb_triangle_pairs: 3 clamped vertices, 8 box corners vs triangle, 36 triangle-edge vs box-edge pairs.
LF_HDN void bts_enum_pairs(Vec3A half, const Vec3A q[3], const Vec3A edges[3], BtsPair* pairs) {
  int idx = 0;
  for (int i = 0; i < 3; ++i) {
    Vec3A v = q[i];
    Vec3A point_box = bts_clamp_point_to_aabb(v, half);
    pairs[idx++] = BtsPair{point_box, v, length_squared(point_box - v)};
  }
  Vec3A ab = edges[0];
  Vec3A ac = q[2] - q[0];
  for (int ix = 0; ix < 2; ++ix) {
    float sx = ix == 0 ? -1.0f : 1.0f;
    for (int iy = 0; iy < 2; ++iy) {
      float sy = iy == 0 ? -1.0f : 1.0f;
      for (int iz = 0; iz < 2; ++iz) {
        float sz = iz == 0 ? -1.0f : 1.0f;
        Vec3A corner = vec3a(lf_mul(sx, half.x), lf_mul(sy, half.y), lf_mul(sz, half.z));
        Vec3A point_tri = bts_closest_point_on_triangle_precomp(corner, q, ab, ac);
        pairs[idx++] = BtsPair{corner, point_tri, length_squared(corner - point_tri)};
      }
    }
  }
  Vec3A c[8] = {vec3a(lf_neg(half.x), lf_neg(half.y), lf_neg(half.z)), vec3a(half.x, lf_neg(half.y), lf_neg(half.z)),
                vec3a(half.x, half.y, lf_neg(half.z)),                 vec3a(lf_neg(half.x), half.y, lf_neg(half.z)),
                vec3a(lf_neg(half.x), lf_neg(half.y), half.z),         vec3a(half.x, lf_neg(half.y), half.z),
                vec3a(half.x, half.y, half.z),                         vec3a(lf_neg(half.x), half.y, half.z)};
  const int be[12][2] = {{0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5}, {5, 6}, {6, 7}, {7, 4}, {0, 4}, {1, 5}, {2, 6}, {3, 7}};
  for (int ti = 0; ti < 3; ++ti) {
    Vec3A tri_p = q[ti];
    Vec3A tri_d = edges[ti];
    float tri_a = dot(tri_d, tri_d);
    for (int k = 0; k < 12; ++k) {
      Vec3A point_tri, point_box;
      bts_tri_edge_vs_box_edge_precomp(tri_p, tri_d, tri_a, c[be[k][0]], c[be[k][1]], &point_tri, &point_box);
      pairs[idx++] = BtsPair{point_box, point_tri, length_squared(point_box - point_tri)};
    }
  }
}

// V3: :1074 first_min_of_pairs (strict <: the first pair wins ties)
LF_HDI BtsPair bts_first_min_of_pairs(const BtsPair* pairs) {
  BtsPair best = pairs[0];
  float best_d2 = best.distance_sq;
  for (int i = 1; i < BTS_NPAIRS; ++i) {
    if (pairs[i].distance_sq < best_d2) {
      best_d2 = pairs[i].distance_sq;
      best = pairs[i];
    }
  }
  return best;
}

// V3: :1096 flat_tie_segment -> (tri-side endpoints of the tie interval), false for None
LF_HDN bool bts_flat_tie_segment(Vec3A half, const BtsPair* pairs, Vec3A* r0, Vec3A* r1) {
  float flat_link = lf_mul(0.5f, max_element(half));
  if (flat_link <= 0.0f || !lf_isfinite(flat_link)) return false;
  float flat_slope = lf_div(BTS_TIE_EPS, flat_link);

  float best_d2 = LF_F32_MAX;
  for (int i = 0; i < BTS_NPAIRS; ++i)
    if (pairs[i].distance_sq < best_d2) best_d2 = pairs[i].distance_sq;
  float best_d = lf_sqrt(best_d2);
  float flat_link2 = lf_mul(flat_link, flat_link);

  float tie_d = lf_add(best_d, BTS_TIE_EPS);
  float tie_d2 = lf_mul(tie_d, tie_d);
  BtsPair tied[BTS_NPAIRS];
  int n_tied = 0;
  for (int i = 0; i < BTS_NPAIRS; ++i)
    if (pairs[i].distance_sq <= tie_d2) tied[n_tied++] = pairs[i];
  if (n_tied == 0) return false;
  Vec3A mean_a = splat3(0.0f), mean_b = splat3(0.0f);
  for (int i = 0; i < n_tied; ++i) {
    mean_a += tied[i].point_box;
    mean_b += tied[i].point_tri;
  }
  float inv = lf_div(1.0f, lf_i2f(n_tied));
  mean_a *= inv;
  mean_b *= inv;
  float d0 = length(mean_a - mean_b);

  bool has_flat = false;
  BtsPair flat = {};
  float best_slope = flat_slope;
  for (int i = 0; i < BTS_NPAIRS; ++i) {
    const BtsPair& pair = pairs[i];
    float sep2 = length_squared(pair.point_tri - mean_b);
    if (sep2 <= flat_link2) continue;
    float sep = lf_sqrt(sep2);
    float slope = lf_div(lf_sub(lf_sqrt(pair.distance_sq), d0), sep);
    if (slope <= best_slope) {
      Vec3A mid_a = (mean_a + pair.point_box) * 0.5f;
      Vec3A mid_b = (mean_b + pair.point_tri) * 0.5f;
      if (length(mid_a - mid_b) <= lf_add(d0, BTS_TIE_EPS)) {
        best_slope = slope;
        flat = pair;
        has_flat = true;
      }
    }
  }

  BtsPair end0 = tied[0], end1 = tied[0];
  float span2 = 0.0f;
  int n_cand = n_tied + (has_flat ? 1 : 0);  // consider(): tied in order, then the flat partner
  for (int ci = 0; ci < n_cand; ++ci) {
    BtsPair cand = ci < n_tied ? tied[ci] : flat;
    for (int pi = 0; pi < n_cand; ++pi) {  // `tied.iter().chain(flat.iter())`
      const BtsPair& pair = pi < n_tied ? tied[pi] : flat;
      float s2 = length_squared(cand.point_tri - pair.point_tri);
      if (s2 > span2) {
        span2 = s2;
        end0 = cand;
        end1 = pair;
      }
    }
  }
  if (span2 <= flat_link2) return false;
  Vec3A mid_a = (end0.point_box + end1.point_box) * 0.5f;
  Vec3A mid_b = (end0.point_tri + end1.point_tri) * 0.5f;
  if (length(mid_a - mid_b) > lf_add(best_d, BTS_TIE_EPS)) return false;
  *r0 = end0.point_tri;
  *r1 = end1.point_tri;
  return true;
}

// V3: :52 box_triangle_sat. Returns V3's Some(contact) as true.
LF_HDN bool box_triangle_sat(const Affine3A& box_trans, Vec3A half_unmargined, float margin, const Vec3A q[3],
                             float maximum_distance, BoxTriangleSatContact* contact) {
  Vec3A tri_min = min(min(q[0], q[1]), q[2]);
  Vec3A tri_max = max(max(q[0], q[1]), q[2]);
  const Vec3A& hu = half_unmargined;
  // pre-screen, sweep order X, Y, Z
  for (int ax = 0; ax < 3; ++ax) {
    float tmin = at(tri_min, ax), tmax = at(tri_max, ax), h = at(hu, ax);
    if (tmin > h) {
      if (lf_sub(lf_sub(tmin, h), margin) >= maximum_distance) return false;
    } else if (tmax < lf_neg(h) && lf_sub(lf_sub(lf_neg(h), tmax), margin) >= maximum_distance) {
      return false;
    }
  }

  Vec3A e0 = q[1] - q[0];
  Vec3A e1 = q[2] - q[1];
  Vec3A e2 = q[0] - q[2];

  Vec3A face_cross = cross(e0, -e2);
  float face_len2 = length_squared(face_cross);
  bool has_face_axis = face_len2 > 1e-20f;
  Vec3A face_axis = has_face_axis ? face_cross / lf_sqrt(face_len2) : splat3(0.0f);

  Vec3A centroid = (q[0] + q[1] + q[2]) * 0.33333334f;
  Vec3A edges[3] = {e0, e1, e2};

  BtsSweep sweep;
  if (!bts_sat_sweep(hu, q, tri_min, tri_max, edges, face_axis, has_face_axis, centroid, margin, maximum_distance, &sweep))
    return false;

  if (sweep.has_sep) {
    Vec3A n_world = normalize_or_zero(box_trans.matrix3 * sweep.best_sep_axis);
    if (length_squared(n_world) < 0.5f) return false;
    BtsPair pairs[BTS_NPAIRS];
    bts_enum_pairs(hu, q, edges, pairs);
    BtsPair first = bts_first_min_of_pairs(pairs);
    Vec3A pa_first = first.point_box, pb_first = first.point_tri;
    if (length_squared(pa_first - pb_first) > 1e-24f) {
      float pair_len = length(pa_first - pb_first);
      Vec3A pb_local = pb_first;
      bool snapped = false;
      Vec3A s0, s1;
      if (lf_sub(pair_len, margin) < 0.0f && bts_flat_tie_segment(hu, pairs, &s0, &s1)) {
        pb_local = bts_closest_point_on_segment(centroid, s0, s1);
        snapped = true;
      }
      float distance = snapped ? lf_sub(length(bts_clamp_point_to_aabb(pb_local, hu) - pb_local), margin)
                               : lf_sub(pair_len, margin);
      if (distance >= maximum_distance) return false;
      contact->point_on_b_world = transform_point3a(box_trans, pb_local);
      contact->distance = distance;
      return true;
    }
    Vec3A pb_local = bts_tri_support_first_max(q, sweep.best_sep_axis);
    contact->point_on_b_world = transform_point3a(box_trans, pb_local);
    contact->distance = lf_sub(sweep.best_gap, margin);
    return true;
  }

  if (sweep.best_overlap == LF_F32_MAX) return false;
  Vec3A n_local;
  if (length_squared(sweep.best_pen_axis) > 0.5f) {
    n_local = sweep.best_pen_axis;
  } else if (has_face_axis) {
    float s = dot(face_axis, -centroid);
    n_local = s < 0.0f ? -face_axis : face_axis;
  } else {
    return false;
  }
  float distance = lf_sub(lf_neg(sweep.best_overlap), margin);
  if (distance >= maximum_distance) return false;
  Vec3A pb_local;
  if (!bts_pen_witness_by_kind(hu, q, sweep.pen_kind, n_local, &pb_local)) {
    if (!bts_clipped_tri_box_centroid(hu, q, &pb_local)) pb_local = bts_tri_support_first_max(q, n_local);
  }
  Vec3A n_world = normalize_or_zero(box_trans.matrix3 * n_local);
  if (length_squared(n_world) < 0.5f) return false;
  contact->point_on_b_world = transform_point3a(box_trans, pb_local);
  contact->distance = distance;
  return true;
}

}  // namespace lfsim
