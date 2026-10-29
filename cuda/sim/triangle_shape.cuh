// V3: bullet/collision/shapes/triangle_shape.rs (the sphere-vs-triangle test the ball's mesh contacts use).
#pragma once
#include "sim/narrowphase.cuh"

namespace lfsim {

struct ContactInfo {  // triangle_shape.rs:5-9
  Vec3A result_normal, contact_point;
  float depth;
};

// V3: triangle_shape.rs:11
LF_HDI float segment_sqr_distance(Vec3A from, Vec3A to, Vec3A p, Vec3A* nearest) {
  Vec3A diff = p - from;
  Vec3A v = to - from;
  float t = dot(v, diff);

  if (t > 0.0f) {
    float dot_vv = dot(v, v);
    if (t < dot_vv) {
      t = lf_div(t, dot_vv);
      diff -= splat3(t) * v;
    } else {
      t = 1.0f;
      diff -= v;
    }
  } else {
    t = 0.0f;
  }

  *nearest = from + splat3(t) * v;
  return dot(diff, diff);
}

// V3: triangle_shape.rs:47 edge(index)
LF_HDI Vec3A tri_edge(const Tri& t, int index) {
  switch (index) {
    case 0: return t.p1 - t.p0;
    case 1: return t.p2 - t.p1;
    default: return t.p0 - t.p2;
  }
}

// V3: triangle_shape.rs:92 face_contains
LF_HDI bool tri_face_contains(const Tri& t, Vec3A n, const Vec3A obj_to_points[3]) {
  Vec3A c0 = cross(tri_edge(t, 0), obj_to_points[0]);
  if (dot(c0, n) < 0.0f) return false;

  Vec3A c1 = cross(tri_edge(t, 1), obj_to_points[1]);
  if (dot(c1, n) < 0.0f) return false;

  Vec3A c2 = cross(tri_edge(t, 2), obj_to_points[2]);
  return dot(c2, n) >= 0.0f;
}

// V3: triangle_shape.rs:196 intersect_sphere_from_plane
LF_HDI bool tri_intersect_sphere_from_plane(const Tri& t, Vec3A obj_center, Vec3A obj_to_center, Vec3A triangle_normal,
                                            float distance_from_plane, float radius, float radius_with_threshold,
                                            float radius_with_threshold_sqr, ContactInfo* out) {
  if (distance_from_plane >= radius_with_threshold) return false;

  Vec3A obj_to_points[3] = {obj_to_center, obj_center - t.p1, obj_center - t.p2};

  Vec3A contact_point;
  if (tri_face_contains(t, triangle_normal, obj_to_points)) {
    contact_point = obj_center - triangle_normal * distance_from_plane;
  } else {
    float contact_capsule_radius_sqr = radius_with_threshold_sqr;
    float min_distance_sqr = contact_capsule_radius_sqr;
    contact_point = splat3(0.0f);

    for (int edge_idx = 0; edge_idx < 3; ++edge_idx) {
      Vec3A from = edge_idx == 0 ? t.p0 : edge_idx == 1 ? t.p1 : t.p2;
      Vec3A to = edge_idx == 0 ? t.p1 : edge_idx == 1 ? t.p2 : t.p0;
      Vec3A nearest_on_edge = splat3(0.0f);
      float distance_sqr = segment_sqr_distance(from, to, obj_center, &nearest_on_edge);
      if (distance_sqr < min_distance_sqr) {
        min_distance_sqr = distance_sqr;
        contact_point = nearest_on_edge;
      }
    }

    if (!(min_distance_sqr < contact_capsule_radius_sqr)) return false;
  }

  Vec3A contact_to_center = obj_center - contact_point;
  float distance_sqr = length_squared(contact_to_center);

  if (distance_sqr >= radius_with_threshold_sqr) return false;

  Vec3A result_normal;
  float depth;
  if (distance_sqr > LF_EPSILON) {
    float distance = lf_sqrt(distance_sqr);
    float inverse_distance = lf_div(1.0f, distance);
    result_normal = contact_to_center * inverse_distance;
    depth = lf_neg(lf_sub(radius, distance));
  } else {
    result_normal = triangle_normal;
    depth = lf_neg(radius);
  }

  out->result_normal = result_normal;
  out->contact_point = contact_point;
  out->depth = depth;
  return true;
}

// V3: triangle_shape.rs:150 intersect_sphere_front_precomputed (front side only; None is `false`)
LF_HDI bool tri_intersect_sphere_front_precomputed(const Tri& t, Vec3A obj_center, float radius, float radius_with_threshold,
                                                   float radius_with_threshold_sqr, ContactInfo* out) {
  Vec3A obj_to_center = obj_center - t.p0;
  float distance_from_plane = dot(obj_to_center, t.normal);
  if (distance_from_plane < 0.0f) return false;

  return tri_intersect_sphere_from_plane(t, obj_center, obj_to_center, t.normal, distance_from_plane, radius,
                                         radius_with_threshold, radius_with_threshold_sqr, out);
}

}  // namespace lfsim
