// V3: bullet/collision/dispatch/sphere_concave_collision_alg.rs (the ball against one arena mesh).
//
// V3 runs the BVH traversal and, for every leaf, the sphere-triangle test and the contact in one callback (SphereTriangleCallback,
// :12, process_triangle :44). Here the same work is a resumable job whose steps the caller interleaves,
// the barrier rule): sphere_mesh_begin, then leaf by leaf sphere_mesh_next and sphere_mesh_test (the contact, if the leaf makes
// one), whose contact the caller adds to the manifold with manifold_add_contact_point (V3's order), then sphere_mesh_finish.
// sphere_concave_process_collision_into runs the whole job in place (V3's process_collision_into).
#pragma once
#include "sim/narrowphase.cuh"
#include "sim/bvh.cuh"
#include "sim/triangle_shape.cuh"
#include "sim/persistent_manifold.cuh"

namespace lfsim {

struct SphereMeshJob {
  Manifold* manifold;  // the pair's retained manifold, body0 = the ball, body1 = the mesh
  uint32_t sphere_obj, tri_obj;
  const MeshInfo* mesh;
  const Tri* tris;
  Vec3A sphere_center;
  float sphere_radius, radius_with_threshold, radius_with_threshold_sqr;
  Vec3A tri_translation;
  Aabb aabb;
  BvhCursor cur;
};

// V3: sphere_concave_collision_alg.rs:78 process_collision_into, up to the traversal (SphereTriangleCallback::new :25). False when
// StaticData has no such mesh (a pair V3 dispatches that the port does not).
LF_HDI bool sphere_mesh_begin(NpCtx& c, uint32_t convex_obj, uint32_t concave_obj, Manifold* manifold, SphereMeshJob& j) {
  Affine3A xform1 = body_trans(c, convex_obj);
  Affine3A mesh_trans = body_trans(c, concave_obj);
  const MeshInfo* mesh = mesh_of_body(c.sd, concave_obj);
  if (mesh == nullptr) {
    c.unimplemented += 1;
    return false;
  }

  Vec3A convex_in_triangle_space = xform1.translation - mesh_trans.translation;

  float contact_breaking_threshold = manifold_breaking_threshold(c, *manifold);
  float sphere_radius = body_const(c, convex_obj).sphere_margin;  // SphereShape::get_radius = margin
  float radius_with_threshold = lf_add(sphere_radius, contact_breaking_threshold);
  j.manifold = manifold;
  j.sphere_obj = convex_obj;
  j.tri_obj = concave_obj;
  j.mesh = mesh;
  j.tris = c.sd->tris + mesh->tri_first;
  j.sphere_center = convex_in_triangle_space;
  j.sphere_radius = sphere_radius;
  j.radius_with_threshold = radius_with_threshold;
  j.radius_with_threshold_sqr = lf_mul(radius_with_threshold, radius_with_threshold);
  j.tri_translation = mesh_trans.translation;
  j.aabb = sphere_aabb(sphere_radius, convex_in_triangle_space);
  bvh_cursor_init(j.cur, *mesh, j.aabb);
  return true;
}

// The next leaf of the traversal, in V3's order (the callback tests every leaf): its triangle index, or false when finished.
LF_HDI bool sphere_mesh_next(const NpCtx& c, SphereMeshJob& j, uint32_t* tri_idx) {
  return bvh_cursor_next(j.cur, c.sd, *j.mesh, j.aabb, tri_idx);
}

// SphereTriangleCallback::process_triangle (:44): the leaf's contact, or false when the sphere does not touch the triangle.
LF_HDI bool sphere_mesh_test(const SphereMeshJob& j, uint32_t tri_idx, MeshContact* out) {
  const Tri& triangle = j.tris[tri_idx];
  ContactInfo contact_info;
  if (!tri_intersect_sphere_front_precomputed(triangle, j.sphere_center, j.sphere_radius, j.radius_with_threshold, j.radius_with_threshold_sqr,
                                              &contact_info))
    return false;
  out->target = j.manifold;
  out->body0 = j.sphere_obj;
  out->body1 = j.tri_obj;
  out->normal = contact_info.result_normal;
  out->point = contact_info.contact_point + j.tri_translation;
  out->depth = contact_info.depth;
  out->idx = (int32_t)tri_idx;
  return true;
}

// The end of process_collision_into: true when the manifold holds points.
LF_HDI bool sphere_mesh_finish(NpCtx& c, SphereMeshJob& j) {
  if (j.manifold->point_count != 0) manifold_refresh_contact_points(c, j.manifold, j.sphere_obj, j.tri_obj);
  return j.manifold->point_count != 0;
}

// V3: sphere_concave_collision_alg.rs:78 process_collision_into(convex_obj, sphere_shape, concave_obj, tri_mesh, manifold, cb), the whole
// job in place (the start generator's legality test, cuda/env/overlap.cuh, which is not a phase). `manifold` is the (retained)
// manifold of the pair, body0 = the ball, body1 = the mesh.
LF_HDI bool sphere_concave_process_collision_into(NpCtx& c, uint32_t convex_obj, uint32_t concave_obj, Manifold* manifold) {
  SphereMeshJob job;
  if (!sphere_mesh_begin(c, convex_obj, concave_obj, manifold, job)) return false;
  uint32_t leaf;
  while (sphere_mesh_next(c, job, &leaf)) {
    MeshContact mc;
    if (sphere_mesh_test(job, leaf, &mc)) manifold_add_contact_point(c, mc.target, mc.body0, mc.body1, mc.normal, mc.point, mc.depth, mc.idx);
  }
  return sphere_mesh_finish(c, job);
}

}  // namespace lfsim
