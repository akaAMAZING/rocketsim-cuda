// V3: bullet/collision/dispatch/compound_collision_alg.rs, process_collision_after_child_gate (:119): a car (compound of one box)
// against the other body, after the dispatcher's child-AABB gate has passed. The StaticPlane arm is direct; the
// TriangleMesh arm is the compound_mesh_* job below.
#pragma once
#include "sim/narrowphase.cuh"
#include "sim/convex_plane_collision_alg.cuh"
#include "sim/box_triangle_sat.cuh"
#include "sim/bvh.cuh"

namespace lfsim {

// V3: compound_collision_alg.rs:178-187, the StaticPlane arm. The wrapper is the car with its box child at
// body_trans * child_trans.
LF_HDI bool alg_compound_plane(NpCtx& c, uint32_t compound_obj, uint32_t plane_obj, Manifold* out) {
  Affine3A new_child_world_trans = compound_child_world_trans(c, compound_obj);
  return convex_plane_process_collision(c, compound_obj, plane_obj, new_child_world_trans, true, out);
}

// ---- the TriangleMesh arm ---------------------------------------------------------------------------------------------
// V3: compound_collision_alg.rs:30 ConvexTriangleCallback, process_triangle (:42), and :138-177, the TriangleMesh arm of
// process_collision_after_child_gate (the dispatcher has done the child-AABB gate). V3 runs the BVH traversal, and for every leaf the
// triangle pre-filter, the box-triangle SAT and the contact, in one callback. Here the same work is a resumable job whose steps the
// caller interleaves (the barrier rule above): compound_mesh_begin (the setup), then leaf by leaf
// compound_mesh_next (the next BVH leaf that passes the pre-filter, in V3's order), compound_mesh_sat (the pure box-triangle test of
// that leaf) and compound_mesh_contact (the contact, which the caller adds to the manifold), then compound_mesh_finish. The SAT is the deepest code of the narrowphase, and
// P3b's driver (collision_dispatcher.cuh) calls it from the driver loop alone instead of from inside the traversal, the pair loop and
// the near callback. The traversal reads nothing the contacts write, so the order of the float operations and of the contacts is V3's.
// The manifold is created lazily by the first triangle that makes a contact (`get_or_insert_with`).
struct CompoundMeshJob {
  // the SAT's inputs (read by compound_mesh_sat)
  Affine3A box_trans, mesh_to_box;
  Vec3A half;
  float margin, maximum_distance;
  const Tri* tris;  // the mesh's triangles, indexed by the mesh-local triangle index
  // the contacts' inputs and result
  Manifold manifold;
  bool has_manifold;
  uint32_t convex_obj, tri_obj;
  Mat3A tri_matrix;
  // the traversal
  const MeshInfo* mesh;
  Aabb local_convex_aabb;
  BvhCursor cur;
};

// Defined SAT inputs for a lane that has no job yet: the SAT call of P3b's driver is made by every lane of the warp in every round,
// the lanes without a leaf pass a triangle the SAT's pre-screen rejects at once, and its result is dropped.
LF_HDI void compound_mesh_idle(CompoundMeshJob& j, const StaticData* sd) {
  j.box_trans = {};
  j.mesh_to_box = {};
  j.half = splat3(1.0f);
  j.margin = 0.0f;
  j.maximum_distance = 1.0f;
  j.tris = sd->tris;
}

// V3: compound_collision_alg.rs:138-165. False when StaticData has no such mesh (a pair V3 dispatches that the port does not).
LF_HDI bool compound_mesh_begin(NpCtx& c, uint32_t compound_obj, uint32_t mesh_obj, CompoundMeshJob& j) {
  const MeshInfo* mesh = mesh_of_body(c.sd, mesh_obj);
  if (mesh == nullptr) {
    c.unimplemented += 1;
    return false;
  }
  const BodyConst& bc = body_const(c, compound_obj);
  Affine3A new_child_world_trans = compound_child_world_trans(c, compound_obj);
  Affine3A other_trans = body_trans(c, mesh_obj);
  Affine3A xform1 = affine_transpose(other_trans);
  Affine3A xform2 = new_child_world_trans;
  Affine3A convex_in_triangle_space = {xform1.matrix3 * xform2.matrix3, transform_point3a(xform1, xform2.translation)};
  Aabb aabb_in_triangle = aabb_from_half_extents_transform(bc.box_half_extents, bc.box_margin, convex_in_triangle_space);

  Affine3A mesh_to_box = affine_transpose(new_child_world_trans) * other_trans;
  float breaking = rs_min(bc.contact_breaking_threshold, body_const(c, mesh_obj).contact_breaking_threshold);

  j.has_manifold = false;
  j.convex_obj = compound_obj;
  j.tri_obj = mesh_obj;
  j.mesh = mesh;
  j.tris = c.sd->tris + mesh->tri_first;
  j.local_convex_aabb = aabb_in_triangle;
  j.half = bc.box_half_extents;
  j.margin = bc.box_margin;
  j.maximum_distance = lf_add(bc.box_margin, breaking);
  j.box_trans = new_child_world_trans;
  j.mesh_to_box = mesh_to_box;
  j.tri_matrix = other_trans.matrix3;
  bvh_cursor_init(j.cur, *mesh, aabb_in_triangle);
  return true;
}

// The next leaf of the traversal that passes the triangle pre-filter (process_triangle's first test), in V3's order: its triangle
// index in `*tri_idx`, or false when the traversal is finished.
LF_HDI bool compound_mesh_next(const NpCtx& c, CompoundMeshJob& j, uint32_t* tri_idx) {
  uint32_t leaf;
  while (bvh_cursor_next(j.cur, c.sd, *j.mesh, j.local_convex_aabb, &leaf)) {
    const Tri& triangle = j.tris[leaf];
    Vec3A tri_min = min(min(triangle.p0, triangle.p1), triangle.p2);
    Vec3A tri_max = max(max(triangle.p0, triangle.p1), triangle.p2);
    if (!(glam::all(cmple(tri_min, j.local_convex_aabb.max)) && glam::all(cmpge(tri_max, j.local_convex_aabb.min)))) continue;
    *tri_idx = leaf;
    return true;
  }
  return false;
}

// process_triangle's SAT: V3's Some(contact) as true. Pure: it reads the job's inputs and the triangle. With `have` false it tests a
// triangle far from the box instead (the pre-screen says no without a division or a call), so that a lane without a leaf can make the
// same call as the lanes with one.
LF_HDI bool compound_mesh_sat(const CompoundMeshJob& j, bool have, uint32_t tri_idx, BoxTriangleSatContact* contact) {
  const Tri& triangle = j.tris[have ? tri_idx : 0];
  const Vec3A far = splat3(1e30f);
  Vec3A q[3] = {have ? transform_point3a(j.mesh_to_box, triangle.p0) : far, have ? transform_point3a(j.mesh_to_box, triangle.p1) : far,
                have ? transform_point3a(j.mesh_to_box, triangle.p2) : far};
  return box_triangle_sat(j.box_trans, j.half, j.margin, q, j.maximum_distance, contact);
}

// process_triangle's contact, for a triangle whose SAT said Some: the manifold is created by the first one.
LF_HDI void compound_mesh_contact(CompoundMeshJob& j, uint32_t tri_idx, const BoxTriangleSatContact& contact, MeshContact* out) {
  const Tri& triangle = j.tris[tri_idx];
  if (!j.has_manifold) {
    j.manifold = manifold_new(j.convex_obj, j.tri_obj);
    j.has_manifold = true;
  }
  out->target = &j.manifold;
  out->body0 = j.convex_obj;
  out->body1 = j.tri_obj;
  out->normal = j.tri_matrix * triangle.normal;
  out->point = contact.point_on_b_world;
  out->depth = contact.distance;
  out->idx = (int32_t)tri_idx;
}

// V3: compound_collision_alg.rs:170-177. Returns true exactly when V3 leaves `*out = Some(manifold)`.
LF_HDI bool compound_mesh_finish(NpCtx& c, CompoundMeshJob& j, Manifold* out) {
  if (j.has_manifold) {
    manifold_refresh_contact_points(c, &j.manifold, j.convex_obj, j.tri_obj);
    if (j.manifold.point_count != 0) {
      *out = j.manifold;
      return true;
    }
  }
  return false;
}

// The whole job in place, one leaf after another (the start generator's legality test, cuda/env/overlap.cuh, which is not a phase):
// V3's alg_compound_mesh. P3b does not call it; its driver interleaves the same steps.
LF_HDI bool alg_compound_mesh(NpCtx& c, uint32_t compound_obj, uint32_t mesh_obj, Manifold* out) {
  CompoundMeshJob job;
  if (!compound_mesh_begin(c, compound_obj, mesh_obj, job)) return false;
  uint32_t leaf;
  while (compound_mesh_next(c, job, &leaf)) {
    BoxTriangleSatContact ct;
    if (compound_mesh_sat(job, true, leaf, &ct)) {
      MeshContact mc;
      compound_mesh_contact(job, leaf, ct, &mc);
      manifold_add_contact_point(c, mc.target, mc.body0, mc.body1, mc.normal, mc.point, mc.depth, mc.idx);
    }
  }
  return compound_mesh_finish(c, job, out);
}

}  // namespace lfsim
