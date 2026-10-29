// V3: shared/bvh.rs (Tree::report_aabb_overlapping_node and WideNode::intersection_mask) over StaticData's nodes and leaves, and
// bvh_triangle_mesh_shape.rs:48 process_all_triangles. Depth-first in V3's exact order: the children that overlap are pushed in
// reverse lane order, so lane 0 pops first; a leaf is reported when popped.
#pragma once
#include "sim/narrowphase.cuh"

namespace lfsim {

constexpr int BVH_STACK_SIZE = 128;           // bvh.rs:25 TRAVERSAL_STACK_SIZE
constexpr uint32_t BVH_LEAF_BIT = 1u << 31;   // bvh.rs:496 WideChild::LEAF_BIT

// V3: bvh.rs:487 WideNode::intersection_mask
LF_HDI uint32_t wide_node_intersection_mask(const WideNode& n, const Aabb& aabb) {
  uint32_t mask = 0;
  for (int l = 0; l < 4; ++l) {
    bool overlap = n.min_x[l] <= aabb.max.x && n.max_x[l] >= aabb.min.x && n.min_y[l] <= aabb.max.y &&
                   n.max_y[l] >= aabb.min.y && n.min_z[l] <= aabb.max.z && n.max_z[l] >= aabb.min.z;
    mask |= (overlap ? 1u : 0u) << l;
  }
  return mask & n.valid_mask;
}

// V3: bvh.rs:243 report_aabb_overlapping_node as a cursor: the traversal stack, resumable between leaves (the compound-mesh pairs of
// P3b collect a batch of leaves, run the box-triangle SAT on them in a flat loop and apply them in order, collision_dispatcher.cuh).
struct BvhCursor {
  uint32_t stack[BVH_STACK_SIZE];
  int stack_len;
};

LF_HDI void bvh_cursor_init(BvhCursor& cur, const MeshInfo& mesh, const Aabb& aabb) {
  cur.stack_len = 0;
  if (!aabb_intersects(aabb, mesh.root_aabb)) return;
  cur.stack[0] = 0;  // WideChild::branch(0)
  cur.stack_len = 1;
}

// The next leaf in V3's order: its triangle index local to the mesh (tri_idx), or false when the traversal is finished.
LF_HDI bool bvh_cursor_next(BvhCursor& cur, const StaticData* sd, const MeshInfo& mesh, const Aabb& aabb, uint32_t* tri_idx) {
  while (cur.stack_len != 0) {
    cur.stack_len -= 1;
    uint32_t work = cur.stack[cur.stack_len];
    if (work & BVH_LEAF_BIT) {
      *tri_idx = sd->leaves[mesh.leaf_first + (work & ~BVH_LEAF_BIT)].leaf_idx;
      return true;
    }

    const WideNode& node = sd->nodes[mesh.node_first + work];
    uint32_t mask = wide_node_intersection_mask(node, aabb);
    for (int lane = (int)node.child_count - 1; lane >= 0; --lane) {
      if (mask & (1u << lane)) {
        if (cur.stack_len < BVH_STACK_SIZE) cur.stack[cur.stack_len++] = node.children[lane];
      }
    }
  }
  return false;
}

// V3: bvh.rs:243 report_aabb_overlapping_node. `f(triangle_idx)` is NodeOverlapCallback::process_node (tri_bvh_util.rs:24),
// the triangle index local to the mesh. `mesh` is the mesh's MeshInfo.
template <class F>
LF_HDI void bvh_report_aabb_overlapping_node(const StaticData* sd, const MeshInfo& mesh, const Aabb& aabb, F& f) {
  BvhCursor cur;
  bvh_cursor_init(cur, mesh, aabb);
  uint32_t tri_idx;
  while (bvh_cursor_next(cur, sd, mesh, aabb, &tri_idx)) f(tri_idx);
}

// MeshInfo of a mesh body (bodies 0-15 are the meshes in order); nullptr if StaticData disagrees.
LF_HDI const MeshInfo* mesh_of_body(const StaticData* sd, uint32_t body) {
  return (body < (uint32_t)NUM_MESHES && sd->meshes[body].body == body) ? &sd->meshes[body] : nullptr;
}

}  // namespace lfsim
