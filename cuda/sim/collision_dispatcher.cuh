// V3: bullet/collision/dispatch/collision_dispatcher.rs (the pool of persistent manifolds, release, the near callback), the
// P3b entry (collision_world.rs:123 -> collision_dispatcher.rs:695 dispatch_all_collision_pairs -> overlapping_pair_cache.rs:131
// process_all_overlapping_pairs). The manifold pool is ArenaState.manifolds[0..n_manifolds) in V3's `persistent_manifolds`
// order; V3's `manifold_table` is a search of that pool; `active` is `active_manifolds`.
#pragma once
#include "sim/narrowphase.cuh"
#include "sim/persistent_manifold.cuh"
#include "sim/sphere_concave_collision_alg.cuh"
#include "sim/convex_plane_collision_alg.cuh"
#include "sim/compound_collision_alg.cuh"
#include "sim/sphere_obb_collision_alg.cuh"
#include "sim/obb_obb_collision_alg.cuh"

namespace lfsim {

// V3: collision_dispatcher.rs:112 lookup_persistent_manifold: the pool index of the manifold of the unordered pair, or -1
LF_HDI int32_t dispatcher_lookup_manifold(const ArenaState* s, uint32_t body0_idx, uint32_t body1_idx) {
  uint32_t lo = body0_idx < body1_idx ? body0_idx : body1_idx;
  uint32_t hi = body0_idx < body1_idx ? body1_idx : body0_idx;
  for (uint32_t i = 0; i < s->n_manifolds; ++i) {
    uint32_t a = s->manifolds[i].body0_idx, b = s->manifolds[i].body1_idx;
    uint32_t mlo = a < b ? a : b, mhi = a < b ? b : a;
    if (mlo == lo && mhi == hi) return (int32_t)i;
  }
  return -1;
}

// V3: collision_dispatcher.rs:135 release_manifold(body0_idx, body1_idx): called by the near callback and by the broadphase
// when it removes a pair (axis_sweep.rs:321). The active list is retained without the released index, and the pool's
// last manifold moves into the hole (swap_remove), with its active entries renumbered.
LF_HDI void release_manifold(ArenaState* s, uint32_t body0_idx, uint32_t body1_idx) {
  int32_t found = dispatcher_lookup_manifold(s, body0_idx, body1_idx);
  if (found < 0) return;
  uint32_t index = (uint32_t)found;
  uint32_t last_index = s->n_manifolds - 1;

  // active_manifolds.retain(|&active| active != index)
  uint32_t w = 0;
  for (uint32_t r = 0; r < s->n_active; ++r) {
    if (s->active[r] != index) s->active[w++] = s->active[r];
  }
  for (uint32_t r = w; r < s->n_active; ++r) s->active[r] = 0;
  s->n_active = w;

  // persistent_manifolds.swap_remove(index)
  if (index != last_index) {
    s->manifolds[index] = s->manifolds[last_index];
    for (uint32_t r = 0; r < s->n_active; ++r) {
      if (s->active[r] == last_index) s->active[r] = index;
    }
  }
  Manifold z = {};
  s->manifolds[last_index] = z;
  s->n_manifolds = last_index;
}

// V3: collision_dispatcher.rs:88 insert_persistent_manifold: push at the end of the pool; -1 if the pool is full
LF_HDI int32_t dispatcher_insert_manifold(ArenaState* s, const Manifold& m) {
  if (s->n_manifolds >= (uint32_t)MAX_MANIFOLDS) {
    s->overflow[OVF_MANIFOLDS] += 1;
    return -1;
  }
  s->manifolds[s->n_manifolds] = m;
  return (int32_t)s->n_manifolds++;
}

LF_HDI void dispatcher_push_active(ArenaState* s, uint32_t idx) {
  if (s->n_active >= (uint32_t)MAX_ACTIVE) {
    s->overflow[OVF_ACTIVE] += 1;
    return;
  }
  s->active[s->n_active++] = idx;
}

// V3: collision_dispatcher.rs:237 empty_manifold_bodies: the body order of the empty manifold a gated-in pair with no contact keeps
LF_HDI void dispatcher_empty_manifold_bodies(const NpCtx& c, uint32_t a, uint32_t b, uint32_t* first, uint32_t* second) {
  uint32_t ka = body_const(c, a).shape_kind, kb = body_const(c, b).shape_kind;
  bool swap = false;
  if (ka == SHAPE_PLANE) swap = true;
  else if (ka == SHAPE_SPHERE && kb == SHAPE_COMPOUND) swap = false;
  else if (ka == SHAPE_SPHERE && kb == SHAPE_MESH) swap = false;
  else if (ka == SHAPE_MESH && kb == SHAPE_SPHERE) swap = true;
  else if (ka == SHAPE_MESH && kb == SHAPE_COMPOUND) swap = true;
  else if (ka == SHAPE_COMPOUND && kb == SHAPE_SPHERE) swap = true;
  *first = swap ? b : a;
  *second = swap ? a : b;
}

// V3: collision_dispatcher.rs:262 process_collision_inner for the pairs that do not go through a mesh or a compound child: the
// fresh manifold, or false for None. (Compound against plane or mesh is process_collision_after_child_gate, not here.)
LF_HDI bool dispatcher_process_collision_inner(NpCtx& c, uint32_t a, uint32_t b, Manifold* out) {
  uint32_t ka = body_const(c, a).shape_kind, kb = body_const(c, b).shape_kind;
  if (ka == SHAPE_PLANE && kb == SHAPE_SPHERE) return convex_plane_process_collision(c, b, a, body_trans(c, b), false, out);
  if (ka == SHAPE_SPHERE && kb == SHAPE_PLANE) return convex_plane_process_collision(c, a, b, body_trans(c, a), false, out);
  if (ka == SHAPE_SPHERE && kb == SHAPE_COMPOUND) return alg_sphere_obb(c, a, b, out);
  if (ka == SHAPE_COMPOUND && kb == SHAPE_SPHERE) return alg_sphere_obb(c, b, a, out);
  if (ka == SHAPE_COMPOUND && kb == SHAPE_COMPOUND) return alg_obb_obb(c, a, b, out);
  c.unimplemented += 1;  // a pair of shapes V3 does not dispatch in Soccar 1v1
  return false;
}

// V3: collision_dispatcher.rs:535 near_callback_with_cached_bounds / near_callback_with_bounds (the caches only memoise
// get_aabb, whose value does not change within the phase). In two parts around the steps that need the deep narrowphase, the ball or a
// car against a triangle mesh (the arena): dispatcher_near_head runs the callback up to there and hands such a pair over as a
// SphereMeshJob or a CompoundMeshJob, dispatch_all_collision_pairs runs the job, dispatcher_near_tail finishes the pair
// (the barrier rule above). Every other pair is finished in the head.
struct NearPair {
  uint32_t rb0_idx, rb1_idx;
  int32_t cached_idx;
  int32_t sphere_idx;  // a ball-mesh pair: the pool index of its manifold
};
enum NearResult : int { NEAR_DONE = 0, NEAR_COMPOUND_JOB = 1, NEAR_SPHERE_JOB = 2 };

// The second half of the callback (:629-692): the pair's fresh manifold (if any) merged into the retained one, or inserted, and the
// manifold made active.
LF_HDI void dispatcher_near_tail(NpCtx& c, const NearPair& np, bool has_fresh, const Manifold& fresh) {
  ArenaState* s = c.s;
  uint32_t rb0_idx = np.rb0_idx, rb1_idx = np.rb1_idx;
  int32_t cached_idx = np.cached_idx;

  // Share the persistent manifold with the solver by index. Merge and refresh order is as in V3 (:629-692).
  int32_t active_idx;
  if (cached_idx >= 0 && has_fresh) {
    Manifold* cached = &s->manifolds[cached_idx];
    uint32_t body0_idx = cached->body0_idx, body1_idx = cached->body1_idx;
    if (cached->point_count != 0) manifold_refresh_contact_points(c, cached, body0_idx, body1_idx);
    manifold_merge_contact_points(c, cached, fresh);
    active_idx = cached_idx;
  } else if (cached_idx >= 0) {
    Manifold* cached = &s->manifolds[cached_idx];
    uint32_t body0_idx = cached->body0_idx, body1_idx = cached->body1_idx;
    if (cached->point_count != 0) manifold_refresh_contact_points(c, cached, body0_idx, body1_idx);

    // No fresh point reported this pair. Report the retained contacts after refresh removes invalid points.
    for (uint32_t i = 0; i < cached->point_count; ++i) {
      ManifoldPoint point = cached->points[i];
      tracker_contact_processed(c, point, body0_idx, body1_idx);
    }
    active_idx = cached_idx;
  } else if (has_fresh) {
    active_idx = dispatcher_insert_manifold(s, fresh);
  } else {
    uint32_t b0, b1;
    dispatcher_empty_manifold_bodies(c, rb0_idx, rb1_idx, &b0, &b1);
    active_idx = dispatcher_insert_manifold(s, manifold_new(b0, b1));
  }
  if (active_idx < 0) return;

  dispatcher_push_active(s, (uint32_t)active_idx);
}

// The first half of the callback: NEAR_DONE when the pair is finished, else the kind of job set up in `cj` or `sj` (`np` is then what
// the end of the pair needs once the job is finished).
LF_HDI int dispatcher_near_head(NpCtx& c, uint32_t rb0_idx, uint32_t rb1_idx, NearPair& np, CompoundMeshJob& cj, SphereMeshJob& sj) {
  ArenaState* s = c.s;

  if ((!body_is_active(c, rb0_idx) && !body_is_active(c, rb1_idx)) || !body_has_contact_response(c, rb0_idx) ||
      !body_has_contact_response(c, rb1_idx))
    return NEAR_DONE;

  int32_t cached_idx = dispatcher_lookup_manifold(s, rb0_idx, rb1_idx);
  np.rb0_idx = rb0_idx;
  np.rb1_idx = rb1_idx;
  np.cached_idx = cached_idx;

  uint32_t k0 = body_const(c, rb0_idx).shape_kind, k1 = body_const(c, rb1_idx).shape_kind;

  // compound_child_pair (:182): a compound against a non-compound (a triangle mesh counts), the compound first
  bool has_compound_pair = false;
  uint32_t compound_idx = 0, other_idx = 0;
  if (k0 == SHAPE_COMPOUND && k1 != SHAPE_COMPOUND) {
    has_compound_pair = true;
    compound_idx = rb0_idx;
    other_idx = rb1_idx;
  } else if (k1 == SHAPE_COMPOUND && k0 != SHAPE_COMPOUND) {
    has_compound_pair = true;
    compound_idx = rb1_idx;
    other_idx = rb0_idx;
  }
  if (has_compound_pair) {
    // cached_compound_child_overlaps_other (:219)
    Aabb child_aabb = compound_child_world_aabb(c, compound_idx);
    Aabb other_aabb = body_raw_aabb(c, other_idx);
    if (!aabb_intersects(child_aabb, other_aabb)) {
      release_manifold(s, rb0_idx, rb1_idx);
      return NEAR_DONE;
    }
  }

  // mesh_collision (:299): the ball against a mesh works in the pair's retained manifold (ConvexHull shapes do not occur)
  if ((k0 == SHAPE_SPHERE && k1 == SHAPE_MESH) || (k0 == SHAPE_MESH && k1 == SHAPE_SPHERE)) {
    uint32_t convex_obj = k0 == SHAPE_SPHERE ? rb0_idx : rb1_idx;
    uint32_t concave_obj = k0 == SHAPE_SPHERE ? rb1_idx : rb0_idx;
    int32_t persistent_idx = cached_idx;
    if (persistent_idx < 0) persistent_idx = dispatcher_insert_manifold(s, manifold_new(convex_obj, concave_obj));
    if (persistent_idx < 0) return NEAR_DONE;

    np.sphere_idx = persistent_idx;
    if (sphere_mesh_begin(c, convex_obj, concave_obj, &s->manifolds[persistent_idx], sj)) return NEAR_SPHERE_JOB;
    dispatcher_push_active(s, (uint32_t)persistent_idx);
    return NEAR_DONE;
  }

  Manifold fresh = {};
  bool has_fresh = false;
  if (has_compound_pair && (body_const(c, other_idx).shape_kind == SHAPE_PLANE || body_const(c, other_idx).shape_kind == SHAPE_MESH)) {
    // compound_collision_alg::process_collision_after_child_gate
    if (body_const(c, other_idx).shape_kind == SHAPE_PLANE)
      has_fresh = alg_compound_plane(c, compound_idx, other_idx, &fresh);
    else if (compound_mesh_begin(c, compound_idx, other_idx, cj))
      return NEAR_COMPOUND_JOB;
  } else {
    has_fresh = dispatcher_process_collision_inner(c, rb0_idx, rb1_idx, &fresh);
  }
  dispatcher_near_tail(c, np, has_fresh, fresh);
  return NEAR_DONE;
}

// V3: collision_dispatcher.rs:695 dispatch_all_collision_pairs -> overlapping_pair_cache.rs:131 process_all_overlapping_pairs:
// the retained pairs in pair-array order. Returns the number of pair algorithms that are not ported (0: the phase is complete).
//
// The ball or a car against a triangle mesh (the arena) is a job, and the driver loop gives it one leaf per round: the lane takes the
// next leaf of its BVH traversal and tests it (the ball: inline; the car: the box-triangle SAT, a call that every lane of the warp
// makes in every round, with a triangle that is rejected at once when it has no leaf) and adds the leaf's contact to the manifold,
// in one place for both. The contact's code (the tracker, the internal-edge fix, a correctly rounded sin and cos) and the SAT are the
// deepest code of the phase: called from inside the BVH traversal, the pair loop and the near callback, as V3's callbacks are,
// the divergent regions around them and their own did not fit the twelve convergence barriers (B0..B11) and ptxas spilled barriers
// to registers. The traversal reads nothing the contacts write, so every lane does V3's operations in
// V3's order.
LF_HDI uint32_t dispatch_all_collision_pairs(ArenaState* s, const StaticData* sd) {
  NpCtx c = {s, sd, 0};
  uint32_t n = s->n_pairs < (uint32_t)MAX_PAIRS ? s->n_pairs : (uint32_t)MAX_PAIRS;
  uint32_t i = 0;
  int kind = NEAR_DONE;  // the job in progress: NEAR_DONE (none), NEAR_COMPOUND_JOB or NEAR_SPHERE_JOB
  NearPair np = {};
  CompoundMeshJob cj;
  SphereMeshJob sj;
  compound_mesh_idle(cj, sd);
  LF_NO_UNROLL
  while (kind != NEAR_DONE || i < n) {
    if (kind == NEAR_DONE) {
      LF_NO_UNROLL
      while (i < n) {
        Pair p = s->pairs[i++];
        kind = dispatcher_near_head(c, sd->proxies[p.proxy0].client_obj_idx, sd->proxies[p.proxy1].client_obj_idx, np, cj, sj);
        if (kind != NEAR_DONE) break;
      }
    }
    const bool compound = kind == NEAR_COMPOUND_JOB, sphere = kind == NEAR_SPHERE_JOB;
    uint32_t leaf = 0;
    bool have = false;
    if (compound) have = compound_mesh_next(c, cj, &leaf);
    if (sphere) have = sphere_mesh_next(c, sj, &leaf);

    BoxTriangleSatContact ct;
    bool sat_hit = compound_mesh_sat(cj, have & compound, leaf, &ct);

    MeshContact mc;
    bool hit = false;
    if (compound & have & sat_hit) {
      compound_mesh_contact(cj, leaf, ct, &mc);
      hit = true;
    }
    if (sphere & have) hit = sphere_mesh_test(sj, leaf, &mc);

    // One flat test each, not nested ifs, and one call of manifold_add_contact_point for both kinds: every enclosing region counts
    // against the twelve barriers on sm_120 too.
    if (hit) manifold_add_contact_point(c, mc.target, mc.body0, mc.body1, mc.normal, mc.point, mc.depth, mc.idx);
    if (compound & !have) {
      Manifold fresh = {};
      bool has_fresh = compound_mesh_finish(c, cj, &fresh);
      dispatcher_near_tail(c, np, has_fresh, fresh);
      kind = NEAR_DONE;
    }
    if (sphere & !have) {
      sphere_mesh_finish(c, sj);
      dispatcher_push_active(s, (uint32_t)np.sphere_idx);
      kind = NEAR_DONE;
    }
  }
  return c.unimplemented;
}

}  // namespace lfsim
