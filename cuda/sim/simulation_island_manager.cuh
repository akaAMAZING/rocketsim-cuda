// P4 (phase index 5): V3's overlapping_dynamic_body_pairs and SimulationIslandManager::sort_manifolds.
//
// V3 sources (rev ad8b38c, paths under rocketsim/src/bullet/):
//   dynamics/simulation_island_manager.rs   sort_manifolds, find, union_dynamic_pairs, sort_active_manifolds, hoare_sort
//   collision/broadphase/grid_broadphase.rs:306 overlapping_dynamic_body_pairs
//   dynamics/discrete_dynamics_world.rs:220-230 (the call site)
//
// Differences from V3, each proved identical:
//   * the sort cache (cached_input / cached_output / sort_keys memo) is omitted: a hit returns what the sort would
//    . sort_keys persists between ticks in V3 but is only read for indices of the current sort, after
//     being rewritten from key_for, so a local array is the same.
//   * the island manager's body_to_dynamic / parents / island_tags vectors are local arrays of NUM_DYN entries (3 dynamic
//     bodies); the "mapping is current" memo only avoids recomputing a map that the dynamic body list fixes.
//   * V3's recursive Hoare partition is an explicit stack of (lo, hi) ranges. The two sub-ranges of a partition are
//     disjoint, so the order the ranges are visited in cannot change the result; the stack still pops the left range
//     first, as the recursion does. Depth is bounded: every pending range holds >= 2 of the <= MAX_ACTIVE elements.
//     tests/wp_f.rs compares it with the recursive form (lfsim::hoare_sort_recursive, below) on random keys.
#pragma once
#include "sim/arena_state.cuh"
#include "sim/rigid_body.cuh"

namespace lfsim {

constexpr uint32_t COLLISION_GROUP_STATIC = 1u << 1;  // broadphase_proxy.rs:7 CollisionFilterGroups::Static
constexpr uint32_t FLAG_NO_CONTACT_RESPONSE = 1u << 3;  // rigid_body.rs:54 CollisionFlags::NoContactResponse

// Dynamic index (0 ball, 1 car 0, 2 car 1) of world body w, or -1 for a static body: V3's body_to_dynamic
// (simulation_island_manager.rs:87-104; usize::MAX is -1 here). sd->dyn_body is dynamic_body_idcs (discrete_dynamics_world.rs:27).
LF_HDI int dyn_index_of(const StaticData* sd, uint32_t w) {
  for (int d = 0; d < NUM_DYN; ++d)
    if (sd->dyn_body[d] == w) return d;
  return -1;
}

// V3: simulation_island_manager.rs:177-182 union_dynamic_pairs::find, and :101-114 find (recursive, full path compression). The iterative form sets
// every node on the path to the root, which is what the recursion's assignments do.
LF_HDI int island_find(int* parents, int index) {
  int root = index;
  while (parents[root] != root) root = parents[root];
  int cur = index;
  while (parents[cur] != root && cur != root) {
    int next = parents[cur];
    parents[cur] = root;
    cur = next;
  }
  return root;
}

// V3: simulation_island_manager.rs:193-230 hoare_sort: Hoare partition of `indices` by `keys[index]`, explicit stack.
// Non-stable: the algorithm itself decides the order of equal keys.
LF_HDI void hoare_sort(uint32_t* indices, int n, const int* keys) {
  if (n <= 1) return;
  int stack_lo[32], stack_hi[32];
  int sp = 0;
  stack_lo[0] = 0;
  stack_hi[0] = n - 1;
  sp = 1;
  while (sp > 0) {
    --sp;
    int lo = stack_lo[sp], hi = stack_hi[sp];
    int i = lo, j = hi;
    int pivot = keys[indices[(lo + hi) / 2]];
    for (;;) {
      while (keys[indices[i]] < pivot) ++i;
      while (pivot < keys[indices[j]]) --j;
      if (i > j) break;
      uint32_t t = indices[i]; indices[i] = indices[j]; indices[j] = t;
      ++i;
      if (j == 0) break;
      --j;
      if (i > j) break;
    }
    // recursion order: left range first, so push the right range first
    if (i < hi) { stack_lo[sp] = i; stack_hi[sp] = hi; ++sp; }
    if (lo < j) { stack_lo[sp] = lo; stack_hi[sp] = j; ++sp; }
  }
}

// The recursive form of V3 (simulation_island_manager.rs:194-226), kept for the equivalence test only.
LF_HDI void hoare_sort_recursive(uint32_t* indices, const int* keys, int lo, int hi) {
  int i = lo, j = hi;
  int pivot = keys[indices[(lo + hi) / 2]];
  for (;;) {
    while (keys[indices[i]] < pivot) ++i;
    while (pivot < keys[indices[j]]) --j;
    if (i > j) break;
    uint32_t t = indices[i]; indices[i] = indices[j]; indices[j] = t;
    ++i;
    if (j == 0) break;
    --j;
    if (i > j) break;
  }
  if (lo < j) hoare_sort_recursive(indices, keys, lo, j);
  if (i < hi) hoare_sort_recursive(indices, keys, i, hi);
}

// V3: simulation_island_manager.rs:22-78 sort_manifolds, with grid_broadphase.rs:306 overlapping_dynamic_body_pairs inlined.
// Puts to sleep the bodies that want deactivation in islands with no active body, then reorders s->active[0..n_active) (pool
// indices) by island.
LF_HD inline void phase_p4(ArenaState* s, const StaticData* sd) {
  const int n = (int)s->n_active;

  int parents[NUM_DYN];  // :38 parents[d] = d
  for (int d = 0; d < NUM_DYN; ++d) parents[d] = d;

  // pairs of the retained list whose two proxies are both non-static, in list order (grid_broadphase.rs:310-321)
  for (uint32_t p = 0; p < s->n_pairs; ++p) {
    const ProxyConst& px0 = sd->proxies[s->pairs[p].proxy0];
    const ProxyConst& px1 = sd->proxies[s->pairs[p].proxy1];
    if ((px0.filter_group & COLLISION_GROUP_STATIC) != 0 || (px1.filter_group & COLLISION_GROUP_STATIC) != 0) continue;
    // :43-62 + :176-191 union_dynamic_pairs over the pairs with two dynamic bodies that both have contact response
    int a = dyn_index_of(sd, px0.client_obj_idx), b = dyn_index_of(sd, px1.client_obj_idx);
    if (a < 0 || b < 0) continue;
    if ((s->bodies[a].collision_flags & FLAG_NO_CONTACT_RESPONSE) != 0 || (s->bodies[b].collision_flags & FLAG_NO_CONTACT_RESPONSE) != 0) continue;
    int root0 = island_find(parents, a);  // :177-182
    int root1 = island_find(parents, b);
    if (root0 != root1) parents[root0] = root1;
  }

  int island_tags[NUM_DYN];  // :64-66
  for (int d = 0; d < NUM_DYN; ++d) island_tags[d] = island_find(parents, d);

  bool island_has_active_body[NUM_DYN] = {};  // :60-75
  for (int d = 0; d < NUM_DYN; ++d)
    if (s->bodies[d].activation == ACT_ACTIVE) island_has_active_body[island_tags[d]] = true;
  for (int d = 0; d < NUM_DYN; ++d)
    if (!island_has_active_body[island_tags[d]] && s->bodies[d].activation == ACT_WANTS_DEACTIVATION) rb_sleep(s->bodies[d]);

  if (n <= 1) return;  // :77-79

  // :129-174 sort_active_manifolds (n <= 64 branch, without the cache): ascending sort of the indices, then Hoare by island key
  uint32_t idx[MAX_ACTIVE];
  for (int i = 0; i < n; ++i) idx[i] = s->active[i];
  for (int i = 1; i < n; ++i) {  // sort_unstable of distinct integers has one result
    uint32_t v = idx[i];
    int j = i - 1;
    while (j >= 0 && idx[j] > v) { idx[j + 1] = idx[j]; --j; }
    idx[j + 1] = v;
  }
  int keys[MAX_MANIFOLDS];
  for (int i = 0; i < n; ++i) {
    const Manifold& m = s->manifolds[idx[i]];
    int d0 = dyn_index_of(sd, m.body0_idx), d1 = dyn_index_of(sd, m.body1_idx);
    int tag0 = d0 < 0 ? -1 : island_tags[d0];  // :116 body_tag
    int tag1 = d1 < 0 ? -1 : island_tags[d1];
    keys[idx[i]] = tag0 >= 0 ? tag0 : tag1;  // :125 manifold_island_key
  }
  hoare_sort(idx, n, keys);
  for (int i = 0; i < n; ++i) s->active[i] = idx[i];
}

}  // namespace lfsim
