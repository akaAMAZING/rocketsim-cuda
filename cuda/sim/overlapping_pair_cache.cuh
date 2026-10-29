// P3a: the overlapping pair array. V3's triangular pair_indices table is a search over the pair array here
//: a pair is found by its two handles, which are the proxies' unique ids (unique_id == handle).
// Pairs are appended at the end and removed by swap with the last, so the array order is V3's history.
#pragma once
#include "sim/arena_state.cuh"

namespace lfsim {

// V3: bullet/collision/broadphase/overlapping_pair_cache.rs:100 needs_broadphase_collision
LF_HDI bool needs_broadphase_collision(const ProxyConst& p0, const ProxyConst& p1) {
  return (p0.filter_group & p1.filter_mask) != 0 && (p1.filter_group & p0.filter_mask) != 0;
}

// V3: overlapping_pair_cache.rs:38 find the retained pair (id0 <= id1 are unique ids); -1 when absent
LF_HDI int find_pair(const ArenaState* s, const StaticData* sd, uint32_t id0, uint32_t id1) {
  for (uint32_t i = 0; i < s->n_pairs; ++i) {
    if (sd->proxies[s->pairs[i].proxy0].unique_id == id0 && sd->proxies[s->pairs[i].proxy1].unique_id == id1) return (int)i;
  }
  return -1;
}

// V3: overlapping_pair_cache.rs:28 add_overlapping_pair: filter, deduplicate, append at the end
LF_HDI bool add_overlapping_pair(ArenaState* s, const StaticData* sd, uint32_t h0, uint32_t h1) {
  const ProxyConst& p0 = sd->proxies[h0];
  const ProxyConst& p1 = sd->proxies[h1];
  if (!needs_broadphase_collision(p0, p1)) return false;
  uint32_t id0, idx0, id1, idx1;
  if (p0.unique_id <= p1.unique_id) { id0 = p0.unique_id; idx0 = h0; id1 = p1.unique_id; idx1 = h1; }
  else { id0 = p1.unique_id; idx0 = h1; id1 = p0.unique_id; idx1 = h0; }
  if (find_pair(s, sd, id0, id1) >= 0) return false;
  if (s->n_pairs >= (uint32_t)MAX_PAIRS) { s->overflow[OVF_PAIRS] += 1; return false; }
  s->pairs[s->n_pairs] = Pair{idx0, idx1};
  s->n_pairs += 1;
  return true;
}

// V3: overlapping_pair_cache.rs:63 remove_overlapping_pair: swap with last and pop; returns false when the pair is absent
LF_HDI bool remove_overlapping_pair(ArenaState* s, const StaticData* sd, uint32_t id0, uint32_t id1, Pair* removed) {
  int index = find_pair(s, sd, id0, id1);
  if (index < 0) return false;
  uint32_t last = s->n_pairs - 1;
  *removed = s->pairs[index];
  if ((uint32_t)index != last) s->pairs[index] = s->pairs[last];
  s->pairs[last] = Pair{0, 0};
  s->n_pairs -= 1;
  return true;
}

}  // namespace lfsim
