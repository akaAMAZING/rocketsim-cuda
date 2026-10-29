// P3a: the one piece of the dispatcher that the sweep-and-prune reaches, release_manifold. V3's manifold_table is a
// search over the manifold pool. Named p3a_* so that it does not clash with the
// full dispatcher's own release_manifold.
#pragma once
#include "sim/arena_state.cuh"

namespace lfsim {

// V3: bullet/collision/dispatch/collision_dispatcher.rs:135 release_manifold (lookup at :122 is the search)
LF_HDI void p3a_release_manifold(ArenaState* s, uint32_t body0_idx, uint32_t body1_idx) {
  uint32_t lo = body0_idx < body1_idx ? body0_idx : body1_idx;
  uint32_t hi = body0_idx < body1_idx ? body1_idx : body0_idx;
  int found = -1;
  for (uint32_t i = 0; i < s->n_manifolds; ++i) {
    uint32_t a = s->manifolds[i].body0_idx, b = s->manifolds[i].body1_idx;
    uint32_t mlo = a < b ? a : b, mhi = a < b ? b : a;
    if (mlo == lo && mhi == hi) { found = (int)i; break; }
  }
  if (found < 0) return;
  uint32_t index = (uint32_t)found;
  uint32_t last = s->n_manifolds - 1;
  // active_manifolds.retain(|&a| a != index)
  uint32_t w = 0;
  for (uint32_t i = 0; i < s->n_active; ++i)
    if (s->active[i] != index) s->active[w++] = s->active[i];
  for (uint32_t i = w; i < s->n_active; ++i) s->active[i] = 0;
  s->n_active = w;
  // persistent_manifolds.swap_remove(index)
  if (index != last) s->manifolds[index] = s->manifolds[last];
  {
    uint32_t* z = reinterpret_cast<uint32_t*>(&s->manifolds[last]);
    for (size_t k = 0; k < sizeof(Manifold) / 4; ++k) z[k] = 0;
  }
  s->n_manifolds -= 1;
  if (index != last) {
    for (uint32_t i = 0; i < s->n_active; ++i)
      if (s->active[i] == last) s->active[i] = index;
  }
}

}  // namespace lfsim
