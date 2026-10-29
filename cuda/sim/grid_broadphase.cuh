// P3a: GridBroadphase::set_aabb for a dynamic body: the sweep-and-prune update, then the dynamic ray-cell lists.
// The per-cell Vec<usize> lists are the flat dyn_cells array sorted by cell, entries of one cell in list order
//: push is an insert after the cell's last entry, remove(first match) deletes the cell's first match.
#pragma once
#include "sim/axis_sweep.cuh"

namespace lfsim {
using namespace glam;

// V3: grid_broadphase.rs:51 get_cell_indices (pos - min_pos) / cell_size, truncated, clamped to [0, num_cells - 1]
LF_HDI void get_cell_indices(const StaticData* sd, Vec3A pos, uint32_t out[3]) {
  Vec3A f = (pos - sd->grid_min_pos) / sd->grid_cell_size;
  int32_t v[3] = {lf_f2i(f.x), lf_f2i(f.y), lf_f2i(f.z)};
  for (int a = 0; a < 3; ++a) {
    int32_t m = v[a] > 0 ? v[a] : 0;  // IVec3::max(ZERO), then as usize
    uint32_t hi = sd->grid_num_cells[a] - 1;
    out[a] = (uint32_t)m < hi ? (uint32_t)m : hi;
  }
}

// V3: grid_broadphase.rs:69 cell_indices_to_idx
LF_HDI uint32_t cell_indices_to_idx(const StaticData* sd, const uint32_t i[3]) {
  return i[0] * sd->grid_num_cells[1] * sd->grid_num_cells[2] + i[1] * sd->grid_num_cells[2] + i[2];
}

// V3: grid_broadphase.rs:123 update_cells_dynamic, ADD branch (GridCell.dyn_handles.push)
LF_HDI void dyn_cell_push(ArenaState* s, uint32_t cell, uint32_t handle) {
  if (s->n_dyn_cells >= (uint32_t)MAX_DYN_CELLS) { s->overflow[OVF_DYN_CELLS] += 1; return; }
  uint32_t at = 0;
  while (at < s->n_dyn_cells && s->dyn_cells[at].cell <= cell) ++at;
  for (uint32_t i = s->n_dyn_cells; i > at; --i) s->dyn_cells[i] = s->dyn_cells[i - 1];
  s->dyn_cells[at] = DynCell{cell, handle};
  s->n_dyn_cells += 1;
}

// V3: grid_broadphase.rs:34 GridCell::remove_dyn (first match, order kept)
LF_HDI void dyn_cell_remove(ArenaState* s, uint32_t cell, uint32_t handle) {
  for (uint32_t at = 0; at < s->n_dyn_cells; ++at) {
    if (s->dyn_cells[at].cell == cell && s->dyn_cells[at].handle == handle) {
      for (uint32_t i = at; i + 1 < s->n_dyn_cells; ++i) s->dyn_cells[i] = s->dyn_cells[i + 1];
      s->dyn_cells[s->n_dyn_cells - 1] = DynCell{0, 0};
      s->n_dyn_cells -= 1;
      return;
    }
  }
}

// V3: grid_broadphase.rs:123 update_cells_dynamic<ADD>
LF_HDI void update_cells_dynamic(ArenaState* s, const StaticData* sd, bool add, uint32_t handle, const uint32_t idx[3]) {
  uint32_t lo[3], hi[3];
  for (int a = 0; a < 3; ++a) {
    uint32_t one_max = idx[a] > 1 ? idx[a] : 1;
    lo[a] = one_max - 1;
    uint32_t top = sd->grid_num_cells[a] - 1;
    uint32_t p = idx[a] + 1;
    hi[a] = top < p ? top : p;
  }
  for (uint32_t i = lo[0]; i <= hi[0]; ++i)
    for (uint32_t j = lo[1]; j <= hi[1]; ++j)
      for (uint32_t k = lo[2]; k <= hi[2]; ++k) {
        uint32_t c[3] = {i, j, k};
        uint32_t cell = cell_indices_to_idx(sd, c);
        if (add) dyn_cell_push(s, cell, handle); else dyn_cell_remove(s, cell, handle);
      }
}

// V3: grid_broadphase.rs:171 GridBroadphase::set_aabb, dynamic branch; d is the dynamic body index (proxy dyn_proxy[d])
LF_HDI void grid_set_aabb_dynamic(ArenaState* s, const StaticData* sd, uint32_t d, Aabb aabb) {
  uint32_t handle = sd->dyn_handle[d];
  s->dyn_proxy[d].aabb = aabb;                                  // axis_sweep.rs:141 handles[handle].aabb = aabb
  axis_sweep_set_aabb(s, sd, handle, aabb);                     // cache the bounds and update the edges first
  uint32_t old_idx = s->dyn_proxy[d].cell_idx;
  uint32_t ni[3];
  get_cell_indices(sd, aabb.min, ni);
  uint32_t new_idx = cell_indices_to_idx(sd, ni);
  s->dyn_proxy[d].cell_idx = new_idx;
  if (new_idx != old_idx) {
    update_cells_dynamic(s, sd, false, handle, s->dyn_proxy[d].indices);
    s->dyn_proxy[d].indices[0] = ni[0];
    s->dyn_proxy[d].indices[1] = ni[1];
    s->dyn_proxy[d].indices[2] = ni[2];
    update_cells_dynamic(s, sd, true, handle, ni);
  }
}

}  // namespace lfsim
