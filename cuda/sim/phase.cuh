// The phase entry points. `phase(k, s, sd)` runs phase k of V3's
// tick on one arena: it takes the state at boundary k and leaves the state at boundary k + 1 (phase indices: 0 P0, 1 P1,
// 2 P2, 3 P3a, 4 P3b, 5 P4, 6 P5, 7 P6, 8 P7, 9 P8). The same function is the host C ABI rscuda_phase (csrc/phase.cpp) and the
// body of the device kernel phase_k (sim.cu). step_tick (arena_step.cuh) runs the same ten tick_p* functions in order.
#pragma once
#include "sim/arena_base.cuh"
#include "sim/arena_base_p1.cuh"
#include "sim/arena_state.cuh"
#include "sim/collision_world.cuh"
#include "sim/discrete_dynamics_world.cuh"
#include "sim/seq_impulse_constraint_solver.cuh"
#include "sim/simulation_island_manager.cuh"
#include "sim/collision_dispatcher.cuh"

namespace lfsim {

constexpr int NUM_PHASES = 10;
enum PhaseStatus : int { PHASE_OK = 0, PHASE_UNIMPLEMENTED = -1, PHASE_BAD_INDEX = -2 };

// The ten phases, each a function of its own on the device (LF_NOINLINE): phase_k and the tick kernels share one copy of each.
LF_HD LF_NOINLINE inline void tick_p0(ArenaState* s, const StaticData* sd) { phase_p0(s, sd); }
LF_HD LF_NOINLINE inline void tick_p1(ArenaState* s, const StaticData* sd) { phase_p1(s, sd); }
LF_HD LF_NOINLINE inline void tick_p2(ArenaState* s, const StaticData* sd) { ddw_phase_p2(s, sd); }
LF_HD LF_NOINLINE inline void tick_p3a(ArenaState* s, const StaticData* sd) { update_aabbs(s, sd); }
// returns the number of pair algorithms reached that are not ported (0: the phase is complete)
LF_HD LF_NOINLINE inline uint32_t tick_p3b(ArenaState* s, const StaticData* sd) { return dispatch_all_collision_pairs(s, sd); }
LF_HD LF_NOINLINE inline void tick_p4(ArenaState* s, const StaticData* sd) { phase_p4(s, sd); }
LF_HD LF_NOINLINE inline void tick_p5(ArenaState* s, const StaticData* sd) { phase_p5(s, sd); }
LF_HD LF_NOINLINE inline void tick_p6(ArenaState* s, const StaticData* sd) { ddw_phase_p6(s, sd); }
LF_HD LF_NOINLINE inline void tick_p7(ArenaState* s, const StaticData* sd) { phase_p7(s, sd); }
LF_HD LF_NOINLINE inline void tick_p8(ArenaState* s, const StaticData* sd) { phase_p8(s, sd); }
LF_HD inline int phase(int k, ArenaState* s, const StaticData* sd) {
  switch (k) {
    case 0: tick_p0(s, sd); return PHASE_OK;
    case 1: tick_p1(s, sd); return PHASE_OK;
    case 2: tick_p2(s, sd); return PHASE_OK;
    case 3: tick_p3a(s, sd); return PHASE_OK;
    case 4: return tick_p3b(s, sd) == 0 ? PHASE_OK : PHASE_UNIMPLEMENTED;  // a blocked state
    case 5: tick_p4(s, sd); return PHASE_OK;
    case 6: tick_p5(s, sd); return PHASE_OK;
    case 7: tick_p6(s, sd); return PHASE_OK;
    case 8: tick_p7(s, sd); return PHASE_OK;
    case 9: tick_p8(s, sd); return PHASE_OK;
    default: return PHASE_BAD_INDEX;
  }
}

}  // namespace lfsim
