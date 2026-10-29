// Host side of the phase harness: the C ABI over the same single-source phase, step_tick and step_ticks functions the device kernels
// run, and the AoSoA pack/unpack helpers (aosoa.cuh) for the cross-check against Rust's independent implementation.
#include "sim/aosoa.cuh"
#include "sim/arena_step.cuh"

using namespace lfsim;

extern "C" {
// Runs phase k on one arena; returns a PhaseStatus.
int rscuda_phase(int k, ArenaState* s, const StaticData* sd) { return phase(k, s, sd); }
// One full tick (V3 step_tick); returns a PhaseStatus.
int rscuda_step_tick(ArenaState* s, const StaticData* sd) { return step_tick(s, sd); }
// One decision: n ticks with controls[0..NUM_CARS) set before each; *s ends as the state after the scoring tick (*scored_tick, 0 for
// none) or after tick n; trace: null or n states, trace[t - 1] = the state after tick t up to the goal. Returns a PhaseStatus.
int rscuda_step_ticks(ArenaState* s, const StaticData* sd, const Controls* controls, uint32_t n, uint32_t* scored_tick, ArenaState* trace) {
  return step_ticks_to_goal(s, sd, controls, n, scored_tick, TraceCopy{trace});
}
uint64_t lf_arena_words() { return ARENA_WORDS; }
uint64_t lf_static_words() { return STATIC_WORDS; }
uint64_t lf_tick_words() { return TICK_WORDS; }
uint64_t lf_aosoa_words(uint32_t n) { return aosoa_words(n); }
uint64_t lf_aosoa_index(uint32_t a, int w) { return aosoa_index(a, w); }

// Copy-in then copy-out through a local ArenaState, arena by arena (the host twin of sim_identity).
void lf_aosoa_identity(const uint32_t* in, uint32_t* out, uint32_t n) {
  ArenaState s;
  for (uint32_t a = 0; a < n; ++a) {
    load_state(&s, in, a);
    store_state(out, a, &s);
  }
}

void lf_aosoa_identity_tick(const uint32_t* in, uint32_t* out, uint32_t n) {
  ArenaState s;
  for (uint32_t a = 0; a < n; ++a) {
    load_state_tick(&s, in, a);
    store_state_tick(out, a, &s);
  }
}
}
