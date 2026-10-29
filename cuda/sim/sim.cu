// Device side of the simulator (strict float policy, built by rocketsim-cuda's build.rs): the AoSoA copy-in / copy-out identity round
// trip, the per-phase test kernel, the full tick (step_tick_k), and the decision kernels sim8 and sim8_trace.
#include "sim/aosoa.cuh"
#include "sim/arena_step.cuh"

using namespace lfsim;

// Copy arena a of `in` into a local ArenaState and back into `out`: byte-identical or the layout is wrong.
extern "C" __global__ void sim_identity(const uint32_t* in, uint32_t* out, uint32_t n) {
  uint32_t a = blockIdx.x * blockDim.x + threadIdx.x;
  if (a >= n) return;
  ArenaState s;
  load_state(&s, in, a);
  store_state(out, a, &s);
}

// The tick-boundary variant: only the persistent part of the state moves.
extern "C" __global__ void sim_identity_tick(const uint32_t* in, uint32_t* out, uint32_t n) {
  uint32_t a = blockIdx.x * blockDim.x + threadIdx.x;
  if (a >= n) return;
  ArenaState s;
  load_state_tick(&s, in, a);
  store_state_tick(out, a, &s);
}

// Runs phase k on every arena of the buffer in place; status[a] is the PhaseStatus of arena a.
extern "C" __global__ void __launch_bounds__(128) phase_k(int k, uint32_t* buf, const StaticData* sd, uint32_t n, int32_t* status) {
  uint32_t a = blockIdx.x * blockDim.x + threadIdx.x;
  if (a >= n) return;
  ArenaState s;
  load_state(&s, buf, a);
  status[a] = phase(k, &s, sd);
  store_state(buf, a, &s);
}

// Largest block of the tick kernels. One thread per arena, no shared memory, no synchronisation: no thread touches another's
// arena, so the block size cannot change a result (tests/step.rs launches 32 and 64 and permutes the arenas). The host launches
// 32 (crates/rocketsim-cuda/src/device.rs SIM8_BLOCK): one warp is then one AoSoA-32 block, whose copy-in and copy-out are one
// contiguous run of 32 words per state word and never straddle two AoSoA blocks. 64 is allowed.
// __launch_bounds__(64, 1) leaves the compiler the full 255 registers per thread (64 x 255 fits one SM's
// register file), which the solver needs.
constexpr int SIM_MAX_BLOCK = 64;

// One full tick (V3 step_tick) of every arena in place, at tick boundaries (tick copies: the mid-tick part is zero between ticks);
// status[a] is step_tick's PhaseStatus. The controls are those already in the state.
extern "C" __global__ void __launch_bounds__(SIM_MAX_BLOCK, 1) step_tick_k(uint32_t* buf, const StaticData* sd, uint32_t n, int32_t* status) {
  uint32_t a = blockIdx.x * blockDim.x + threadIdx.x;
  if (a >= n) return;
  ArenaState s;
  load_state_tick(&s, buf, a);
  status[a] = step_tick(&s, sd);
  store_state_tick(buf, a, &s);
}

// sim8_trace's hook: the tick-boundary words after tick t into AoSoA buffer t - 1 of `trace` (TICKS_PER_DECISION buffers of
// aosoa_words(n) words, back to back).
struct TraceStore {
  uint32_t* trace;
  size_t words;
  uint32_t a;
  LF_HDI void operator()(uint32_t t, const ArenaState* s) const { store_state_tick(trace + (size_t)(t - 1) * words, a, s); }
};

// sim8's goal hook: the goal state's tick-boundary words into arena a of the AoSoA buffer, the decision's result (the ticks the
// loop runs after the goal are discarded), and whether that state overflowed.
struct GoalStore {
  uint32_t* states;
  uint32_t a;
  bool* overflow;
  LF_HDI void operator()(uint32_t, const ArenaState* s) const {
    store_state_tick(states, a, s);
    *overflow = any_overflow(s);
  }
};

// One decision of arena a: copy-in, step_ticks over TICKS_PER_DECISION ticks with the arena's controls, copy-out (at the goal for
// a scored arena), result word. Every lane runs all the ticks (arena_step.cuh step_ticks), so the warp calls each phase together.
template <class AfterTick>
__device__ __forceinline__ void sim8_arena(uint32_t* states, const StaticData* sd, const Controls* controls, uint32_t* result, uint32_t a,
                                           const AfterTick& after_tick) {
  ArenaState s;
  load_state_tick(&s, states, a);
  Controls ctl[NUM_CARS];
  for (int i = 0; i < NUM_CARS; ++i) ctl[i] = controls[(size_t)a * NUM_CARS + i];
  uint32_t scored_tick;
  bool overflow = false;
  int status = step_ticks(&s, sd, ctl, TICKS_PER_DECISION, &scored_tick, after_tick, GoalStore{states, a, &overflow});
  if (scored_tick == 0) {
    store_state_tick(states, a, &s);
    overflow = any_overflow(&s);
  }
  result[a] = scored_tick | (status == PHASE_OK ? 0u : SIM8_UNIMPLEMENTED) | (overflow ? SIM8_OVERFLOW : 0u);
}

// The decision kernel: TICKS_PER_DECISION ticks of every arena of the AoSoA-32 buffer `states` in place, one thread per arena.
// controls[a * NUM_CARS + i] are car i's controls in arena a, set before every tick; a decision ends after the tick on which
// is_ball_scored is true, and that tick's state is the one written back; result[a] is its SIM8_* word. The overflow counters ride
// in the state.
extern "C" __global__ void __launch_bounds__(SIM_MAX_BLOCK, 1) sim8(uint32_t* states, const StaticData* sd, const Controls* controls,
                                                                    uint32_t* result, uint32_t n) {
  uint32_t a = blockIdx.x * blockDim.x + threadIdx.x;
  if (a >= n) return;
  sim8_arena(states, sd, controls, result, a, NoHook{});
}

// sim8 that also writes the state after each tick (for tests). Ticks after the scored tick are not written.
extern "C" __global__ void __launch_bounds__(SIM_MAX_BLOCK, 1) sim8_trace(uint32_t* states, const StaticData* sd, const Controls* controls,
                                                                          uint32_t* result, uint32_t n, uint32_t* trace) {
  uint32_t a = blockIdx.x * blockDim.x + threadIdx.x;
  if (a >= n) return;
  sim8_arena(states, sd, controls, result, a, TraceStore{trace, aosoa_words(n), a});
}
