// V3 sim/arena/base.rs around the tick: step_tick (the ten phases in order), set_car_controls,
// is_ball_scored, and the decision loop a caller drives (step_ticks: controls set before each tick, the decision's result
// the state after the scoring tick). One source for the host C ABI (crates/rocketsim-cuda/csrc/phase*.cpp) and the device kernels
// step_tick_k, sim8 and sim8_trace (sim.cu).
#pragma once
#include "sim/phase.cuh"

namespace lfsim {

constexpr uint32_t TICKS_PER_DECISION = 8;  // the controls are held for 8 ticks (TICK_SKIP = 8)
// sim8's result word per arena: bits 0-7 the scored tick (0 none, else the tick 1..8 of the decision after which is_ball_scored
// was first true; the state written back is the state after it), bit 31 set when a tick up to then reached a P3b shape pair the
// port does not implement (none in Soccar 1v1; the arena's state is then not V3's), bit 30 set when an overflow counter of the
// state written back is nonzero (a cap was hit: the state is not V3's; the counters only grow, so the bit stays set until the
// arena's state is replaced).
constexpr uint32_t SIM8_SCORED_MASK = 0xFFu;
constexpr uint32_t SIM8_UNIMPLEMENTED = 0x80000000u;
constexpr uint32_t SIM8_OVERFLOW = 0x40000000u;

// Any overflow counter nonzero (SIM8_OVERFLOW).
LF_HDI bool any_overflow(const ArenaState* s) {
  uint32_t o = 0;
  for (int i = 0; i < OVF_COUNT; ++i) o |= s->overflow[i];
  return o != 0;
}

// V3: arena/base.rs:1020-1022 set_car_controls (the bool fields are 0 or 1, as V3's bool)
LF_HDI void set_car_controls(ArenaState* s, int car_idx, const Controls& controls) { s->cars[car_idx].controls = controls; }

// V3: arena/base.rs:404-425 is_ball_scored, the Soccar arm: the ball body's translation in uu against the goal threshold
LF_HDI bool is_ball_scored(const ArenaState* s, const StaticData* sd) {
  Vec3A ball_pos = s->bodies[0].world_trans.translation * c::BT_TO_UU;
  return lf_abs(ball_pos.y) > lf_add(sd->goal_base_threshold_y, sd->mut_f[c::MF_BALL_RADIUS]);
}

// V3: arena/base.rs:611-777 step_tick: P0 .. P8 in V3's order; the same functions phase(k) runs). Events,
// the visualiser and Dropshot's tile update are not Soccar 1v1 state. Returns PHASE_OK, or PHASE_UNIMPLEMENTED when P3b reached a
// shape pair the port does not implement (the tick still completes; its state is then not V3's).
LF_HD inline int step_tick(ArenaState* s, const StaticData* sd) {
  tick_p0(s, sd);                                   // k 0, base.rs:612-641
  tick_p1(s, sd);                                   // k 1, base.rs:643-673
  tick_p2(s, sd);                                   // k 2, ddw:247 and ddw:215 (step_simulation from here)
  tick_p3a(s, sd);                                  // k 3, collision_world.rs:121
  uint32_t unimplemented = tick_p3b(s, sd);         // k 4, collision_world.rs:123
  tick_p4(s, sd);                                   // k 5, ddw:220-230
  tick_p5(s, sd);                                   // k 6, ddw:232-237
  tick_p6(s, sd);                                   // k 7, ddw:238-239
  tick_p7(s, sd);                                   // k 8, base.rs:678-733
  tick_p8(s, sd);                                   // k 9, base.rs:735-767
  return unimplemented == 0 ? PHASE_OK : PHASE_UNIMPLEMENTED;
}

// Hooks of step_ticks, called as hook(t, s): none (production), a copy of the state after each tick (the host trace: `out[t - 1]` is
// the state after tick t), a copy of the goal state (the host). The device's are in sim.cu.
struct NoHook {
  LF_HDI void operator()(uint32_t, const ArenaState*) const {}
};
struct TraceCopy {
  ArenaState* out;  // null: no trace
  LF_HDI void operator()(uint32_t t, const ArenaState* s) const {
    if (out) out[t - 1] = *s;
  }
};
struct GoalCopy {
  ArenaState* out;
  LF_HDI void operator()(uint32_t, const ArenaState* s) const { *out = *s; }
};

// The decision loop: n ticks of V3's env loop, which ends after the tick on which is_ball_scored is first true. Before each tick
// set_car_controls(i, controls[i]) for both cars (the held action), then step_tick; with switch ticks kk (car i plays the
// controls IN FORCE at the call, those of its previous tick, for ticks 1..kk[i] and its new controls from tick kk[i] + 1 on;
// kk null or 0 = the new controls on every tick; a data read per tick, no call); after each tick up to the goal,
// after_tick(t, s), then is_ball_scored. *scored_tick is the tick of the goal (1..n), else 0; at that tick at_goal(t, s) receives
// the goal state, which is the decision's result. Every arena runs all n ticks: the ticks after a goal are stepped and discarded
// (no status, hook or goal check), so that on the device no lane leaves the loop while its warp-mates go on calling the phase
// functions. The rule for every kernel around step_ticks or the phases: n is the same for the whole warp, no lane leaves a loop
// of phase calls early and no lane skips a phase call its warp-mates make; the hooks make no call. An early exit here let the
// scoring lane run the code after the loop while its warp-mates were still ticking, and ptxas, which compiles that code as
// converged, moved a warp-shared register they were using. *s ends as the state after tick n,
// which is the result only when no goal was scored (step_ticks_to_goal keeps the goal state). Returns PHASE_OK, or
// PHASE_UNIMPLEMENTED when a tick up to the goal returned it. Not unrolled on the device: a fully unrolled build of the earlier
// early-exit loop failed with an illegal address on some arenas.
template <class AfterTick, class AtGoal>
LF_HD inline int step_ticks(ArenaState* s, const StaticData* sd, const Controls* controls, uint32_t n, uint32_t* scored_tick,
                            const AfterTick& after_tick, const AtGoal& at_goal, const uint32_t* kk = nullptr) {
  int status = PHASE_OK;
  uint32_t scored = 0;
  Controls hold[NUM_CARS];
  for (int i = 0; i < NUM_CARS; ++i) hold[i] = s->cars[i].controls;
  LF_NO_UNROLL
  for (uint32_t t = 1; t <= n; ++t) {
    for (int i = 0; i < NUM_CARS; ++i) set_car_controls(s, i, kk != nullptr && t <= kk[i] ? hold[i] : controls[i]);
    int tick_status = step_tick(s, sd);
    if (scored == 0) {
      if (tick_status != PHASE_OK) status = PHASE_UNIMPLEMENTED;
      after_tick(t, s);
      if (is_ball_scored(s, sd)) {
        scored = t;
        at_goal(t, s);
      }
    }
  }
  *scored_tick = scored;
  return status;
}

// step_ticks with the decision's result left in *s, the goal state when the ball was scored (the host C ABI,
// crates/rocketsim-cuda/csrc/phase*.cpp). Host only: a second ArenaState holds the goal state until the loop ends.
template <class AfterTick>
inline int step_ticks_to_goal(ArenaState* s, const StaticData* sd, const Controls* controls, uint32_t n, uint32_t* scored_tick,
                              const AfterTick& after_tick) {
  ArenaState goal;
  int status = step_ticks(s, sd, controls, n, scored_tick, after_tick, GoalCopy{&goal});
  if (*scored_tick != 0) *s = goal;
  return status;
}

}  // namespace lfsim
