// V3 sim/arena/base.rs: phase P1 (k = 1), the car pre-tick of the tick: the flip z-damp prepass over the cars, car
// pre_tick_update in index order, then the ball's pre_tick_update and the ball_only flags.
#pragma once
#include "sim/car_pre_tick.cuh"

namespace lfsim {

// V3: sim/arena/base.rs:643-673 (patched step_phase arm 1, base.rs:1747); CarLanded goes into ArenaState.events. Soccar has no ball
// pre-tick work (ball/base.rs:141 pre_tick_update does nothing for GameMode::Soccar) and ball_only is false: the flags it sets
// are not in ArenaState.
LF_HD inline void phase_p1(ArenaState* s, const StaticData* sd) {
  // Apply flip damping before another car's wheels read its velocity.
  for (int ci = 0; ci < NUM_CARS; ++ci) car_apply_flip_zdamp_prepass(s, sd, ci);

  for (int ci = 0; ci < NUM_CARS; ++ci) {
    uint32_t was_on_ground = s->cars[ci].is_on_ground;
    car_pre_tick_update(s, sd, ci);
    if (!was_on_ground && s->cars[ci].is_on_ground) s->events |= 1u << (EV_LANDED + ci);  // CarLanded (base.rs:658)
  }
}

}  // namespace lfsim
