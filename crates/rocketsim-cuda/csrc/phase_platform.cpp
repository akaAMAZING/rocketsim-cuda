// The phase, step_tick and step_ticks functions (cuda/sim/phase.cuh, arena_step.cuh) built a second time, with the platform C library behind the lf:: names
// (LF_LIBM_PLATFORM: the porting verification against V3 on one platform). The default build (phase.cpp, rscuda_phase) uses the
// correctly rounded libm. Both live in one binary, so everything that depends on the libm choice is renamed here: the
// namespaces lfsim (the physics) and lf (the libm names) become lfsim_platform and lf_platform for this translation unit,
// otherwise the linker would fold the identically named inline functions of the two builds into one.
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define LF_LIBM_PLATFORM 1
#define lfsim lfsim_platform
#define lf lf_platform
#include "sim/arena_step.cuh"

using namespace lfsim_platform;

extern "C" int rscuda_phase_platform(int k, ArenaState* s, const StaticData* sd) { return phase(k, s, sd); }
extern "C" int rscuda_step_tick_platform(ArenaState* s, const StaticData* sd) { return step_tick(s, sd); }
extern "C" int rscuda_step_ticks_platform(ArenaState* s, const StaticData* sd, const Controls* controls, uint32_t n, uint32_t* scored_tick,
                                         ArenaState* trace) {
  return step_ticks_to_goal(s, sd, controls, n, scored_tick, TraceCopy{trace});
}
