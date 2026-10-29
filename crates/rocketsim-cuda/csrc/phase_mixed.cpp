// The phase, step_tick and step_ticks functions built with a per-call choice of libm (LF_LIBM_MIXED, cuda/common/libm.cuh): a set bit of `mask` makes
// that function (1 sin, 2 cos, 4 tan, 8 atan2, 16 powf, 32 atan2_64, 64 hypot64) the platform library, the rest stay correctly rounded. The phase
// driver uses it to attribute each word that differs from V3 to the libm function that causes it (tests/common/mod.rs).
// Namespaces renamed as in phase_platform.cpp.
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define LF_LIBM_MIXED 1
#define lfsim lfsim_mixed
#define lf lf_mixed
#include "sim/arena_step.cuh"

using namespace lfsim_mixed;

extern "C" int rscuda_phase_mixed(int k, ArenaState* s, const StaticData* sd, unsigned mask) {
  libm_platform_mask = mask;
  return phase(k, s, sd);
}
extern "C" int rscuda_step_tick_mixed(ArenaState* s, const StaticData* sd, unsigned mask) {
  libm_platform_mask = mask;
  return step_tick(s, sd);
}
extern "C" int rscuda_step_ticks_mixed(ArenaState* s, const StaticData* sd, const Controls* controls, uint32_t n, uint32_t* scored_tick,
                                      ArenaState* trace, unsigned mask) {
  libm_platform_mask = mask;
  return step_ticks_to_goal(s, sd, controls, n, scored_tick, TraceCopy{trace});
}
