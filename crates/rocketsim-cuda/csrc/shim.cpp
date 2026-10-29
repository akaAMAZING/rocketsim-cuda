// Host side of the glam parity test: the shared headers compiled for the CPU behind a C ABI. Built by build.rs with
// MSVC /fp:precise or gcc/clang -ffp-contract=off, and no FMA target (lf_host_fma() reports what the compiler allows).
#include "simtest/ops.cuh"

namespace {
enum {
#define X(name, in_n, out_n, ...) OP_##name,
  LF_OPS(X)
#undef X
  OP_COUNT
};
struct Info { const char* name; int in_n, out_n; };
const Info kInfo[] = {
#define X(name, in_n, out_n, ...) {#name, in_n, out_n},
    LF_OPS(X)
#undef X
};
}  // namespace

extern "C" {
int lf_op_count() { return OP_COUNT; }
const char* lf_op_name(int op) { return kInfo[op].name; }
int lf_op_in(int op) { return kInfo[op].in_n; }
int lf_op_out(int op) { return kInfo[op].out_n; }
int lf_host_fma() { return LF_HOST_FMA; }

// Runs `n` cases of op `op`: case k reads in[k * in_n ..] and writes out[k * out_n ..].
void lf_run(int op, int n, const float* in, uint32_t* out) {
  const int in_n = kInfo[op].in_n, out_n = kInfo[op].out_n;
  for (int k = 0; k < n; ++k) {
    const float* i = in + (size_t)k * in_n;
    uint32_t* o = out + (size_t)k * out_n;
    switch (op) {
#define X(name, a, b, ...) \
  case OP_##name: {        \
    __VA_ARGS__;           \
    break;                 \
  }
      LF_OPS(X)
#undef X
      default: break;
    }
  }
}
}
