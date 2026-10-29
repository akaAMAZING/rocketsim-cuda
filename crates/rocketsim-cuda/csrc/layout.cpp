// Host side of the layout test: offset and size of every field of every struct of arena_state.cuh, as the C++ compiler
// lays them out, for tests/layout.rs to compare with the Rust mirrors (layout.rs).
#include <stddef.h>
#include <string.h>
#include "sim/arena_state.cuh"

using namespace lfsim;

namespace {
struct Row { const char* rust; const char* field; size_t off, size; };
struct Struct { const char* rust; size_t size, align; };
const Row kRows[] = {
#define STRUCT(rust, cpp)
#define ROW(rust, cpp, f) {#rust, #f, offsetof(cpp, f), sizeof(((cpp*)0)->f)},
#include "layout_fields.inc"
#undef ROW
#undef STRUCT
};
const Struct kStructs[] = {
#define ROW(rust, cpp, f)
#define STRUCT(rust, cpp) {#rust, sizeof(cpp), alignof(cpp)},
#include "layout_fields.inc"
#undef ROW
#undef STRUCT
    {"Opt<u32>", sizeof(Opt<uint32_t>), alignof(Opt<uint32_t>)},
    {"Opt<Vec3>", sizeof(Opt<Vec3A>), alignof(Opt<Vec3A>)},
    {"Opt<U64>", sizeof(Opt<U64>), alignof(Opt<U64>)},
    {"Opt<HitInfo>", sizeof(Opt<HitInfo>), alignof(Opt<HitInfo>)},
    {"Opt<RaycastInfo>", sizeof(Opt<RaycastInfo>), alignof(Opt<RaycastInfo>)},
};
}  // namespace

extern "C" {
size_t lf_layout_rows() { return sizeof(kRows) / sizeof(kRows[0]); }
const char* lf_layout_row(size_t i, const char** field, size_t* off, size_t* size) {
  *field = kRows[i].field; *off = kRows[i].off; *size = kRows[i].size;
  return kRows[i].rust;
}
size_t lf_layout_structs() { return sizeof(kStructs) / sizeof(kStructs[0]); }
const char* lf_layout_struct(size_t i, size_t* size, size_t* align) {
  *size = kStructs[i].size; *align = kStructs[i].align;
  return kStructs[i].rust;
}
// Offsets inside Opt<T> (some, v) for the one instantiation with the largest alignment demand.
size_t lf_opt_v_offset() { return offsetof(Opt<Vec3A>, v); }
}
