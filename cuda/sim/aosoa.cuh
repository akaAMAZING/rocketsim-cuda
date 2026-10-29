// AoSoA-32 layout of N arenas and the copy-in / copy-out between it and a thread-local
// ArenaState. Arena a lives in block a / 32; word w of its ArenaState (a 4-byte word, ArenaState is an array of them)
// sits at block_base + w * 32 + a % 32, so the 32 lanes of a warp that copy their arenas touch 32 consecutive words.
// One source for the host (checked against Rust's independent pack/unpack) and the device.
#pragma once
#include "sim/arena_state.cuh"

namespace lfsim {

constexpr uint32_t AOSOA_LANES = 32;

// Words of one block (32 arenas).
LF_HDI size_t aosoa_block_words() { return (size_t)ARENA_WORDS * AOSOA_LANES; }
// Words of a buffer for n arenas (the last block is padded to 32 lanes).
LF_HDI size_t aosoa_words(uint32_t n) { return (size_t)((n + AOSOA_LANES - 1) / AOSOA_LANES) * aosoa_block_words(); }
// Index of word w of arena a.
LF_HDI size_t aosoa_index(uint32_t a, int w) {
  return (size_t)(a / AOSOA_LANES) * aosoa_block_words() + (size_t)w * AOSOA_LANES + (a % AOSOA_LANES);
}

// Copy-in: arena a of the buffer into a local ArenaState.
LF_HDI void load_state(ArenaState* dst, const uint32_t* buf, uint32_t a) {
  uint32_t* d = reinterpret_cast<uint32_t*>(dst);
  for (int w = 0; w < ARENA_WORDS; ++w) d[w] = buf[aosoa_index(a, w)];
}
// Tick-boundary copies: only the first TICK_WORDS words move (the mid-tick part is zero between ticks); copy-in zero-fills the rest.
LF_HDI void load_state_tick(ArenaState* dst, const uint32_t* buf, uint32_t a) {
  uint32_t* d = reinterpret_cast<uint32_t*>(dst);
  for (int w = 0; w < TICK_WORDS; ++w) d[w] = buf[aosoa_index(a, w)];
  for (int w = TICK_WORDS; w < ARENA_WORDS; ++w) d[w] = 0;
}
LF_HDI void store_state_tick(uint32_t* buf, uint32_t a, const ArenaState* src) {
  const uint32_t* s = reinterpret_cast<const uint32_t*>(src);
  for (int w = 0; w < TICK_WORDS; ++w) buf[aosoa_index(a, w)] = s[w];
}

// Copy-out: a local ArenaState into arena a of the buffer.
LF_HDI void store_state(uint32_t* buf, uint32_t a, const ArenaState* src) {
  const uint32_t* s = reinterpret_cast<const uint32_t*>(src);
  for (int w = 0; w < ARENA_WORDS; ++w) buf[aosoa_index(a, w)] = s[w];
}

}  // namespace lfsim
