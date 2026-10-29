//! AoSoA-32 packing of `ArenaState`s, written independently of the C++ `cuda/sim/aosoa.cuh` it is
//! checked against: arena `a` lives in block `a / 32`, word `w` of its state at `block * ARENA_WORDS * 32 + w * 32 + a % 32`.

use crate::layout::{ARENA_WORDS, ArenaState};

pub const LANES: usize = 32;

/// Words of a buffer for `n` arenas (the last block is padded to 32 lanes).
pub fn words(n: usize) -> usize {
    n.div_ceil(LANES) * ARENA_WORDS * LANES
}

pub fn index(a: usize, w: usize) -> usize {
    (a / LANES) * ARENA_WORDS * LANES + w * LANES + a % LANES
}

pub fn pack(states: &[Box<ArenaState>]) -> Vec<u32> {
    let mut buf = vec![0u32; words(states.len())];
    for (a, s) in states.iter().enumerate() {
        for (w, &x) in s.words().iter().enumerate() {
            buf[index(a, w)] = x;
        }
    }
    buf
}

pub fn unpack(buf: &[u32], n: usize) -> Vec<Box<ArenaState>> {
    assert_eq!(buf.len(), words(n));
    (0..n)
        .map(|a| {
            let mut s = ArenaState::zeroed();
            for (w, x) in s.words_mut().iter_mut().enumerate() {
                *x = buf[index(a, w)];
            }
            s
        })
        .collect()
}
