//! The C++ host shim of the harness (csrc/layout.cpp, csrc/phase*.cpp): layout rows, the phase and tick C ABI, the AoSoA twin.

use std::ffi::{CStr, c_char, c_int};

use crate::layout::{ArenaState, Controls, NUM_CARS, StaticData};

unsafe extern "C" {
    fn lf_layout_rows() -> usize;
    fn lf_layout_row(i: usize, field: *mut *const c_char, off: *mut usize, size: *mut usize) -> *const c_char;
    fn lf_layout_structs() -> usize;
    fn lf_layout_struct(i: usize, size: *mut usize, align: *mut usize) -> *const c_char;
    fn rscuda_phase(k: c_int, s: *mut ArenaState, sd: *const StaticData) -> c_int;
    fn rscuda_phase_platform(k: c_int, s: *mut ArenaState, sd: *const StaticData) -> c_int;
    fn rscuda_phase_mixed(k: c_int, s: *mut ArenaState, sd: *const StaticData, mask: u32) -> c_int;
    fn rscuda_step_tick(s: *mut ArenaState, sd: *const StaticData) -> c_int;
    fn rscuda_step_tick_platform(s: *mut ArenaState, sd: *const StaticData) -> c_int;
    fn rscuda_step_tick_mixed(s: *mut ArenaState, sd: *const StaticData, mask: u32) -> c_int;
    fn rscuda_step_ticks(s: *mut ArenaState, sd: *const StaticData, controls: *const Controls, n: u32, scored_tick: *mut u32, trace: *mut ArenaState) -> c_int;
    fn rscuda_step_ticks_platform(s: *mut ArenaState, sd: *const StaticData, controls: *const Controls, n: u32, scored_tick: *mut u32, trace: *mut ArenaState) -> c_int;
    fn rscuda_step_ticks_mixed(s: *mut ArenaState, sd: *const StaticData, controls: *const Controls, n: u32, scored_tick: *mut u32, trace: *mut ArenaState, mask: u32) -> c_int;
    fn lf_arena_words() -> u64;
    fn lf_static_words() -> u64;
    fn lf_aosoa_words(n: u32) -> u64;
    fn lf_aosoa_index(a: u32, w: c_int) -> u64;
    fn lf_aosoa_identity(input: *const u32, out: *mut u32, n: u32);
    fn lf_aosoa_identity_tick(input: *const u32, out: *mut u32, n: u32);
    fn lf_tick_words() -> u64;
}

/// `(rust struct name, field, offset, size)` of every field the C++ compiler laid out.
pub fn layout_rows() -> Vec<(String, String, usize, usize)> {
    // SAFETY: the tables are static; the returned strings are static NUL-terminated literals.
    unsafe {
        (0..lf_layout_rows())
            .map(|i| {
                let (mut f, mut off, mut size) = (std::ptr::null(), 0, 0);
                let s = lf_layout_row(i, &mut f, &mut off, &mut size);
                (CStr::from_ptr(s).to_string_lossy().into_owned(), CStr::from_ptr(f).to_string_lossy().into_owned(), off, size)
            })
            .collect()
    }
}

/// `(name, size, alignment)` of every struct, plus the `Opt<..>` instantiations.
pub fn layout_structs() -> Vec<(String, usize, usize)> {
    // SAFETY: as `layout_rows`.
    unsafe {
        (0..lf_layout_structs())
            .map(|i| {
                let (mut size, mut align) = (0, 0);
                let s = lf_layout_struct(i, &mut size, &mut align);
                (CStr::from_ptr(s).to_string_lossy().into_owned(), size, align)
            })
            .collect()
    }
}

/// `rscuda_phase`: runs phase `k` of the tick on one arena (status -1: a shape pair the port does not implement was reached).
pub fn phase_host(k: usize, s: &mut ArenaState, sd: &StaticData) -> i32 {
    // SAFETY: both pointers come from live references.
    unsafe { rscuda_phase(k as c_int, s, sd) }
}

/// `rscuda_phase_platform`: the same phase function built with the platform C library behind `lf::sin` ... (LF_LIBM_PLATFORM),
/// the porting verification against V3 on one platform. `phase_host` is the correctly rounded build.
pub fn phase_host_platform(k: usize, s: &mut ArenaState, sd: &StaticData) -> i32 {
    // SAFETY: both pointers come from live references.
    unsafe { rscuda_phase_platform(k as c_int, s, sd) }
}

/// `rscuda_phase_mixed`: the phase function with a per-call libm choice: bit i of `mask` (0 sin, 1 cos, 2 tan, 3 atan2, 4 powf, 5 atan2_64, 6 hypot64)
/// set makes that function the platform library, the others stay correctly rounded. Mask 0 is `phase_host`, mask 127 is
/// `phase_host_platform`.
pub fn phase_host_mixed(k: usize, s: &mut ArenaState, sd: &StaticData, mask: u32) -> i32 {
    // SAFETY: both pointers come from live references.
    unsafe { rscuda_phase_mixed(k as c_int, s, sd, mask) }
}

/// Which libm a host build of the tick uses (cuda/common/libm.cuh): the correctly rounded default (`rscuda_step_tick`, equal to the
/// device bit for bit), the platform C library as V3 calls it (`rscuda_step_tick_platform`, the porting verification), or the
/// per-call mask of `rscuda_step_tick_mixed` (bit i: 0 sin, 1 cos, 2 tan, 3 atan2, 4 powf, 5 atan2_64, 6 hypot64 on the platform library).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Libm {
    Cr,
    Platform,
    Mixed(u32),
}

/// One full tick (V3 `Arena::step_tick`, cuda/sim/arena_step.cuh) with the controls already in the state; returns a PhaseStatus.
pub fn step_tick_host(libm: Libm, s: &mut ArenaState, sd: &StaticData) -> i32 {
    // SAFETY: both pointers come from live references.
    unsafe {
        match libm {
            Libm::Cr => rscuda_step_tick(s, sd),
            Libm::Platform => rscuda_step_tick_platform(s, sd),
            Libm::Mixed(mask) => rscuda_step_tick_mixed(s, sd, mask),
        }
    }
}

/// `step_ticks`: one decision of `n` ticks with `controls` set before each (V3 `set_car_controls`), which ends after the tick on
/// which V3's `is_ball_scored` is true: `s` is left at that tick's state. Returns (PhaseStatus, scored tick: 1..=n, 0 for none)
/// and, when `trace` is set, the state after each tick up to the goal.
pub fn step_ticks_host(libm: Libm, s: &mut ArenaState, sd: &StaticData, controls: &[Controls; NUM_CARS], n: u32, trace: bool) -> (i32, u32, Vec<ArenaState>) {
    let mut t: Vec<ArenaState> = if trace { vec![*ArenaState::zeroed(); n as usize] } else { Vec::new() };
    let tp = if trace { t.as_mut_ptr() } else { std::ptr::null_mut() };
    let mut scored = 0u32;
    // SAFETY: the pointers come from live references; `t` holds n states when tracing (the C side writes the ticks that ran).
    let st = unsafe {
        match libm {
            Libm::Cr => rscuda_step_ticks(s, sd, controls.as_ptr(), n, &mut scored, tp),
            Libm::Platform => rscuda_step_ticks_platform(s, sd, controls.as_ptr(), n, &mut scored, tp),
            Libm::Mixed(mask) => rscuda_step_ticks_mixed(s, sd, controls.as_ptr(), n, &mut scored, tp, mask),
        }
    };
    t.truncate((if scored > 0 { scored } else { n }) as usize);
    (st, scored, t)
}

pub fn cpp_arena_words() -> usize {
    // SAFETY: plain query.
    unsafe { lf_arena_words() as usize }
}
pub fn cpp_static_words() -> usize {
    // SAFETY: plain query.
    unsafe { lf_static_words() as usize }
}
pub fn cpp_aosoa_words(n: u32) -> usize {
    // SAFETY: plain query.
    unsafe { lf_aosoa_words(n) as usize }
}
pub fn cpp_aosoa_index(a: u32, w: usize) -> usize {
    // SAFETY: plain query.
    unsafe { lf_aosoa_index(a, w as c_int) as usize }
}
/// The host twin of the device identity kernel: copy-in then copy-out of each of `n` arenas through a local state.
pub fn cpp_aosoa_identity(input: &[u32], n: u32) -> Vec<u32> {
    assert_eq!(input.len(), cpp_aosoa_words(n));
    let mut out = vec![0u32; input.len()];
    // SAFETY: both buffers hold `aosoa_words(n)` words.
    unsafe { lf_aosoa_identity(input.as_ptr(), out.as_mut_ptr(), n) };
    out
}

/// The host twin of `sim_identity_tick`: only the tick-boundary words move.
pub fn cpp_aosoa_identity_tick(input: &[u32], n: u32) -> Vec<u32> {
    assert_eq!(input.len(), cpp_aosoa_words(n));
    let mut out = vec![0u32; input.len()];
    // SAFETY: both buffers hold `aosoa_words(n)` words.
    unsafe { lf_aosoa_identity_tick(input.as_ptr(), out.as_mut_ptr(), n) };
    out
}

pub fn cpp_tick_words() -> usize {
    // SAFETY: plain query.
    unsafe { lf_tick_words() as usize }
}
