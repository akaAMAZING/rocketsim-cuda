//! The host side of the libm tests (`csrc/libm.cpp`): the correctly rounded binary32 functions of `cuda/common/libm_cr.cuh`
//! and the platform C library beside them, over index ranges the device kernels (`simtest.cu`) draw identically.

use std::ffi::c_int;

unsafe extern "C" {
    fn lf_libm_unary(fnc: c_int, platform: c_int, base: u64, n: u64, out: *mut u32);
    fn lf_libm_stream(op: c_int, platform: c_int, base: u64, n: u64, out: *mut u32);
    fn lf_libm_input(op: c_int, idx: u64, a: *mut f32, b: *mut f32);
    fn lf_libm_f64(op: c_int, platform: c_int, base: u64, n: u64, out: *mut u64);
    fn lf_libm_autoflip(base: u64, n: u64, out: *mut u64);
    fn lf_libm_grid_len(fnc: c_int) -> c_int;
    fn lf_libm_grid(fnc: c_int, platform: c_int, out: *mut u32);
}

/// Names of the five functions, in the order of the `fn` codes (0 sin, 1 cos, 2 tan, 3 atan2, 4 pow).
pub const FUNCS: [&str; 5] = ["sinf", "cosf", "tanf", "atan2f", "powf"];

/// `out[i] = f(f32::from_bits(base + i))` for sin, cos, tan (`f` 0..3); the platform C library when `platform`.
pub fn unary(f: usize, platform: bool, base: u64, out: &mut [u32]) {
    assert!(f < 3);
    // SAFETY: `out` holds `n` words.
    unsafe { lf_libm_unary(f as c_int, platform as c_int, base, out.len() as u64, out.as_mut_ptr()) }
}

/// `out[i] = op(case base + i of the random stream)`, `op` 0 atan2f, 1 powf.
pub fn stream(op: usize, platform: bool, base: u64, out: &mut [u32]) {
    assert!(op < 2);
    // SAFETY: `out` holds `n` words.
    unsafe { lf_libm_stream(op as c_int, platform as c_int, base, out.len() as u64, out.as_mut_ptr()) }
}

/// The two arguments of case `idx` of the random stream of `op`.
pub fn stream_input(op: usize, idx: u64) -> (f32, f32) {
    let (mut a, mut b) = (0.0, 0.0);
    // SAFETY: plain query.
    unsafe { lf_libm_input(op as c_int, idx, &mut a, &mut b) };
    (a, b)
}

/// The special-value grid of function `f` (0..5): one result per special (sin, cos, tan) or per ordered pair of specials.
pub fn grid(f: usize, platform: bool) -> Vec<u32> {
    // SAFETY: the buffer has the length the shim reports.
    unsafe {
        let mut out = vec![0u32; lf_libm_grid_len(f as c_int) as usize];
        lf_libm_grid(f as c_int, platform as c_int, out.as_mut_ptr());
        out
    }
}

/// The f64 functions of the auto-flip path on the host: `op` 0 atan2_64, 1 hypot64; the C library when `platform`.
pub fn f64_stream(op: usize, platform: bool, base: u64, out: &mut [u64]) {
    assert!(op < 2);
    // SAFETY: `out` holds `n` words.
    unsafe { lf_libm_f64(op as c_int, platform as c_int, base, out.len() as u64, out.as_mut_ptr()) }
}

/// Counts over `n` samples of the auto-flip roll sign's inputs: [samples, atan2 doubles changed, atan2 floats changed, hypot
/// doubles changed, hypot floats changed, yaw ints, pitch ints, roll ints changed] when atan2/hypot are the port's instead of
/// the platform's.
pub fn autoflip_changes(base: u64, n: u64) -> [u64; 8] {
    let mut out = [0u64; 8];
    // SAFETY: `out` holds eight words.
    unsafe { lf_libm_autoflip(base, n, out.as_mut_ptr()) };
    out
}
