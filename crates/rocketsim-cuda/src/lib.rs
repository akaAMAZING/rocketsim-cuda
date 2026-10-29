//! The host side of the CUDA port of RocketSim V3: `ArenaState` / `StaticData` mirrors of `cuda/sim/arena_state.cuh`, the
//! importer and exporter of V3's state dump, the comparator, the AoSoA layout, the device plumbing, and the glam parity shim
//! (`cuda/common/glam.cuh` compiled for the CPU) with its device test kernels.

pub mod aosoa;
pub mod compare;
pub mod cuda;
pub mod device;
pub mod dump;
pub mod ffi;
pub mod layout;
pub mod libm;

use std::ffi::{CStr, c_char, c_int};

/// The device-side test kernels (`cuda/simtest/simtest.cu`, strict float policy).
pub const SIMTEST_FATBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/simtest.fatbin"));
/// The simulator harness kernels (`cuda/sim/sim.cu`, strict float policy): AoSoA identity round trip and `phase_k`.
pub const SIM_FATBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/sim.fatbin"));

unsafe extern "C" {
    fn lf_op_count() -> c_int;
    fn lf_op_name(op: c_int) -> *const c_char;
    fn lf_op_in(op: c_int) -> c_int;
    fn lf_op_out(op: c_int) -> c_int;
    fn lf_host_fma() -> c_int;
    fn lf_run(op: c_int, n: c_int, input: *const f32, out: *mut u32);
}

/// One op of the glam parity table: its name and the floats it reads and words it writes per case.
#[derive(Clone, Debug)]
pub struct Op {
    pub index: usize,
    pub name: String,
    pub in_n: usize,
    pub out_n: usize,
}

/// Every op of `cuda/simtest/ops.cuh`.
pub fn ops() -> Vec<Op> {
    // SAFETY: the shim's tables are static and `op` is in range.
    unsafe {
        (0..lf_op_count())
            .map(|i| Op {
                index: i as usize,
                name: CStr::from_ptr(lf_op_name(i)).to_string_lossy().into_owned(),
                in_n: lf_op_in(i) as usize,
                out_n: lf_op_out(i) as usize,
            })
            .collect()
    }
}

/// True when the host shim was compiled where the compiler may fuse multiply-add (forbidden for the reference).
pub fn shim_may_fuse() -> bool {
    // SAFETY: plain query.
    unsafe { lf_host_fma() != 0 }
}

/// Runs the C++ host build of `op` over `input` (`n * in_n` floats) into `out` (`n * out_n` words).
pub fn run_host(op: &Op, input: &[f32], out: &mut [u32]) {
    let n = input.len() / op.in_n;
    assert!(input.len() == n * op.in_n && out.len() == n * op.out_n);
    // SAFETY: the buffers hold exactly `n` cases of this op.
    unsafe { lf_run(op.index as c_int, n as c_int, input.as_ptr(), out.as_mut_ptr()) }
}
