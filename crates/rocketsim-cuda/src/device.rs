//! Device plumbing: `StaticData` uploaded once, `ArenaState`s in AoSoA-32 buffers, the identity round trip (`sim_identity`), the
//! per-phase test kernel (`phase_k`), the full tick (`step_tick_k`) and the decision kernel (`sim8`, `sim8_trace`) of
//! `cuda/sim/sim.cu`.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, DriverError, LaunchConfig, PushKernelArg};
use crate::cuda::{context, load};

use crate::{
    SIM_FATBIN, aosoa,
    layout::{ArenaState, Controls, NUM_CARS, StaticData},
};

/// Ticks of one decision (cuda/sim/arena_step.cuh TICKS_PER_DECISION).
pub const TICKS_PER_DECISION: usize = 8;
/// sim8's result word per arena: the scored tick (0 none, else the tick 1..=8 after which `is_ball_scored` was first true; the
/// state written back is the state after it) ...
pub const SIM8_SCORED_MASK: u32 = 0xFF;
/// ... this bit when a tick up to then reached a P3b shape pair the port does not implement (the state is then not V3's) ...
pub const SIM8_UNIMPLEMENTED: u32 = 0x8000_0000;
/// ... and this bit when an overflow counter of the state written back is nonzero (a cap was hit: the state is not V3's; the
/// counters only grow, so it stays set until the arena's state is replaced).
pub const SIM8_OVERFLOW: u32 = 0x4000_0000;
/// Threads per block of the tick kernels in production: one warp per AoSoA-32 block (sim.cu, SIM_MAX_BLOCK).
pub const SIM8_BLOCK: u32 = 32;
/// The largest block the tick kernels accept (`__launch_bounds__(64, 1)`).
pub const SIM_MAX_BLOCK: u32 = 64;

pub struct SimDevice {
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    identity: CudaFunction,
    identity_tick: CudaFunction,
    phase: CudaFunction,
    step_tick: CudaFunction,
    sim8: CudaFunction,
    sim8_trace: CudaFunction,
}

/// N arenas in one AoSoA-32 buffer on the device.
pub struct DeviceStates {
    pub buf: CudaSlice<u32>,
    pub n: usize,
}

/// `StaticData` on the device (uploaded once).
pub struct DeviceStatic {
    pub buf: CudaSlice<u8>,
}

impl SimDevice {
    pub fn new(ordinal: usize) -> Result<Self, DriverError> {
        let ctx = context(ordinal)?;
        let stream = ctx.new_stream()?;
        let module = load(&ctx, SIM_FATBIN)?;
        Ok(Self {
            identity: module.load_function("sim_identity")?,
            identity_tick: module.load_function("sim_identity_tick")?,
            phase: module.load_function("phase_k")?,
            step_tick: module.load_function("step_tick_k")?,
            sim8: module.load_function("sim8")?,
            sim8_trace: module.load_function("sim8_trace")?,
            ctx,
            stream,
        })
    }

    pub fn upload_static(&self, sd: &StaticData) -> Result<DeviceStatic, DriverError> {
        Ok(DeviceStatic { buf: self.stream.clone_htod(sd.as_bytes())? })
    }

    /// Packs the states into AoSoA-32 (host side) and uploads them.
    pub fn upload_states(&self, states: &[Box<ArenaState>]) -> Result<DeviceStates, DriverError> {
        Ok(DeviceStates { buf: self.stream.clone_htod(&aosoa::pack(states))?, n: states.len() })
    }

    pub fn alloc_states(&self, n: usize) -> Result<DeviceStates, DriverError> {
        Ok(DeviceStates { buf: self.stream.alloc_zeros::<u32>(aosoa::words(n))?, n })
    }

    pub fn download_words(&self, d: &DeviceStates) -> Result<Vec<u32>, DriverError> {
        self.stream.clone_dtoh(&d.buf)
    }

    pub fn download_states(&self, d: &DeviceStates) -> Result<Vec<Box<ArenaState>>, DriverError> {
        Ok(aosoa::unpack(&self.download_words(d)?, d.n))
    }

    /// 128 threads per block: the phase kernel (the solver especially) needs many registers, so a 1024-thread block cannot launch.
    fn cfg(n: usize) -> LaunchConfig {
        LaunchConfig { grid_dim: ((n as u32).div_ceil(128).max(1), 1, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 }
    }

    /// Copy-in to a local `ArenaState` and copy-out into `out`, one thread per arena.
    pub fn identity(&self, input: &DeviceStates, out: &mut DeviceStates) -> Result<(), DriverError> {
        let n = input.n as u32;
        unsafe { self.stream.launch_builder(&self.identity).arg(&input.buf).arg(&mut out.buf).arg(&n).launch(Self::cfg(input.n))? };
        self.stream.synchronize()
    }

    /// The tick-boundary copy: only the persistent words of each state move (the mid-tick part stays zero in `out`).
    pub fn identity_tick(&self, input: &DeviceStates, out: &mut DeviceStates) -> Result<(), DriverError> {
        let n = input.n as u32;
        unsafe { self.stream.launch_builder(&self.identity_tick).arg(&input.buf).arg(&mut out.buf).arg(&n).launch(Self::cfg(input.n))? };
        self.stream.synchronize()
    }

    /// Runs phase `k` on every arena in place; returns the status of each arena (`PhaseStatus`).
    pub fn phase(&self, k: usize, states: &mut DeviceStates, sd: &DeviceStatic) -> Result<Vec<i32>, DriverError> {
        let (kk, n) = (k as i32, states.n as u32);
        let mut status = self.stream.alloc_zeros::<i32>(states.n.max(1))?;
        unsafe {
            self.stream
                .launch_builder(&self.phase)
                .arg(&kk)
                .arg(&mut states.buf)
                .arg(&sd.buf)
                .arg(&n)
                .arg(&mut status)
                .launch(Self::cfg(states.n))?
        };
        self.stream.synchronize()?;
        let mut s = self.stream.clone_dtoh(&status)?;
        s.truncate(states.n);
        Ok(s)
    }

    /// One full tick (V3 `step_tick`) of every arena in place, at tick boundaries; returns each arena's PhaseStatus. Launched in
    /// `SIM8_BLOCK`-thread blocks.
    pub fn step_tick(&self, states: &mut DeviceStates, sd: &DeviceStatic) -> Result<Vec<i32>, DriverError> {
        let n = states.n as u32;
        let mut status = self.stream.alloc_zeros::<i32>(states.n.max(1))?;
        unsafe {
            self.stream.launch_builder(&self.step_tick).arg(&mut states.buf).arg(&sd.buf).arg(&n).arg(&mut status).launch(Self::cfg_block(states.n, SIM8_BLOCK))?
        };
        self.stream.synchronize()?;
        let mut s = self.stream.clone_dtoh(&status)?;
        s.truncate(states.n);
        Ok(s)
    }

    /// The controls of one decision as the kernel reads them: `controls[a][i]` is car i's in arena a (`Controls[n][NUM_CARS]`,
    /// 8 words each: throttle, steer, pitch, yaw, roll as f32, then jump, boost, handbrake as 0 or 1).
    pub fn upload_controls(&self, controls: &[[Controls; NUM_CARS]]) -> Result<CudaSlice<u32>, DriverError> {
        // SAFETY: Controls is repr(C), eight 4-byte fields, no padding.
        let words = unsafe { std::slice::from_raw_parts(controls.as_ptr().cast::<u32>(), std::mem::size_of_val(controls) / 4) };
        self.stream.clone_htod(words)
    }

    /// Launches one decision (`TICKS_PER_DECISION` ticks, `sim8`) on every arena of `states` in place, asynchronously on
    /// `self.stream`: `controls` from `upload_controls` (set before every tick), `result[a]` the `SIM8_*` word of arena a. `block`
    /// threads per block (1..=`SIM_MAX_BLOCK`; `SIM8_BLOCK` in production). With `trace` (`TICKS_PER_DECISION * aosoa::words(n)`
    /// words) it runs `sim8_trace`, which also writes the state after tick t into trace buffer t - 1.
    pub fn sim8(&self, states: &mut DeviceStates, sd: &DeviceStatic, controls: &CudaSlice<u32>, result: &mut CudaSlice<u32>, block: u32, trace: Option<&mut CudaSlice<u32>>) -> Result<(), DriverError> {
        let n = states.n as u32;
        assert!((1..=SIM_MAX_BLOCK).contains(&block), "sim8: block of {block} threads");
        assert!(controls.len() >= states.n * NUM_CARS * std::mem::size_of::<Controls>() / 4 && result.len() >= states.n, "sim8: buffers too short");
        let cfg = Self::cfg_block(states.n, block);
        match trace {
            None => unsafe { self.stream.launch_builder(&self.sim8).arg(&mut states.buf).arg(&sd.buf).arg(controls).arg(result).arg(&n).launch(cfg)? },
            Some(t) => {
                assert!(t.len() >= TICKS_PER_DECISION * aosoa::words(states.n), "sim8: trace too short");
                unsafe { self.stream.launch_builder(&self.sim8_trace).arg(&mut states.buf).arg(&sd.buf).arg(controls).arg(result).arg(&n).arg(t).launch(cfg)? }
            }
        };
        Ok(())
    }

    fn cfg_block(n: usize, block: u32) -> LaunchConfig {
        LaunchConfig { grid_dim: ((n as u32).div_ceil(block).max(1), 1, 1), block_dim: (block, 1, 1), shared_mem_bytes: 0 }
    }
}
