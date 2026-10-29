//! The device layer: a CUDA context through the driver API (`cudarc`, loaded at run time, so a build needs no CUDA libraries
//! and a host needs only the driver) and the loader for the fat binaries `build.rs` compiles.

use std::sync::Arc;

pub use cudarc::driver::{CudaContext, CudaModule, CudaStream, DriverError, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;

/// The architectures the fat binaries hold (SASS only), e.g. `86,120`.
pub const ARCHS: &str = env!("ROCKETSIM_CUDA_ARCHS");

/// The per-thread device stack the kernels run with (bytes).
pub const STACK_BYTES: usize = 8192;

/// Opens the device's context with cudarc's per-buffer event tracking off: callers order their work themselves (streams and
/// explicit events), and tracking would record and wait on an event per kernel argument at every launch and break stream
/// capture.
pub fn context(ordinal: usize) -> Result<Arc<CudaContext>, DriverError> {
    let ctx = CudaContext::new(ordinal)?;
    // SAFETY: no buffer exists yet, and the caller synchronises its streams explicitly.
    unsafe { ctx.disable_event_tracking() };
    // Some kernels call non-inlined device functions whose call chain is deeper than the driver's default 1 KiB stack: a
    // stack overflow reads as CUDA_ERROR_LAUNCH_FAILED. A larger stack costs local memory per resident thread only.
    // SAFETY: the context is current on this thread (CudaContext::new binds it); the limit is a plain setting
    unsafe { cudarc::driver::sys::cuCtxSetLimit(cudarc::driver::sys::CUlimit::CU_LIMIT_STACK_SIZE, STACK_BYTES).result()? };
    Ok(ctx)
}

/// Loads a fat binary into the context. Fails on a card whose architecture the binary does not hold.
pub fn load(ctx: &Arc<CudaContext>, image: &[u8]) -> Result<Arc<CudaModule>, DriverError> {
    ctx.load_module(Ptx::from_binary(image.to_vec()))
}
