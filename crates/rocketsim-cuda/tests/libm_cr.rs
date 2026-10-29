//! The correctly rounded libm (`cuda/common/libm_cr.cuh`, ported from CORE-MATH): the device kernels and the host build must
//! return the same bits, on every input tested.
//!   * sinf, cosf, tanf: all 2^32 binary32 inputs, device against host.
//!   * atan2f, powf: `ROCKETSIM_LIBM_N` (default 10^9) random cases each, drawn across the binades from an index
//!     (`cuda/simtest/libm_inputs.cuh`, the same generator on both sides), device against host.
//!   * the special-value grid (zeros, infinities, NaNs, denormals, +-1, extremes): all five functions, every pair.
//! Beside each comparison the platform C library (host) and CUDA's own library (device) are counted against the correctly
//! rounded value, for the record. The host side runs on all but two cores, at below-normal priority.
//!   cargo test --release -p rocketsim-cuda --test libm_cr -- --nocapture --test-threads=1
//! The correctness of the values themselves is CORE-MATH's (it proves its functions correctly rounded); this test checks that
//! the port computes the same bits as the host build of the same source.

use std::sync::Mutex;
use std::time::Instant;

use rocketsim_cuda::cuda::{LaunchConfig, PushKernelArg, context, load};
use rocketsim_cuda::{SIMTEST_FATBIN, libm};

static HEAVY: Mutex<()> = Mutex::new(());
const CHUNK: usize = 1 << 26;

#[cfg(windows)]
fn below_normal() {
    unsafe extern "system" {
        fn GetCurrentProcess() -> isize;
        fn SetPriorityClass(h: isize, class: u32) -> i32;
    }
    // SAFETY: the pseudo handle of this process.
    unsafe { SetPriorityClass(GetCurrentProcess(), 0x0000_4000) };
}
#[cfg(not(windows))]
fn below_normal() {}

fn threads() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get()).saturating_sub(2).max(1)
}

/// Fills `out` (case `base + i` at `i`) from several threads.
fn par_fill(out: &mut [u32], base: u64, f: impl Fn(u64, &mut [u32]) + Sync) {
    let per = out.len().div_ceil(threads());
    std::thread::scope(|s| {
        for (t, ch) in out.chunks_mut(per).enumerate() {
            let f = &f;
            s.spawn(move || f(base + (t * per) as u64, ch));
        }
    });
}

/// Words that differ between `a` and `b` (two NaNs are equal when `nan_equal`), and the first few indices.
fn par_diff(a: &[u32], b: &[u32], nan_equal: bool) -> (u64, Vec<usize>) {
    let per = a.len().div_ceil(threads());
    let parts: Vec<(u64, Vec<usize>)> = std::thread::scope(|s| {
        let hs: Vec<_> = a
            .chunks(per)
            .zip(b.chunks(per))
            .enumerate()
            .map(|(t, (x, y))| {
                s.spawn(move || {
                    let (mut n, mut first) = (0u64, Vec::new());
                    for (i, (&p, &q)) in x.iter().zip(y).enumerate() {
                        if p != q && !(nan_equal && f32::from_bits(p).is_nan() && f32::from_bits(q).is_nan()) {
                            n += 1;
                            if first.len() < 5 {
                                first.push(t * per + i);
                            }
                        }
                    }
                    (n, first)
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let n = parts.iter().map(|p| p.0).sum();
    (n, parts.into_iter().flat_map(|p| p.1).take(5).collect())
}

struct Gpu {
    stream: std::sync::Arc<cudarc::driver::CudaStream>,
    unary: cudarc::driver::CudaFunction,
    stream_k: cudarc::driver::CudaFunction,
    grid: cudarc::driver::CudaFunction,
    f64k: cudarc::driver::CudaFunction,
}

impl Gpu {
    fn new() -> Self {
        let ctx = context(0).expect("CUDA device");
        let stream = ctx.new_stream().unwrap();
        let module = load(&ctx, SIMTEST_FATBIN).unwrap();
        Self {
            stream,
            unary: module.load_function("libm_unary").unwrap(),
            stream_k: module.load_function("libm_stream").unwrap(),
            grid: module.load_function("libm_grid").unwrap(),
            f64k: module.load_function("libm_f64").unwrap(),
        }
    }

    /// Device results for the range [base, base + n): `f` 0..3 unary over bit patterns, 3 and 4 the random streams of
    /// atan2f and powf.
    fn run(&self, f: usize, native: bool, base: u64, d: &mut cudarc::driver::CudaSlice<u32>, host: &mut [u32], copy: bool) {
        let (nat, n) = (native as i32, host.len() as u64);
        let cfg = LaunchConfig::for_num_elems(host.len() as u32);
        let k = if f < 3 { &self.unary } else { &self.stream_k };
        let op = if f < 3 { f as i32 } else { f as i32 - 3 };
        // SAFETY: `d` holds `n` words.
        unsafe { self.stream.launch_builder(k).arg(&op).arg(&nat).arg(&base).arg(&n).arg(&mut *d).launch(cfg).unwrap() };
        if copy {
            self.stream.memcpy_dtoh(&d.slice(0..host.len()), host).unwrap();
        }
        self.stream.synchronize().unwrap();
    }
}

fn input_str(f: usize, idx: u64) -> String {
    if f < 3 {
        format!("{:08x} ({:e})", idx as u32, f32::from_bits(idx as u32))
    } else {
        let (a, b) = libm::stream_input(f - 3, idx);
        format!("({a:e} [{:08x}], {b:e} [{:08x}])", a.to_bits(), b.to_bits())
    }
}

/// Compares device and host over `n` cases of function `f`; returns (device != host CR, platform != CR, CUDA native != CR).
fn compare_range(g: &Gpu, f: usize, base0: u64, n: u64) -> [u64; 3] {
    let mut d = g.stream.alloc_zeros::<u32>(CHUNK).unwrap();
    let (mut dev, mut host, mut other) = (vec![0u32; CHUNK], vec![0u32; CHUNK], vec![0u32; CHUNK]);
    let mut tally = [0u64; 3];
    let host_fn = |platform: bool| move |base: u64, out: &mut [u32]| if f < 3 { libm::unary(f, platform, base, out) } else { libm::stream(f - 3, platform, base, out) };
    let mut base = base0;
    while base < base0 + n {
        let len = CHUNK.min((base0 + n - base) as usize);
        let (dev, host, other) = (&mut dev[..len], &mut host[..len], &mut other[..len]);
        g.run(f, false, base, &mut d, dev, true);
        par_fill(host, base, host_fn(false));
        let (bad, first) = par_diff(dev, host, false);
        for &i in &first {
            println!("  MISMATCH {} {}: device {:08x}, host {:08x}", libm::FUNCS[f], input_str(f, base + i as u64), dev[i], host[i]);
        }
        tally[0] += bad;
        par_fill(other, base, host_fn(true));
        tally[1] += par_diff(other, host, true).0;
        g.run(f, true, base, &mut d, other, true);
        tally[2] += par_diff(other, host, true).0;
        base += len as u64;
    }
    tally
}

#[test]
fn exhaustive_sin_cos_tan_device_equals_host() {
    let _l = HEAVY.lock().unwrap_or_else(|e| e.into_inner());
    below_normal();
    let g = Gpu::new();
    println!("all 2^32 binary32 inputs, device (sm_{}) against the host build, {} host threads, below-normal priority", rocketsim_cuda::cuda::ARCHS, threads());
    println!("{:<8} {:>12} {:>20} {:>26} {:>26}", "func", "inputs", "device != host (CR)", "host platform != CR", "device CUDA libm != CR");
    let mut bad = 0;
    for f in 0..3 {
        let t = Instant::now();
        let [a, p, n] = compare_range(&g, f, 0, 1 << 32);
        println!("{:<8} {:>12} {:>20} {:>26} {:>26}   [{:.0} s]", libm::FUNCS[f], 1u64 << 32, a, p, n, t.elapsed().as_secs_f64());
        bad += a;
    }
    assert_eq!(bad, 0, "the device differs from the host build");
}

#[test]
fn random_atan2_powf_device_equals_host() {
    let _l = HEAVY.lock().unwrap_or_else(|e| e.into_inner());
    below_normal();
    let g = Gpu::new();
    let n: u64 = std::env::var("ROCKETSIM_LIBM_N").ok().and_then(|v| v.parse().ok()).unwrap_or(1_000_000_000);
    println!("{n} random cases per function across the binades, device (sm_{}) against the host build", rocketsim_cuda::cuda::ARCHS);
    println!("{:<8} {:>12} {:>20} {:>26} {:>26}", "func", "cases", "device != host (CR)", "host platform != CR", "device CUDA libm != CR");
    let mut bad = 0;
    for f in 3..5 {
        let t = Instant::now();
        let [a, p, nn] = compare_range(&g, f, 0, n);
        println!("{:<8} {:>12} {:>20} {:>26} {:>26}   [{:.0} s]", libm::FUNCS[f], n, a, p, nn, t.elapsed().as_secs_f64());
        bad += a;
    }
    assert_eq!(bad, 0, "the device differs from the host build");
}

#[test]
fn special_value_grid_device_equals_host() {
    let _l = HEAVY.lock().unwrap_or_else(|e| e.into_inner());
    let g = Gpu::new();
    let mut bad = 0;
    for f in 0..5 {
        let host = libm::grid(f, false);
        let mut d = g.stream.alloc_zeros::<u32>(host.len()).unwrap();
        let (fi, nat) = (f as i32, 0i32);
        // SAFETY: `d` holds the whole grid.
        unsafe { g.stream.launch_builder(&g.grid).arg(&fi).arg(&nat).arg(&mut d).launch(LaunchConfig::for_num_elems(host.len() as u32)).unwrap() };
        let dev = g.stream.clone_dtoh(&d).unwrap();
        let diff = dev.iter().zip(&host).filter(|(a, b)| a != b).count();
        println!("{:<8} {:>5} special cases, device != host: {diff}", libm::FUNCS[f], host.len());
        bad += diff;
    }
    assert_eq!(bad, 0);
    // a few exact values every correct implementation returns
    let sin = libm::grid(0, false);
    assert_eq!(sin[0], 0, "sin(+0) = +0");
    assert_eq!(sin[1], 0x8000_0000, "sin(-0) = -0");
    assert!(f32::from_bits(sin[4]).is_nan(), "sin(inf) is NaN");
    let pow = libm::grid(4, false);
    let n = 40;
    assert_eq!(pow[2 * n + 6], 0x3F80_0000, "pow(1, NaN) = 1");
    assert_eq!(pow[6 * n + 0], 0x3F80_0000, "pow(NaN, +0) = 1");
}

/// A short timing, as a by-product of the correctness tests (no benchmark campaign).
#[test]
fn timing_report() {
    let _l = HEAVY.lock().unwrap_or_else(|e| e.into_inner());
    let g = Gpu::new();
    let n = 1usize << 26;
    let mut d = g.stream.alloc_zeros::<u32>(n).unwrap();
    let mut host = vec![0u32; n];
    println!("{:<8} {:>22} {:>22} {:>22} {:>22}", "func", "device CR ns/call", "device CUDA ns/call", "host CR ns/call", "host platform ns/call");
    for f in 0..5 {
        let time = |native: bool, g: &Gpu, d: &mut cudarc::driver::CudaSlice<u32>, host: &mut Vec<u32>| {
            let base = if f < 3 { 0x3C00_0000 } else { 0 }; // sin, cos, tan over 2^-7 .. 2; the random streams otherwise
            g.run(f, native, base, d, host, false); // warm up
            (0..3).map(|_| { let t = Instant::now(); g.run(f, native, base, d, host, false); t.elapsed().as_secs_f64() }).fold(f64::MAX, f64::min) / n as f64 * 1e9
        };
        let (cr, nat) = (time(false, &g, &mut d, &mut host), time(true, &g, &mut d, &mut host));
        let m = 1usize << 22;
        let mut h = |platform: bool| {
            let t = Instant::now();
            if f < 3 { libm::unary(f, platform, 0x3F00_0000, &mut host[..m]) } else { libm::stream(f - 3, platform, 0, &mut host[..m]) }
            t.elapsed().as_secs_f64() / m as f64 * 1e9
        };
        let (hc, hp) = (h(false), h(true));
        println!("{:<8} {:>22.3} {:>22.3} {:>22.1} {:>22.1}", libm::FUNCS[f], cr, nat, hc, hp);
    }
    println!("(device: kernel alone, 2^26 calls, best of 3; host: one thread, 2^22 calls; sin, cos, tan over the first range of 2^-7..2 and 0.5..0.75)");
}

/// The f64 functions of the auto-flip path: atan2_64 (double-double) and hypot64 (CORE-MATH): device against the host build on
/// `ROCKETSIM_LIBM_N` (default 10^9) random cases each, with the platform library counted for the record; then how often the roll
/// sign's integers would change against V3's platform functions.
#[test]
fn f64_functions_device_equals_host() {
    let _l = HEAVY.lock().unwrap_or_else(|e| e.into_inner());
    below_normal();
    let g = Gpu::new();
    let n: u64 = std::env::var("ROCKETSIM_LIBM_N").ok().and_then(|v| v.parse().ok()).unwrap_or(1_000_000_000);
    const C: usize = 1 << 25;
    let mut d = g.stream.alloc_zeros::<u64>(C).unwrap();
    let (mut dev, mut host, mut plat) = (vec![0u64; C], vec![0u64; C], vec![0u64; C]);
    println!("{n} random cases per function (half in the auto-flip range, half across every binade), device against host");
    println!("{:<9} {:>12} {:>20} {:>26}", "func", "cases", "device != host (CR)", "host platform != port");
    let mut bad_total = 0;
    for (op, name) in ["atan2_64", "hypot64"].iter().enumerate() {
        let (mut bad, mut pl) = (0u64, 0u64);
        let mut base = 0u64;
        while base < n {
            let len = C.min((n - base) as usize);
            let (opi, nn) = (op as i32, len as u64);
            // SAFETY: `d` holds `C >= len` words.
            unsafe { g.stream.launch_builder(&g.f64k).arg(&opi).arg(&base).arg(&nn).arg(&mut d).launch(LaunchConfig::for_num_elems(len as u32)).unwrap() };
            g.stream.memcpy_dtoh(&d.slice(0..len), &mut dev[..len]).unwrap();
            g.stream.synchronize().unwrap();
            let per = len.div_ceil(threads());
            for (platform, out) in [(false, &mut host), (true, &mut plat)] {
                std::thread::scope(|s| {
                    for (t, ch) in out[..len].chunks_mut(per).enumerate() {
                        s.spawn(move || libm::f64_stream(op, platform, base + (t * per) as u64, ch));
                    }
                });
            }
            let nan_eq = |a: u64, b: u64| f64::from_bits(a).is_nan() && f64::from_bits(b).is_nan();
            for i in 0..len {
                if dev[i] != host[i] && !nan_eq(dev[i], host[i]) {
                    bad += 1;
                    if bad <= 5 {
                        println!("  MISMATCH {name} case {}: device {:016x} host {:016x}", base + i as u64, dev[i], host[i]);
                    }
                }
                pl += u64::from(plat[i] != host[i] && !nan_eq(plat[i], host[i]));
            }
            base += len as u64;
        }
        println!("{:<9} {:>12} {:>20} {:>26}", name, n, bad, pl);
        bad_total += bad;
    }
    assert_eq!(bad_total, 0, "the device differs from the host build");

    let m = n.min(1_000_000_000);
    let per = m.div_ceil(threads() as u64);
    let parts: Vec<[u64; 8]> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..threads() as u64).map(|t| s.spawn(move || libm::autoflip_changes(t * per, per.min(m.saturating_sub(t * per))))).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let c = parts.iter().fold([0u64; 8], |mut a, p| {
        a.iter_mut().zip(p).for_each(|(x, y)| *x += y);
        a
    });
    println!("auto-flip roll sign, port's atan2_64/hypot64 against the platform's (V3), {m} sampled directions, three atan2 calls each:");
    println!("  atan2 results (double) changed {} of {} calls; after the f32 cast {}", c[1], 3 * c[0], c[2]);
    println!("  hypot results (double) changed {} of {}; after the f32 cast {}", c[3], c[0], c[4]);
    println!("  int results changed: yaw {}, pitch {}, roll {} (of {} each)", c[5], c[6], c[7], c[0]);
}
