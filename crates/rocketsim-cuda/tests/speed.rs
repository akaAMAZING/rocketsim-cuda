//! sim8 throughput on start states played by V3 (`v3dump` scenarios), for comparing two builds of the same layout. Ignored by
//! default: a timing is only worth reading on a quiet GPU.
//!   ROCKETSIM_MESHES=<dir> cargo test --release -p rocketsim-cuda --test speed -- --ignored --nocapture
//! Environment: `ROCKETSIM_SIM8_FATBIN=<path>` times that fat binary (default the embedded one; build another with nvcc using the
//! flags of crates/rocketsim-cuda-build/src/lib.rs); `ROCKETSIM_SPEED_ARENAS` (default 65,536), `ROCKETSIM_SPEED_REPS` (default
//! 20), `ROCKETSIM_CUDA_SAMPLES` states per class (default 200). The states are the scenarios' (all five classes, the most
//! common car configuration), tiled to the arena count and shuffled; every launch starts from the same states (a device copy
//! before it, not timed). Prints the median and best launch in arena-ticks per second (8 ticks per arena).

mod common;

use std::{collections::BTreeMap, fs, time::Instant};

use common::{CLASSES, all_configs, decision_controls, permutation, read_rows, samples_per_class};
use cudarc::driver::{LaunchConfig, PushKernelArg};
use rocketsim_cuda::{
    device::{SIM8_BLOCK, SimDevice, TICKS_PER_DECISION},
    dump::parse_arena,
    layout::{ARENA_WORDS, ArenaState, Controls, NUM_CARS},
};
use rocketsim::dump::ArenaDump;

fn var(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

#[test]
#[ignore = "a timing: run on a quiet GPU"]
fn sim8_arena_ticks_per_second() {
    let (arenas, reps, per_class) = (var("ROCKETSIM_SPEED_ARENAS", 65_536), var("ROCKETSIM_SPEED_REPS", 20), samples_per_class(200));
    let mut configs = all_configs();
    let dev = SimDevice::new(0).expect("CUDA device");
    let fat = std::env::var("ROCKETSIM_SIM8_FATBIN").ok().map(|p| fs::read(p).unwrap());
    let module = rocketsim_cuda::cuda::load(&dev.ctx, fat.as_deref().unwrap_or(rocketsim_cuda::SIM_FATBIN)).unwrap();
    let f = module.load_function("sim8").unwrap();

    let mut by_cfg: BTreeMap<u64, Vec<(Box<ArenaState>, [Controls; NUM_CARS])>> = BTreeMap::new();
    for (ci, class) in CLASSES.iter().enumerate() {
        for row in read_rows(class, per_class) {
            let h = ArenaDump::from_bytes(row.dump.clone()).unwrap().config_hash();
            let (start, _) = parse_arena(&row.dump, &configs.get_mut(&h).unwrap().sd).unwrap();
            by_cfg.entry(h).or_default().push((start, decision_controls(ci, row.index)));
        }
    }
    let (h, set) = by_cfg.iter().max_by_key(|(_, v)| v.len()).unwrap();
    let sd = dev.upload_static(&configs[h].sd).unwrap();
    let order = permutation(arenas);
    let at = |a: usize, k: usize| (a / 32) * ARENA_WORDS * 32 + k * 32 + a % 32;
    let mut buf = vec![0u32; arenas.div_ceil(32) * ARENA_WORDS * 32];
    let mut ctls = Vec::with_capacity(arenas);
    for (a, &i) in order.iter().enumerate() {
        let (s, c) = &set[i % set.len()];
        for (k, &w) in s.words().iter().enumerate() {
            buf[at(a, k)] = w;
        }
        ctls.push(*c);
    }
    let pristine = dev.stream.clone_htod(&buf).unwrap();
    let mut work = dev.stream.clone_htod(&buf).unwrap();
    let cbuf = dev.upload_controls(&ctls).unwrap();
    let mut r = dev.stream.alloc_zeros::<u32>(arenas).unwrap();
    let n = arenas as u32;
    let cfg = LaunchConfig { grid_dim: (n.div_ceil(SIM8_BLOCK), 1, 1), block_dim: (SIM8_BLOCK, 1, 1), shared_mem_bytes: 0 };
    let mut times = Vec::new();
    for rep in 0..reps + 3 {
        dev.stream.memcpy_dtod(&pristine, &mut work).unwrap();
        dev.stream.synchronize().unwrap();
        let t = Instant::now();
        // SAFETY: the buffers are the kernel's arguments, sized for `arenas`.
        unsafe { dev.stream.launch_builder(&f).arg(&mut work).arg(&sd.buf).arg(&cbuf).arg(&mut r).arg(&n).launch(cfg).unwrap() };
        dev.stream.synchronize().unwrap();
        if rep >= 3 {
            times.push(t.elapsed().as_secs_f64());
        }
    }
    times.sort_by(f64::total_cmp);
    let rate = |t: f64| arenas as f64 * TICKS_PER_DECISION as f64 / t;
    println!(
        "SIM8_SPEED: {arenas} arenas of {} start states (configuration {h:016x}), {reps} launches: median {:.2} ms = {:.3} M arena-ticks/s, best {:.2} ms = {:.3} M arena-ticks/s",
        set.len(),
        times[times.len() / 2] * 1e3,
        rate(times[times.len() / 2]) / 1e6,
        times[0] * 1e3,
        rate(times[0]) / 1e6
    );
}
