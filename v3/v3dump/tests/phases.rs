//! The phase-split V3 (feature `trace`): (1) running phases P0..P8 equals `step_tick` byte for byte, tick after tick over
//! the whole scenario set; (2) mid-tick completeness: the dump taken at every phase boundary, imported into another
//! arena, finishes the tick to the same bytes as the uninterrupted one.
//!   ROCKETSIM_MESHES=<dir> cargo test --release -p v3dump --test phases -- --nocapture
//! `V3TRACE_STRIDE=n` samples the mid-tick check every n-th tick (default 4); `V3DUMP_SCALE=n` lengthens the plays.

use std::sync::atomic::{AtomicUsize, Ordering};

use rocketsim::{Arena, PHASE_NAMES, PHASES, dump::ARENA_COUNTS};
use v3dump::{
    bots::Bot,
    scenario::{Coverage, Prev, Spec, build_arena, diff, observe, specs, start_arena, step_arena_with},
};

#[derive(Clone, Default)]
struct Out {
    name: String,
    ticks: u64,
    identity_checks: u64,
    boundary_checks: u64,
    fresh_checks: u64,
    failures: Vec<String>,
    /// largest header count seen at each boundary k = 0..=PHASES
    max_counts: Vec<[u32; ARENA_COUNTS]>,
    max_len: usize,
    cov: Coverage,
}

fn run(spec: &Spec, stride: u32) -> Out {
    let mut o = Out {
        name: spec.name(),
        max_counts: vec![[0; ARENA_COUNTS]; PHASES + 1],
        ..Out::default()
    };
    let mut a = start_arena(spec); // stepped by step_tick
    let mut b = a.clone(); // stepped by the phases
    let mut pool = build_arena(spec); // a used arena that receives the mid-tick dumps
    let mut bots = [Bot::new(spec.modes[0], spec.seed ^ 0x11), Bot::new(spec.modes[1], spec.seed ^ 0x22)];
    let mut prev = Prev::default();
    let mut cov_b = Coverage::default();
    let mut samples = 0u64;
    // V3TRACE_SENSITIVITY=1: skip the re-export check, so that a field zeroed on import shows only if it matters
    let sensitivity = std::env::var_os("V3TRACE_SENSITIVITY").is_some();
    for t in 0..spec.ticks {
        let ctrls = [bots[0].controls(&a, 0), bots[1].controls(&a, 1)];

        if t % stride == 0 {
            // mid-tick completeness at every boundary of this tick
            let mut start = a.clone();
            start.set_car_controls(0, ctrls[0]);
            start.set_car_controls(1, ctrls[1]);
            let mut whole = start.clone();
            whole.step_tick();
            let next = whole.export_state();
            samples += 1;
            let mut fresh = (samples % 64 == 1).then(|| build_arena(spec));
            let mut c = start.clone();
            for k in 0..=PHASES {
                if k > 0 {
                    c.step_phase(k - 1);
                }
                if k == PHASES {
                    // boundary PHASES is the end of the tick: it must equal the uninterrupted tick
                    if let Some(m) = diff(&c.export_state(), &next) {
                        o.failures.push(format!("tick {t}: phases 0..{PHASES} vs step_tick: {m}"));
                    }
                    continue;
                }
                let d = c.export_state();
                let counts = d.counts();
                for i in 0..ARENA_COUNTS {
                    o.max_counts[k][i] = o.max_counts[k][i].max(counts[i]);
                }
                o.max_len = o.max_len.max(d.as_bytes().len());
                for (name, target) in [("used", Some(&mut pool)), ("fresh", fresh.as_mut())] {
                    let Some(f) = target else { continue };
                    if let Err(e) = f.import_state(&d) {
                        o.failures.push(format!("tick {t} boundary {k} {name}: import failed: {e}"));
                        continue;
                    }
                    if !sensitivity {
                        if let Some(m) = diff(&d, &f.export_state()) {
                            o.failures.push(format!("tick {t} boundary {k} {name}: re-export differs: {m}"));
                            continue;
                        }
                    }
                    for j in k..PHASES {
                        f.step_phase(j);
                    }
                    o.boundary_checks += 1;
                    o.fresh_checks += u64::from(name == "fresh");
                    if let Some(m) = diff(&f.export_state(), &next) {
                        o.failures.push(format!(
                            "tick {t} boundary {k} (before {}) {name}: finishing the tick differs: {m}",
                            PHASE_NAMES[k]
                        ));
                    }
                }
            }
            if o.failures.len() > 5 {
                return o;
            }
        }

        step_arena_with(&mut a, &ctrls, &mut o.cov, false);
        step_arena_with(&mut b, &ctrls, &mut cov_b, true);
        observe(&a, &mut prev, &mut o.cov);
        o.ticks += 1;
        o.identity_checks += 1;
        if let Some(m) = diff(&a.export_state(), &b.export_state()) {
            o.failures.push(format!("tick {t}: step_tick vs phases: {m}"));
            return o;
        }
    }
    o
}

fn run_all(specs: &[Spec], stride: u32) -> Vec<Out> {
    let cores = std::thread::available_parallelism().map_or(4, |n| n.get());
    // half the cores by default: the PC is shared
    let threads = std::env::var("V3TRACE_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or((cores / 2).max(1));
    let next = AtomicUsize::new(0);
    let out = std::sync::Mutex::new(vec![Out::default(); specs.len()]);
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= specs.len() {
                        break;
                    }
                    let r = std::panic::catch_unwind(|| run(&specs[i], stride)).unwrap_or_else(|e| {
                        let msg = e
                            .downcast_ref::<String>()
                            .cloned()
                            .or_else(|| e.downcast_ref::<&str>().map(|s| (*s).to_string()))
                            .unwrap_or_default();
                        Out { name: specs[i].name(), failures: vec![format!("PANIC {msg}")], ..Out::default() }
                    });
                    out.lock().unwrap()[i] = r;
                }
            });
        }
    });
    out.into_inner().unwrap()
}

#[test]
fn phases_equal_step_tick_and_mid_tick_dumps_are_complete() {
    let stride: u32 = std::env::var("V3TRACE_STRIDE").ok().and_then(|v| v.parse().ok()).unwrap_or(4).max(1);
    let all = specs();
    let t0 = std::time::Instant::now();
    let outs = run_all(&all, stride);
    let mut text = String::new();
    let (mut ticks, mut id, mut bc, mut fc) = (0, 0, 0, 0);
    let mut max = vec![[0u32; ARENA_COUNTS]; PHASES + 1];
    let mut max_len = 0;
    for o in &outs {
        ticks += o.ticks;
        id += o.identity_checks;
        bc += o.boundary_checks;
        fc += o.fresh_checks;
        max_len = max_len.max(o.max_len);
        for k in 0..=PHASES {
            for i in 0..ARENA_COUNTS {
                max[k][i] = max[k][i].max(o.max_counts[k][i]);
            }
        }
        for f in &o.failures {
            text.push_str(&format!("FAIL {}: {f}\n", o.name));
        }
    }
    text.push_str(&format!(
        "{} scenarios, {ticks} ticks: {id} tick-by-tick identity checks (phases vs step_tick, byte for byte), {bc} mid-tick \
         boundary checks ({fc} into fresh arenas), {} failures, {:.0}s\n",
        outs.len(),
        outs.iter().map(|o| o.failures.len()).sum::<usize>(),
        t0.elapsed().as_secs_f32()
    ));
    let sum = |f: fn(&Coverage) -> u64| outs.iter().map(|o| f(&o.cov)).sum::<u64>();
    text.push_str(&format!(
        "ball sleep (788c382): {} ticks asleep, {} falls asleep, {} wake-ups\n",
        sum(|c| c.ball_sleep_ticks),
        sum(|c| c.ball_sleeps),
        sum(|c| c.ball_wakes)
    ));
    text.push_str("max header counts by boundary [bodies, manifolds, active, points, table, handles, pairs, ray cells, cars, pads, records, ball-world records]\n");
    for k in 0..PHASES {
        text.push_str(&format!("  before {:<3} {:?}\n", PHASE_NAMES[k], max[k]));
    }
    text.push_str(&format!("largest mid-tick dump: {max_len} bytes\n"));
    println!("{text}");
    let _ = std::fs::write(std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("v3trace-report.txt"), &text);
    assert!(outs.iter().all(|o| o.failures.is_empty()), "{text}");
    assert!(bc > 100_000, "too few boundary checks: {bc}");
    let _: Option<&Arena> = None;
}
