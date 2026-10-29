//! The harness proves itself on V3's own scenario set (v3/v3dump) before the other tests rely on it:
//!   1. every V3 dump, at every tick and phase boundary, becomes an `ArenaState` and back without losing anything V3 needs:
//!      the dump rebuilt from the state, imported into V3, finishes the tick to the same `ArenaState` as the uninterrupted tick;
//!   2. the caps hold (no list over its cap) and the static bodies, the template and the static proxies never change.
//!   ROCKETSIM_MESHES=<dir> cargo test --release -p rocketsim-cuda --test harness -- --nocapture
//! `ROCKETSIM_HARNESS_STRIDE=n` samples every n-th tick (default 8); `ROCKETSIM_HARNESS_THREADS` (default: half the cores).

mod common;

use std::sync::atomic::{AtomicUsize, Ordering};

use common::{Config, load, make_config, snap};
use rocketsim_cuda::{
    compare::{Comparator, Stats},
    dump::{parse_arena, write_arena},
    layout::{MAX_ACTIVE, MAX_DYN_CELLS, MAX_MANIFOLDS, MAX_PAIRS, MAX_RECORDS},
};
use rocketsim::{PHASE_NAMES, PHASES};
use v3dump::{
    bots::Bot,
    scenario::{Coverage, Spec, build_arena, specs, start_arena, step_arena_with},
};

#[derive(Clone, Default)]
struct Out {
    name: String,
    ticks: u64,
    dumps: u64,
    finished: u64,
    failures: Vec<String>,
    /// largest list sizes seen at any boundary: stored records, manifolds, active, dyn cells, pairs
    max: [u32; 5],
    /// boundaries with more than 0, 1, 2, 4, 6, 8, 12 stored (car-ball and car-car) contact records
    rec_hist: [u64; 7],
    stats: Stats,
}

fn run(spec: &Spec, stride: u32) -> Out {
    let mut o = Out { name: spec.name(), ..Out::default() };
    let cmp = Comparator::new();
    // ROCKETSIM_ABLATE=field,field: zero these ArenaState fields before the rebuilt dump is imported; a field that matters
    // makes the run fail (how the field set was justified)
    let ablate: Vec<String> = std::env::var("ROCKETSIM_ABLATE").map(|v| v.split(',').map(str::to_string).collect()).unwrap_or_default();
    let leaves = rocketsim_cuda::layout::arena_leaves();
    let mut cfg: Config = make_config(spec);
    let mut a = start_arena(spec);
    let mut pool = build_arena(spec);
    let mut bots = [Bot::new(spec.modes[0], spec.seed ^ 0x11), Bot::new(spec.modes[1], spec.seed ^ 0x22)];
    let mut cov = Coverage::default();
    for t in 0..spec.ticks {
        let ctrls = [bots[0].controls(&a, 0), bots[1].controls(&a, 1)];
        if t % stride == 0 {
            let mut start = a.clone();
            start.set_car_controls(0, ctrls[0]);
            start.set_car_controls(1, ctrls[1]);
            let mut whole = start.clone();
            whole.step_tick();
            let next = snap(&whole, &mut cfg);
            let mut c = start.clone();
            for k in 0..PHASES {
                let d = c.export_state();
                let (s, tpl) = match parse_arena(d.as_bytes(), &cfg.sd) {
                    Ok(x) => x,
                    Err(e) => {
                        o.failures.push(format!("tick {t} boundary {k}: parse: {e}"));
                        return o;
                    }
                };
                o.dumps += 1;
                o.max = [
                    o.max[0].max(s.n_records),
                    o.max[1].max(s.n_manifolds),
                    o.max[2].max(s.n_active),
                    o.max[3].max(s.n_dyn_cells),
                    o.max[4].max(s.n_pairs),
                ];
                for (i, th) in [0, 1, 2, 4, 6, 8, 12].into_iter().enumerate() {
                    o.rec_hist[i] += u64::from(s.n_records > th);
                }
                if cfg.template.is_none() {
                    cfg.template = Some(tpl.clone());
                }
                let reference = cfg.template.as_ref().unwrap();
                if tpl.static_body != reference.static_body
                    || tpl.ball_other != reference.ball_other
                    || tpl.num_handles != reference.num_handles
                    || tpl.next_new_handle != reference.next_new_handle
                    || tpl.config_hash != reference.config_hash
                {
                    o.failures.push(format!("tick {t} boundary {k}: a static body record, the ball info or a counter changed"));
                    return o;
                }
                // writer: state -> dump -> state is the identity
                let bytes = match write_arena(&s, &cfg.sd, reference) {
                    Ok(b) => b,
                    Err(e) => {
                        o.failures.push(format!("tick {t} boundary {k}: write: {e}"));
                        return o;
                    }
                };
                match parse_arena(&bytes, &cfg.sd) {
                    Ok((s2, _)) if s2.words() == s.words() => {}
                    Ok(_) => {
                        o.failures.push(format!("tick {t} boundary {k}: write then parse changed the state"));
                        return o;
                    }
                    Err(e) => {
                        o.failures.push(format!("tick {t} boundary {k}: rewritten dump does not parse: {e}"));
                        return o;
                    }
                }
                // finishing the tick from the rebuilt dump gives the uninterrupted tick
                let mut s = s;
                for g in &ablate {
                    for (w, leaf) in leaves.iter().enumerate() {
                        if leaf.0 == *g || leaf.0.strip_prefix(g.as_str()).is_some_and(|r| r.starts_with('#')) {
                            s.words_mut()[w] = 0;
                        }
                    }
                }
                if let Err(e) = load(&mut pool, &cfg, &s) {
                    o.failures.push(format!("tick {t} boundary {k}: V3 refused the rebuilt dump: {e}"));
                    return o;
                }
                for j in k..PHASES {
                    pool.step_phase(j);
                }
                let fin = snap(&pool, &mut cfg);
                let (_, bok) = cmp.compare(&fin, &next, &mut o.stats);
                o.finished += 1;
                if !bok {
                    o.failures.push(format!(
                        "tick {t} boundary {k} (before {}): finishing differs: {:?}",
                        PHASE_NAMES[k],
                        o.stats.worst()
                    ));
                    if ablate.is_empty() {
                        return o;
                    }
                }
                c.step_phase(k);
            }
        }
        step_arena_with(&mut a, &ctrls, &mut cov, false);
        o.ticks += 1;
    }
    o
}

#[test]
fn arena_state_holds_everything_v3_needs_at_every_boundary() {
    let stride: u32 = std::env::var("ROCKETSIM_HARNESS_STRIDE").ok().and_then(|v| v.parse().ok()).unwrap_or(8).max(1);
    let cores = std::thread::available_parallelism().map_or(4, |n| n.get());
    let threads: usize = std::env::var("ROCKETSIM_HARNESS_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or((cores / 2).max(1));
    let all = specs();
    let t0 = std::time::Instant::now();
    let next = AtomicUsize::new(0);
    let outs = std::sync::Mutex::new(vec![Out::default(); all.len()]);
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= all.len() {
                        break;
                    }
                    let r = std::panic::catch_unwind(|| run(&all[i], stride)).unwrap_or_else(|e| {
                        let msg = e
                            .downcast_ref::<String>()
                            .cloned()
                            .or_else(|| e.downcast_ref::<&str>().map(|s| (*s).to_string()))
                            .unwrap_or_default();
                        Out { name: all[i].name(), failures: vec![format!("PANIC {msg}")], ..Out::default() }
                    });
                    outs.lock().unwrap()[i] = r;
                }
            });
        }
    });
    let outs = outs.into_inner().unwrap();
    let mut text = String::new();
    let (mut ticks, mut dumps, mut fin) = (0, 0, 0);
    let mut max = [0u32; 5];
    let mut rec_hist = [0u64; 7];
    let mut stats = Stats::default();
    for o in &outs {
        ticks += o.ticks;
        dumps += o.dumps;
        fin += o.finished;
        for i in 0..5 {
            max[i] = max[i].max(o.max[i]);
        }
        stats.merge(&o.stats);
        for i in 0..7 {
            rec_hist[i] += o.rec_hist[i];
        }
        for f in &o.failures {
            text.push_str(&format!("FAIL {}: {f}\n", o.name));
        }
    }
    text.push_str(&format!(
        "{} scenarios, {ticks} ticks, {dumps} dumps (tick and phase boundaries) became ArenaStates and V3 dumps again, {fin} of them \
         imported into V3 and run to the end of the tick; {} compared states, {} discrete-identical, {} bit-identical; {} failures; {:.0}s\n",
        outs.len(),
        stats.states,
        stats.discrete_equal,
        stats.bit_equal,
        outs.iter().map(|o| o.failures.len()).sum::<usize>(),
        t0.elapsed().as_secs_f32()
    ));
    text.push_str(&format!(
        "largest lists at any boundary (cap): stored contact records (car-ball, car-car) {} ({MAX_RECORDS}), manifolds {} ({MAX_MANIFOLDS}), active {} ({MAX_ACTIVE}), dynamic ray-cell entries {} ({MAX_DYN_CELLS}), pairs {} ({MAX_PAIRS})\n",
        max[0], max[1], max[2], max[3], max[4]
    ));
    text.push_str(&format!("boundaries with more than 0, 1, 2, 4, 6, 8, 12 stored contact records (of {dumps}): {rec_hist:?}
"));
    if stats.diff_words() != (0, 0) {
        text.push_str(&stats.field_table(30));
    }
    println!("{text}");
    let _ = std::fs::write(std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("harness-report.txt"), &text);
    if std::env::var_os("ROCKETSIM_ABLATE").is_some() {
        // an ablation run reports and does not assert: the point is to see which boundaries fail
        return;
    }
    assert!(outs.iter().all(|o| o.failures.is_empty()), "{text}");
    assert!(fin > 50_000, "too few checks: {fin}");
}
