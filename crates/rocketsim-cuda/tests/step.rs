//! Lockstep of the full tick and the decision kernel against V3, on start states that V3 itself plays (`v3dump` scenarios).
//!   * One tick: from every start state (with the controls it holds), V3's `step_tick` against the port's `step_tick` on the
//!     platform host (`rscuda_step_tick_platform`: must equal V3 bit for bit), the correctly rounded host (`rscuda_step_tick`) and the
//!     device (`step_tick_k`), which must equal each other bit for bit. The device's differences from V3 are reported per class,
//!     every differing word attributed to the libm function behind it by the mixed build (`rscuda_step_tick_mixed`).
//!   * One decision: from every start state, V3 runs 8 ticks with a pseudo-random control set per car before each tick
//!     (`set_car_controls`), stopping after the tick on which `is_ball_scored` is true; against `step_ticks` on the two host builds
//!     and `sim8` on the device (`sim8_trace` for the state after each tick): the same three relations, per tick, and the same scored
//!     tick. `sim8`'s final state must equal `sim8_trace`'s and the host's.
//!   * Determinism: `sim8` gives bit-identical states and result words across launches, block sizes (32, 64) and arena orders.
//!   * Events: the port's `ArenaState.events` after every tick (one tick and each tick of the decision) against the event list V3's
//!     `step_tick` returns, in the same bits (`v3_events`): the platform host must equal V3 on every tick; the device's
//!     differences are counted (its state differs from V3's on a few ticks, attributed above).
//! Every port output must have zero overflow counters and status 0 (no unported shape pair reached).
//!   ROCKETSIM_MESHES=<dir> cargo test --release -p rocketsim-cuda --test step -- --nocapture --test-threads=1
//! `ROCKETSIM_CUDA_SAMPLES` states per class (default 1000; the determinism test takes at most 400 per class).

mod common;

use std::collections::BTreeMap;

use common::{Attribution, CLASSES, Config, LIBM_FUNCS, Row, all_configs, decision_controls, diff_words, permutation, read_rows, render_attribution_rows, samples_per_class, snap};
use rocketsim_cuda::{
    aosoa,
    compare::{Comparator, Stats},
    device::{DeviceStatic, SIM8_BLOCK, SIM8_OVERFLOW, SIM8_SCORED_MASK, SIM8_UNIMPLEMENTED, SimDevice, TICKS_PER_DECISION},
    dump::parse_arena,
    ffi::{Libm, step_tick_host, step_ticks_host},
    layout::{ArenaState, Controls, NUM_CARS},
};
use rocketsim::{Arena, ArenaEvent, CarControls, dump::ArenaDump};
use v3dump::scenario::build_arena;

/// States per device launch group (bounds the memory: about CHUNK x 30 states of 14.5 KB alive at once).
const CHUNK: usize = 256;
const N: usize = TICKS_PER_DECISION;
const ALL_LIBM: u32 = (1 << LIBM_FUNCS.len()) - 1;

fn per_class() -> usize {
    samples_per_class(1000)
}


/// V3's events of one tick as `ArenaState.events` bits (cuda/sim/arena_state.cuh EventBit; the goal bit is not an event).
fn v3_events(evs: &[ArenaEvent]) -> u32 {
    let mut c = 0u32;
    for e in evs {
        c |= match e {
            ArenaEvent::CarHitBall(h) => 1 << h.car_idx,
            ArenaEvent::CarHitCar(h) => 1 << (if h.is_demo { 4 } else { 2 } + h.victim_car_idx),
            ArenaEvent::CarPickupBoost(p) => 1 << (6 + p.car_idx),
            ArenaEvent::CarLanded(l) => 1 << (8 + l.car_idx),
            ArenaEvent::CarHitWorld(w) => 1 << (10 + w.car_idx),
            ArenaEvent::BallHitWorld(w) => 1 << 12 | u32::from(w.contact_normal.z.abs() < 0.8) << 14,
        };
    }
    c
}

/// Event bit names, by bit (13, the goal, is not an event).
const EVENT_BITS: [&str; 15] =
    ["touch0", "touch1", "bump0", "bump1", "demo0", "demo1", "pickup0", "pickup1", "landed0", "landed1", "carworld0", "carworld1", "ballworld", "-", "ballwall"];

fn to_v3(c: &Controls) -> CarControls {
    CarControls { throttle: c.throttle, steer: c.steer, pitch: c.pitch, yaw: c.yaw, roll: c.roll, jump: c.jump != 0, boost: c.boost != 0, handbrake: c.handbrake != 0 }
}

/// One start state with V3's results: one tick from it, and the decision (the state after each tick that ran).
struct Item {
    class: usize,
    row: usize,
    config: u64,
    start: Box<ArenaState>,
    one: Box<ArenaState>,
    /// V3's events of the one tick and of each decision tick (`v3_events`)
    one_ev: u32,
    ctl: [Controls; NUM_CARS],
    ticks: Vec<Box<ArenaState>>,
    ticks_ev: Vec<u32>,
    scored: u32,
}

fn v3_item(arenas: &mut BTreeMap<u64, Arena>, configs: &mut BTreeMap<u64, Config>, class: usize, row: &Row) -> Item {
    let d = ArenaDump::from_bytes(row.dump.clone()).expect("start state is an arena dump");
    let config = d.config_hash();
    let cfg = configs.get_mut(&config).unwrap_or_else(|| panic!("{} row {}: unknown config", row.class, row.index));
    let arena = arenas.entry(config).or_insert_with(|| build_arena(&cfg.spec));
    arena.import_state(&d).unwrap_or_else(|e| panic!("{} row {}: import: {e}", row.class, row.index));
    let start = snap(arena, cfg);
    let one_ev = v3_events(arena.step_tick());
    let one = snap(arena, cfg);
    arena.import_state(&d).unwrap();
    let ctl = decision_controls(class, row.index);
    let (mut ticks, mut ticks_ev, mut scored) = (Vec::with_capacity(N), Vec::with_capacity(N), 0);
    for t in 1..=N {
        for (i, c) in ctl.iter().enumerate() {
            arena.set_car_controls(i, to_v3(c));
        }
        ticks_ev.push(v3_events(arena.step_tick()));
        ticks.push(snap(arena, cfg));
        if arena.is_ball_scored() {
            scored = t as u32;
            break;
        }
    }
    Item { class, row: row.index, config, start, one, one_ev, ctl, ticks, ticks_ev, scored }
}

/// The three relations of one key (a class, or a class and a tick).
#[derive(Default)]
struct Rel {
    platform_v3: Stats,
    cr_device: Stats,
    device_v3: Stats,
}

#[derive(Default)]
struct Fails {
    count: BTreeMap<&'static str, u64>,
    examples: Vec<String>,
}
impl Fails {
    fn add(&mut self, kind: &'static str, what: String) {
        *self.count.entry(kind).or_default() += 1;
        if self.examples.len() < 40 {
            self.examples.push(format!("{kind}: {what}"));
        }
    }
}

#[derive(Default)]
struct Report {
    one: BTreeMap<usize, Rel>,
    one_attr: BTreeMap<usize, Attribution>,
    eight: BTreeMap<(usize, usize), Rel>,
    eight_attr: BTreeMap<usize, Attribution>,
    /// class -> [states, scored within the decision by V3, device scored tick != V3's]
    scored: BTreeMap<usize, [u64; 3]>,
    /// class -> [ticks compared, platform host events != V3's, device events != V3's], and V3's event bits seen
    events: BTreeMap<usize, [u64; 3]>,
    event_bits: [u64; 15],
    fails: Fails,
}

fn overflow_zero(s: &ArenaState) -> bool {
    s.overflow.iter().all(|&x| x == 0)
}

fn worst(st: &Stats) -> String {
    st.worst().map_or("-".into(), |(f, u)| if u == u64::MAX { format!("{f} (discrete)") } else { format!("{f} ({u} ulp)") })
}

/// The device runs of one configuration's items: one tick (`step_tick_k`), and one decision through `sim8` and `sim8_trace`.
struct DeviceRuns {
    status1: Vec<i32>,
    one: Vec<Box<ArenaState>>,
    final8: Vec<Box<ArenaState>>,
    result8: Vec<u32>,
    final_trace: Vec<Box<ArenaState>>,
    result_trace: Vec<u32>,
    /// trace[t][j]: arena j after tick t + 1 (tick-boundary words; zero where the arena had stopped)
    trace: Vec<Vec<Box<ArenaState>>>,
}

fn device_runs(dev: &SimDevice, dsd: &DeviceStatic, items: &[&Item]) -> DeviceRuns {
    let n = items.len();
    let starts: Vec<Box<ArenaState>> = items.iter().map(|it| it.start.clone()).collect();
    let mut d1 = dev.upload_states(&starts).unwrap();
    let status1 = dev.step_tick(&mut d1, dsd).unwrap();
    let ctls: Vec<[Controls; NUM_CARS]> = items.iter().map(|it| it.ctl).collect();
    let cbuf = dev.upload_controls(&ctls).unwrap();
    let (mut d8, mut r8) = (dev.upload_states(&starts).unwrap(), dev.stream.alloc_zeros::<u32>(n).unwrap());
    dev.sim8(&mut d8, dsd, &cbuf, &mut r8, SIM8_BLOCK, None).unwrap();
    let w = aosoa::words(n);
    let (mut dt, mut rt) = (dev.upload_states(&starts).unwrap(), dev.stream.alloc_zeros::<u32>(n).unwrap());
    let mut tbuf = dev.stream.alloc_zeros::<u32>(N * w).unwrap();
    dev.sim8(&mut dt, dsd, &cbuf, &mut rt, SIM8_BLOCK, Some(&mut tbuf)).unwrap();
    dev.stream.synchronize().unwrap();
    let tw = dev.stream.clone_dtoh(&tbuf).unwrap();
    DeviceRuns {
        status1,
        one: dev.download_states(&d1).unwrap(),
        final8: dev.download_states(&d8).unwrap(),
        result8: dev.stream.clone_dtoh(&r8).unwrap(),
        final_trace: dev.download_states(&dt).unwrap(),
        result_trace: dev.stream.clone_dtoh(&rt).unwrap(),
        trace: (0..N).map(|t| aosoa::unpack(&tw[t * w..(t + 1) * w], n)).collect(),
    }
}

/// One tick's events: the platform host's must be V3's (a failure otherwise), the device's are counted.
fn check_events(rep: &mut Report, class: usize, label: &str, v3: u32, plat: u32, dev: Option<u32>) {
    let e = rep.events.entry(class).or_default();
    e[0] += 1;
    if plat != v3 {
        e[1] += 1;
        rep.fails.add("events: platform host != V3", format!("{label}: {plat:#06x} vs {v3:#06x}"));
    }
    e[2] += u64::from(dev.is_some_and(|x| x != v3));
    for (b, n) in rep.event_bits.iter_mut().enumerate() {
        *n += u64::from(v3 >> b & 1);
    }
}

fn check_one_tick(it: &Item, cfg: &Config, d: &DeviceRuns, j: usize, cmp: &Comparator, rep: &mut Report) {
    let label = format!("{} row {}", CLASSES[it.class], it.row);
    let mut plat = it.start.clone();
    let sp = step_tick_host(Libm::Platform, &mut plat, &cfg.sd);
    let mut cr = it.start.clone();
    let sc = step_tick_host(Libm::Cr, &mut cr, &cfg.sd);
    let dev = &d.one[j];
    if (sp, sc, d.status1[j]) != (0, 0, 0) {
        rep.fails.add("1 tick: status", format!("{label}: platform {sp}, cr {sc}, device {}", d.status1[j]));
    }
    if !(overflow_zero(&plat) && overflow_zero(&cr) && overflow_zero(dev)) {
        rep.fails.add("1 tick: overflow", label.clone());
    }
    check_events(rep, it.class, &format!("{label}, one tick"), it.one_ev, plat.events, Some(dev.events));
    let rel = rep.one.entry(it.class).or_default();
    if !cmp.compare(&plat, &it.one, &mut rel.platform_v3).1 {
        let mut st = Stats::default();
        cmp.compare(&plat, &it.one, &mut st);
        rep.fails.add("1 tick: platform host != V3", format!("{label}: {}", worst(&st)));
    }
    cmp.compare(&cr, dev, &mut rel.cr_device);
    if cr.words() != dev.words() {
        rep.fails.add("1 tick: cr host != device", label.clone());
    }
    cmp.compare(dev, &it.one, &mut rel.device_v3);
    let attr = rep.one_attr.entry(it.class).or_default();
    attr.ran += 1;
    let diff = diff_words(cmp, dev.words(), it.one.words());
    if diff.is_empty() {
        return;
    }
    let run = |mask: u32| {
        let mut x = it.start.clone();
        step_tick_host(Libm::Mixed(mask), &mut x, &cfg.sd);
        x
    };
    if run(0).words() != cr.words() || run(ALL_LIBM).words() != plat.words() {
        rep.fails.add("1 tick: mixed build differs from the cr or the platform build", label.clone());
    }
    let single: Vec<Box<ArenaState>> = (0..LIBM_FUNCS.len()).map(|f| run(1 << f)).collect();
    let single: Vec<Option<&[u32]>> = single.iter().map(|x| Some(x.words())).collect();
    attr.add(cmp, &label, &diff, dev.words(), it.one.words(), cr.words(), plat.words(), &single);
}

fn check_decision(it: &Item, cfg: &Config, d: &DeviceRuns, j: usize, cmp: &Comparator, rep: &mut Report) {
    let label = format!("{} row {}", CLASSES[it.class], it.row);
    let mut plat = it.start.clone();
    let (sp, scored_p, tp) = step_ticks_host(Libm::Platform, &mut plat, &cfg.sd, &it.ctl, N as u32, true);
    let mut cr = it.start.clone();
    let (sc, scored_c, tc) = step_ticks_host(Libm::Cr, &mut cr, &cfg.sd, &it.ctl, N as u32, true);
    let (r8, rt) = (d.result8[j], d.result_trace[j]);
    let scored_d = r8 & SIM8_SCORED_MASK;
    if sp != 0 || sc != 0 || (r8 | rt) & SIM8_UNIMPLEMENTED != 0 {
        rep.fails.add("decision: status", format!("{label}: platform {sp}, cr {sc}, sim8 {r8:08x}, sim8_trace {rt:08x}"));
    }
    if r8 & !(SIM8_SCORED_MASK | SIM8_UNIMPLEMENTED | SIM8_OVERFLOW) != 0 || scored_d > N as u32 {
        rep.fails.add("decision: result word", format!("{label}: {r8:08x}"));
    }
    // the overflow bit says what the state written back holds (the kernel sets it from the state it writes back)
    if (r8 & SIM8_OVERFLOW != 0) == overflow_zero(&d.final8[j]) || (rt & SIM8_OVERFLOW != 0) == overflow_zero(&d.final_trace[j]) {
        rep.fails.add("decision: overflow bit", format!("{label}: {r8:08x} {rt:08x}"));
    }
    let dev_ticks = if scored_d > 0 { scored_d as usize } else { N };
    let ovf = overflow_zero(&plat) && overflow_zero(&cr) && overflow_zero(&d.final8[j]) && tp.iter().chain(&tc).all(overflow_zero);
    if !ovf || !(0..dev_ticks).all(|t| overflow_zero(&d.trace[t][j])) {
        rep.fails.add("decision: overflow", label.clone());
    }
    // the scored tick: the platform host must find V3's, the correctly rounded host and both kernels the same one
    if scored_p != it.scored {
        rep.fails.add("decision: platform scored tick != V3", format!("{label}: {scored_p} vs {}", it.scored));
    }
    if scored_c != scored_d || r8 != rt {
        rep.fails.add("decision: cr / sim8 / sim8_trace scored tick or result differ", format!("{label}: {scored_c} {r8:08x} {rt:08x}"));
    }
    let sc_row = rep.scored.entry(it.class).or_default();
    sc_row[0] += 1;
    sc_row[1] += u64::from(it.scored > 0);
    sc_row[2] += u64::from(scored_d != it.scored);
    // sim8 stops where the host stops: its final state is the trace kernel's and the host's
    if d.final8[j].words() != d.final_trace[j].words() || d.final8[j].words() != cr.words() {
        rep.fails.add("decision: sim8 final != sim8_trace final or cr host final", label.clone());
    }
    // per tick
    for t in 0..tp.len().min(it.ticks.len()) {
        check_events(rep, it.class, &format!("{label}, decision tick {}", t + 1), it.ticks_ev[t], tp[t].events, (t < dev_ticks).then(|| d.trace[t][j].events));
        let rel = rep.eight.entry((it.class, t + 1)).or_default();
        if !cmp.compare(&tp[t], &it.ticks[t], &mut rel.platform_v3).1 {
            let mut st = Stats::default();
            cmp.compare(&tp[t], &it.ticks[t], &mut st);
            rep.fails.add("decision: platform host != V3", format!("{label} tick {}: {}", t + 1, worst(&st)));
        }
    }
    for t in 0..tc.len().min(dev_ticks) {
        let rel = rep.eight.entry((it.class, t + 1)).or_default();
        cmp.compare(&tc[t], &d.trace[t][j], &mut rel.cr_device);
        if tc[t].words() != d.trace[t][j].words() {
            rep.fails.add("decision: cr host != device", format!("{label} tick {}", t + 1));
        }
    }
    let common = dev_ticks.min(it.ticks.len());
    let mut any_diff = false;
    for t in 0..common {
        let rel = rep.eight.entry((it.class, t + 1)).or_default();
        any_diff |= !cmp.compare(&d.trace[t][j], &it.ticks[t], &mut rel.device_v3).1;
        rep.eight_attr.entry(t + 1).or_default().ran += 1;
    }
    if !any_diff {
        return;
    }
    // attribute the device's differences from V3, tick by tick, with the mixed build run over the whole decision
    let run = |mask: u32| {
        let mut x = it.start.clone();
        step_ticks_host(Libm::Mixed(mask), &mut x, &cfg.sd, &it.ctl, N as u32, true).2
    };
    let same = |a: &[ArenaState], b: &[ArenaState]| a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.words() == y.words());
    if !same(&run(0), &tc) || !same(&run(ALL_LIBM), &tp) {
        rep.fails.add("decision: mixed build differs from the cr or the platform build", label.clone());
    }
    let single: Vec<Vec<ArenaState>> = (0..LIBM_FUNCS.len()).map(|f| run(1 << f)).collect();
    for t in 0..common.min(tp.len()) {
        let diff = diff_words(cmp, d.trace[t][j].words(), it.ticks[t].words());
        if diff.is_empty() {
            continue;
        }
        let s: Vec<Option<&[u32]>> = single.iter().map(|x| x.get(t).map(|y| y.words())).collect();
        rep.eight_attr.entry(t + 1).or_default().add(cmp, &format!("{label} tick {}", t + 1), &diff, d.trace[t][j].words(), it.ticks[t].words(), tc[t].words(), tp[t].words(), &s);
    }
}

fn rel_header(key: &str) -> String {
    format!(
        "{key:<22} {:>6} | {:>9} {:>9} | {:>9} {:>9} | {:>9} {:>9} {:>9} {:>9}  {}\n",
        "states", "plat=V3%", "plat wds", "cr=dev%", "cr wds", "dev disc%", "dev=V3%", "dev wds", "max sulp", "device worst field"
    )
}

fn rel_row(key: &str, r: &Rel) -> String {
    let pct = |s: &Stats, v: u64| 100.0 * v as f64 / s.states.max(1) as f64;
    let wds = |s: &Stats| {
        let (d, f) = s.diff_words();
        d + f
    };
    let dv = &r.device_v3;
    format!(
        "{key:<22} {:>6} | {:>8.2}% {:>9} | {:>8.2}% {:>9} | {:>8.2}% {:>8.2}% {:>9} {:>9}  {}\n",
        dv.states,
        pct(&r.platform_v3, r.platform_v3.bit_equal),
        wds(&r.platform_v3),
        pct(&r.cr_device, r.cr_device.bit_equal),
        wds(&r.cr_device),
        pct(dv, dv.discrete_equal),
        pct(dv, dv.bit_equal),
        wds(dv),
        dv.worst_scaled().map_or("-".into(), |(_, v)| format!("{v:.2}")),
        worst(dv)
    )
}

#[test]
fn full_tick_and_sim8_follow_v3() {
    let per_class = per_class();
    let mut configs = all_configs();
    let dev = SimDevice::new(0).expect("CUDA device");
    let mut statics: BTreeMap<u64, DeviceStatic> = BTreeMap::new();
    let mut arenas: BTreeMap<u64, Arena> = BTreeMap::new();
    let cmp = Comparator::new();
    let mut rep = Report::default();
    for (ci, class) in CLASSES.iter().enumerate() {
        let rows = read_rows(class, per_class);
        for chunk in rows.chunks(CHUNK) {
            let items: Vec<Item> = chunk.iter().map(|row| v3_item(&mut arenas, &mut configs, ci, row)).collect();
            let mut by_cfg: BTreeMap<u64, Vec<&Item>> = BTreeMap::new();
            for it in &items {
                by_cfg.entry(it.config).or_default().push(it);
            }
            for (hash, group) in by_cfg {
                let cfg = &configs[&hash];
                let dsd = statics.entry(hash).or_insert_with(|| dev.upload_static(&cfg.sd).unwrap());
                let d = device_runs(&dev, dsd, &group);
                for (j, it) in group.iter().enumerate() {
                    check_one_tick(it, cfg, &d, j, &cmp, &mut rep);
                    check_decision(it, cfg, &d, j, &cmp, &mut rep);
                }
            }
        }
        println!("{class}: {} states done", rows.len());
    }

    // ---- report ----
    let mut s = format!("one tick from each start state ({per_class} per class), the three relations (bit-equal states % and differing words):\n");
    s.push_str(&rel_header("class"));
    let mut total = Rel::default();
    for (ci, r) in &rep.one {
        s.push_str(&rel_row(CLASSES[*ci], r));
        total.platform_v3.merge(&r.platform_v3);
        total.cr_device.merge(&r.cr_device);
        total.device_v3.merge(&r.device_v3);
    }
    s.push_str(&rel_row("all", &total));
    s.push_str(&format!("\none tick, device against V3, the differing fields (all classes):\n{}", total.device_v3.field_table(16)));
    s.push_str(&format!(
        "\n{}",
        render_attribution_rows("one tick: the device's words that differ from V3, by the libm function behind them", "class", rep.one_attr.iter().map(|(c, a)| (CLASSES[*c].to_string(), a)))
    ));
    s.push_str("\none decision (sim8: 8 ticks, a control set per car before each tick), per class and tick:\n");
    s.push_str(&rel_header("class / tick"));
    let mut by_tick: BTreeMap<usize, Rel> = BTreeMap::new();
    for ((ci, t), r) in &rep.eight {
        s.push_str(&rel_row(&format!("{} t{t}", CLASSES[*ci]), r));
        let bt = by_tick.entry(*t).or_default();
        bt.platform_v3.merge(&r.platform_v3);
        bt.cr_device.merge(&r.cr_device);
        bt.device_v3.merge(&r.device_v3);
    }
    s.push_str("\none decision, all classes, per tick:\n");
    s.push_str(&rel_header("tick"));
    for (t, r) in &by_tick {
        s.push_str(&rel_row(&format!("t{t}"), r));
    }
    if let Some(r) = by_tick.get(&N) {
        s.push_str(&format!("\none decision, tick {N}, device against V3, the differing fields (all classes):\n{}", r.device_v3.field_table(16)));
    }
    s.push_str(&format!(
        "\n{}",
        render_attribution_rows("one decision: the device's words that differ from V3 at each tick, by the libm function behind them", "tick", rep.eight_attr.iter().map(|(t, a)| (format!("t{t}"), a)))
    ));
    s.push_str("\nearly exit: states, scored within the decision by V3 (the decision stops after that tick), device scored tick != V3's\n");
    for (ci, [n, scored, dev_diff]) in &rep.scored {
        s.push_str(&format!("{:<22} {n:>6} {scored:>6} {dev_diff:>6}\n", CLASSES[*ci]));
    }
    s.push_str("\nevents (ArenaState.events against V3's event list, the one tick and every decision tick): ticks, platform host != V3, device != V3\n");
    for (ci, [n, p, dv]) in &rep.events {
        s.push_str(&format!("{:<22} {n:>7} {p:>6} {dv:>6}\n", CLASSES[*ci]));
    }
    let seen: Vec<String> = EVENT_BITS.iter().zip(rep.event_bits).filter(|(n, _)| **n != "-").map(|(n, c)| format!("{n} {c}")).collect();
    s.push_str(&format!("V3 ticks with each event bit: {}\n", seen.join(", ")));
    s.push_str(&format!("\nfailures: {:?}\n", rep.fails.count));
    for e in &rep.fails.examples {
        s.push_str(&format!("  {e}\n"));
    }
    println!("{s}");

    assert!(rep.one.len() == CLASSES.len() && rep.one.values().all(|r| r.device_v3.states >= per_class as u64), "every class must be tested");
    assert!(rep.fails.count.is_empty(), "lockstep fails: {:?}", rep.fails.count);
    for (key, a) in rep.one_attr.iter().map(|(c, a)| (CLASSES[*c].to_string(), a)).chain(rep.eight_attr.iter().map(|(t, a)| (format!("tick {t}"), a))) {
        assert_eq!(a.other, 0, "{key}: the device differs from V3 in words no libm function explains:\n{}", a.examples.join("\n"));
    }
}

#[test]
fn sim8_is_deterministic() {
    let per_class = per_class().min(400);
    let mut configs = all_configs();
    // the start states of the most common car configuration (one StaticData per launch)
    let mut by_cfg: BTreeMap<u64, Vec<(Box<ArenaState>, [Controls; NUM_CARS])>> = BTreeMap::new();
    for (ci, class) in CLASSES.iter().enumerate() {
        for row in read_rows(class, per_class) {
            let hash = ArenaDump::from_bytes(row.dump.clone()).expect("arena dump").config_hash();
            let cfg = configs.get_mut(&hash).expect("known config");
            let (s, _) = parse_arena(&row.dump, &cfg.sd).expect("start state parses");
            by_cfg.entry(hash).or_default().push((s, decision_controls(ci, row.index)));
        }
    }
    let (hash, set) = by_cfg.iter().max_by_key(|(_, v)| v.len()).unwrap();
    let n = set.len();
    let dev = SimDevice::new(0).expect("CUDA device");
    let dsd = dev.upload_static(&configs[hash].sd).unwrap();
    // runs sim8 on the states in `order` (order[k] = the set index placed at arena k) and returns them back in set order
    let run = |order: &[usize], block: u32| -> (Vec<Vec<u32>>, Vec<u32>) {
        let states: Vec<Box<ArenaState>> = order.iter().map(|&i| set[i].0.clone()).collect();
        let ctls: Vec<[Controls; NUM_CARS]> = order.iter().map(|&i| set[i].1).collect();
        let mut d = dev.upload_states(&states).unwrap();
        let cbuf = dev.upload_controls(&ctls).unwrap();
        let mut r = dev.stream.alloc_zeros::<u32>(n).unwrap();
        dev.sim8(&mut d, &dsd, &cbuf, &mut r, block, None).unwrap();
        dev.stream.synchronize().unwrap();
        let (out, res) = (dev.download_states(&d).unwrap(), dev.stream.clone_dtoh(&r).unwrap());
        let (mut words, mut results) = (vec![Vec::new(); n], vec![0; n]);
        for (k, &i) in order.iter().enumerate() {
            words[i] = out[k].words().to_vec();
            results[i] = res[k];
        }
        (words, results)
    };
    let ident: Vec<usize> = (0..n).collect();
    let perm = permutation(n);
    let base = run(&ident, 32);
    let moved = (0..n).filter(|&i| base.0[i] != set[i].0.words()).count();
    let scored = base.1.iter().filter(|&&r| r & SIM8_SCORED_MASK != 0).count();
    println!("determinism: {n} states of config {hash:016x} ({} configs in the sample), {moved} changed by the decision, {scored} scored", by_cfg.len());
    assert!(moved > n * 9 / 10, "sim8 left most states unchanged: the test would be vacuous");
    let mut failed = Vec::new();
    for (what, order, block) in [("a second launch", &ident, 32), ("64-thread blocks", &ident, 64), ("permuted arenas, 32", &perm, 32), ("permuted arenas, 64", &perm, 64)] {
        let other = run(order, block);
        let bad = (0..n).filter(|&i| other.0[i] != base.0[i] || other.1[i] != base.1[i]).count();
        println!("determinism, {what}: {bad} of {n} arenas differ");
        if bad > 0 {
            failed.push(what);
        }
    }
    assert!(failed.is_empty(), "sim8 is not deterministic across {failed:?}");
}
