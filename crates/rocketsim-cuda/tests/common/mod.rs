//! Shared machinery of the simulator tests: car configurations and their `StaticData`, the start states (V3 itself plays
//! the scenarios of `v3dump` and exports its arena every few ticks), V3 stepped by phases and exported as `ArenaState`, the
//! phase runners (V3 as the oracle, the host build, the device kernel) and the per-class table. Needs the patched V3
//! (`bash v3/fetch_v3.sh`) and the collision meshes (`ROCKETSIM_MESHES`).
//!
//! The libm: V3 calls the platform C library; the port's default is the correctly rounded one (cuda/common/libm.cuh). So the
//! host phase function exists twice (`rscuda_phase_platform`, which must equal V3, and `rscuda_phase`, which must equal the
//! device bit for bit) and the suite (`run_phase_suite`) runs both beside the device, then attributes every word in which the
//! device differs from V3 to the libm function that causes it, with a third build that switches one function at a time.
#![allow(dead_code)]

use std::collections::BTreeMap;

use rocketsim_cuda::{
    compare::{Comparator, Stats},
    dump::{Template, parse_arena, parse_static, write_arena},
    layout::{ArenaState, Controls, NUM_CARS, StaticData},
};
use rocketsim::{Arena, PHASES, PHASE_NAMES, dump::ArenaDump};
use v3dump::{
    bots::Mode,
    meshes::init_meshes,
    scenario::{KIND_LABELS, Kind, Spec, build_arena, sample_states, specs},
};

// ---- the decision tests' controls and arena orders (tests/step.rs, tests/speed.rs) ---------------------------------------
pub fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// The decision's controls: a fixed pseudo-random pick per car, determined by (class, row, car). Each analog axis is -1, 0
/// or 1; jump, boost and handbrake are on or off.
pub fn decision_controls(class: usize, row: usize) -> [Controls; NUM_CARS] {
    std::array::from_fn(|car| {
        let h = mix(((class as u64) << 40) ^ ((row as u64) << 2) ^ car as u64);
        let axis = |k: u32| ((h >> (k * 4)) % 3) as f32 - 1.0;
        let flag = |k: u32| u32::from(h >> (24 + k) & 1 != 0);
        Controls { throttle: axis(0), steer: axis(1), pitch: axis(2), yaw: axis(3), roll: axis(4), jump: flag(0), boost: flag(1), handbrake: flag(2) }
    })
}

/// A fixed shuffle of 0..n (perm[k] = the index placed at arena k).
pub fn permutation(n: usize) -> Vec<usize> {
    let mut perm: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        perm.swap(i, (mix(i as u64 ^ 0xd5) % (i as u64 + 1)) as usize);
    }
    perm
}

// ---- car configurations ---------------------------------------------------------------------------------------------------
/// One car configuration: the arena builder, its static data (BVH, poses, wheels ...) and the template of what ArenaState
/// leaves out. Two arenas with the same `hash` have the same `StaticData`.
pub struct Config {
    pub spec: Spec,
    pub hash: u64,
    pub sd: Box<StaticData>,
    pub template: Option<Template>,
}

pub fn spec_for(blue_first: bool, bodies: [usize; 2]) -> Spec {
    Spec {
        id: 0,
        kind: Kind::Play,
        blue_first,
        bodies,
        modes: [Mode::Random, Mode::Random],
        seed: 7,
        ticks: 0,
        checkpoints: vec![],
    }
}

pub fn make_config(spec: &Spec) -> Config {
    let arena = build_arena(spec);
    let sd = parse_static(arena.export_static().as_bytes()).expect("static dump parses");
    Config { spec: spec.clone(), hash: arena.config_hash(), sd, template: None }
}

/// All 98 car configurations a scenario can use (7 bodies x 7 bodies x blue first or not), by config hash.
pub fn all_configs() -> BTreeMap<u64, Config> {
    init_meshes();
    let mut out = BTreeMap::new();
    for blue_first in [true, false] {
        for a in 0..7 {
            for b in 0..7 {
                let c = make_config(&spec_for(blue_first, [a, b]));
                out.entry(c.hash).or_insert(c);
            }
        }
    }
    out
}

/// The V3 arena as an `ArenaState` (a tick or a phase boundary). The first parse of a configuration fixes its template.
pub fn snap(arena: &Arena, cfg: &mut Config) -> Box<ArenaState> {
    let d = arena.export_state();
    let (s, t) = parse_arena(d.as_bytes(), &cfg.sd).unwrap_or_else(|e| panic!("arena dump does not parse: {e}"));
    if cfg.template.is_none() {
        cfg.template = Some(t);
    }
    s
}

/// An `ArenaState` as an imported V3 arena: write the dump, import it into `arena`.
pub fn load(arena: &mut Arena, cfg: &Config, s: &ArenaState) -> Result<(), String> {
    let bytes = write_arena(s, &cfg.sd, cfg.template.as_ref().expect("template"))?;
    arena.import_state(&ArenaDump::from_bytes(bytes)?)
}

// ---- the start states ---------------------------------------------------------------------------------------------------------
/// The classes of start state: the kind of scenario V3 played to reach it (kickoff, free play, head-on drives, pinches in
/// corners and goal mouths, pad pickups).
pub const CLASSES: [&str; 5] = KIND_LABELS;

/// Ticks between two exported states of one scenario (odd, so the states fall at every phase of the 8-tick decision).
pub const SAMPLE_EVERY: u32 = 97;

/// States per class: `ROCKETSIM_CUDA_SAMPLES`, else `default`.
pub fn samples_per_class(default: usize) -> usize {
    std::env::var("ROCKETSIM_CUDA_SAMPLES").ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// Ticks between the exported states of `spec` when it yields `count` of them: `SAMPLE_EVERY`, or less (still odd) for a
/// scenario too short to hold `count` states that far apart.
fn every(spec: &Spec, count: usize) -> u32 {
    let q = (spec.ticks.saturating_sub(1) / count.max(1) as u32).clamp(1, SAMPLE_EVERY);
    if q % 2 == 0 { q - 1 } else { q }
}

pub struct Row {
    pub class: String,
    pub index: usize,
    pub source: String,
    pub dump: Vec<u8>,
}

/// `n` states of a class: V3 plays each scenario of that kind (`v3dump::scenario::specs`) from its start and exports its
/// arena every `SAMPLE_EVERY` ticks or less; the scenarios share the count.
pub fn read_rows(class: &str, n: usize) -> Vec<Row> {
    let specs: Vec<Spec> = specs().into_iter().filter(|s| s.kind.label() == class).collect();
    assert!(!specs.is_empty(), "no scenario of class {class}");
    let per_spec = n.div_ceil(specs.len());
    let next = std::sync::atomic::AtomicUsize::new(0);
    let mut played: Vec<Vec<Vec<u8>>> = vec![Vec::new(); specs.len()];
    let results = std::sync::Mutex::new(&mut played);
    let threads = std::thread::available_parallelism().map_or(1, |p| p.get());
    std::thread::scope(|sc| {
        for _ in 0..threads {
            sc.spawn(|| {
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if i >= specs.len() {
                        break;
                    }
                    let dumps: Vec<Vec<u8>> = sample_states(&specs[i], every(&specs[i], per_spec), per_spec).into_iter().map(|d| d.0).collect();
                    results.lock().unwrap()[i] = dumps;
                }
            });
        }
    });
    let mut out = Vec::with_capacity(n);
    for (spec, dumps) in specs.iter().zip(played) {
        for dump in dumps {
            if out.len() < n {
                out.push(Row { class: class.to_string(), index: out.len(), source: spec.name(), dump });
            }
        }
    }
    out
}

// ---- boundary samples ---------------------------------------------------------------------------------------------------------
/// One start state with V3's state at every phase boundary: `bounds[k]` is the state before phase k, `bounds[PHASES]` the
/// state after P8.
pub struct Sample {
    pub class: String,
    pub config: u64,
    pub bounds: Vec<Box<ArenaState>>,
}

/// Steps `arena` (already holding the start state) through the phases and snaps every boundary.
pub fn trace_boundaries(arena: &mut Arena, cfg: &mut Config) -> Vec<Box<ArenaState>> {
    let mut v = Vec::with_capacity(PHASES + 1);
    v.push(snap(arena, cfg));
    for k in 0..PHASES {
        arena.step_phase(k);
        v.push(snap(arena, cfg));
    }
    v
}

/// Boundary samples of start states. Returns the configs too (their templates are set).
pub fn class_samples(per_class: usize, configs: &mut BTreeMap<u64, Config>) -> Vec<Sample> {
    let mut arenas: BTreeMap<u64, Arena> = BTreeMap::new();
    let mut out = Vec::new();
    for class in CLASSES {
        for row in read_rows(class, per_class) {
            let d = ArenaDump::from_bytes(row.dump.clone()).expect("a start state is an arena dump");
            let hash = d.config_hash();
            let cfg = configs.get_mut(&hash).unwrap_or_else(|| panic!("{class} row {}: unknown config hash {hash:016x}", row.index));
            let arena = arenas.entry(hash).or_insert_with(|| build_arena(&cfg.spec));
            arena.import_state(&d).unwrap_or_else(|e| panic!("{class} row {}: import: {e}", row.index));
            out.push(Sample { class: class.to_string(), config: hash, bounds: trace_boundaries(arena, cfg) });
        }
    }
    out
}

// ---- phase runners --------------------------------------------------------------------------------------------------------------
pub trait PhaseRunner {
    fn name(&self) -> String;
    /// Runs phase `k` from each input (boundary k of the samples of configuration `config`); returns the state after the
    /// phase and the status of each (0 = ran, negative = the phase is not implemented or failed).
    fn run(&mut self, k: usize, config: &mut Config, inputs: &[&ArenaState]) -> (Vec<Box<ArenaState>>, Vec<i32>);
}

/// V3 itself: rebuild the V3 dump from the `ArenaState`, import it, run the phase, snap. Zero differences prove the harness
/// (the writer, the importer, the comparator and the driver); it is what a ported phase is held to.
pub struct Oracle {
    arenas: BTreeMap<u64, Arena>,
}
impl Oracle {
    pub fn new() -> Self {
        Self { arenas: BTreeMap::new() }
    }
}
impl PhaseRunner for Oracle {
    fn name(&self) -> String {
        "oracle (V3 itself)".into()
    }
    fn run(&mut self, k: usize, config: &mut Config, inputs: &[&ArenaState]) -> (Vec<Box<ArenaState>>, Vec<i32>) {
        let arena = self.arenas.entry(config.hash).or_insert_with(|| build_arena(&config.spec));
        let mut out = Vec::new();
        for s in inputs {
            load(arena, config, s).unwrap_or_else(|e| panic!("oracle import: {e}"));
            arena.step_phase(k);
            out.push(snap(arena, config));
        }
        let n = out.len();
        (out, vec![0; n])
    }
}

/// Which libm a host phase function is built with.
#[derive(Clone, Copy, PartialEq)]
pub enum HostLibm {
    /// `rscuda_phase`: the correctly rounded libm (the default; equal to the device bit for bit).
    Cr,
    /// `rscuda_phase_platform`: the platform C library, as V3 calls it (the porting verification).
    Platform,
}

/// The host C ABI (the single-source phase function compiled for the CPU).
pub struct HostPhase(pub HostLibm);
impl PhaseRunner for HostPhase {
    fn name(&self) -> String {
        match self.0 {
            HostLibm::Cr => "host rscuda_phase (correctly rounded libm)".into(),
            HostLibm::Platform => "host rscuda_phase_platform (platform libm)".into(),
        }
    }
    fn run(&mut self, k: usize, config: &mut Config, inputs: &[&ArenaState]) -> (Vec<Box<ArenaState>>, Vec<i32>) {
        let mut out = Vec::new();
        let mut status = Vec::new();
        for s in inputs {
            let mut x = Box::new(**s);
            status.push(match self.0 {
                HostLibm::Cr => rocketsim_cuda::ffi::phase_host(k, &mut x, &config.sd),
                HostLibm::Platform => rocketsim_cuda::ffi::phase_host_platform(k, &mut x, &config.sd),
            });
            out.push(x);
        }
        (out, status)
    }
}

/// `phase_k`, the device kernel (AoSoA-32 buffers, `StaticData` uploaded once per configuration).
pub struct DevicePhase {
    pub dev: rocketsim_cuda::device::SimDevice,
    statics: BTreeMap<u64, rocketsim_cuda::device::DeviceStatic>,
}
impl DevicePhase {
    pub fn new() -> Self {
        Self { dev: rocketsim_cuda::device::SimDevice::new(0).expect("CUDA device"), statics: BTreeMap::new() }
    }
}
impl PhaseRunner for DevicePhase {
    fn name(&self) -> String {
        "device phase_k".into()
    }
    fn run(&mut self, k: usize, config: &mut Config, inputs: &[&ArenaState]) -> (Vec<Box<ArenaState>>, Vec<i32>) {
        let sd = self.statics.entry(config.hash).or_insert_with(|| self.dev.upload_static(&config.sd).unwrap());
        let owned: Vec<Box<ArenaState>> = inputs.iter().map(|s| Box::new(**s)).collect();
        let mut d = self.dev.upload_states(&owned).unwrap();
        let status = self.dev.phase(k, &mut d, sd).unwrap();
        (self.dev.download_states(&d).unwrap(), status)
    }
}

// ---- the table ------------------------------------------------------------------------------------------------------------------
#[derive(Default, Clone)]
pub struct Cell {
    pub stats: Stats,
    /// The comparison over the states the runner completed (status 0) only: a phase that is ported for some pair types
    /// (P3b) reports the rest as failed, and a required phase is asserted on these.
    pub ok: Stats,
    pub ran: u64,
    pub failed: u64,
}

pub struct Table {
    pub runner: String,
    /// (class, phase) -> cell
    pub cells: BTreeMap<(String, usize), Cell>,
}

/// For every sample and every phase: from V3's boundary k run the phase and compare with V3's boundary k + 1.
pub fn run_phase_tests(samples: &[Sample], configs: &mut BTreeMap<u64, Config>, runner: &mut dyn PhaseRunner, only: &[usize]) -> Table {
    let cmp = Comparator::new();
    let mut cells: BTreeMap<(String, usize), Cell> = BTreeMap::new();
    for k in 0..PHASES {
        if !only.is_empty() && !only.contains(&k) {
            continue;
        }
        // group by configuration (one StaticData per kernel launch)
        let mut by_cfg: BTreeMap<u64, Vec<&Sample>> = BTreeMap::new();
        for s in samples {
            by_cfg.entry(s.config).or_default().push(s);
        }
        for (hash, group) in by_cfg {
            let cfg = configs.get_mut(&hash).unwrap();
            let inputs: Vec<&ArenaState> = group.iter().map(|s| &*s.bounds[k]).collect();
            let (got, status) = runner.run(k, cfg, &inputs);
            for ((s, g), st) in group.iter().zip(&got).zip(&status) {
                let cell = cells.entry((s.class.clone(), k)).or_default();
                if *st == 0 {
                    cell.ran += 1;
                    cmp.compare(g, &s.bounds[k + 1], &mut cell.ok);
                } else {
                    cell.failed += 1;
                }
                cmp.compare(g, &s.bounds[k + 1], &mut cell.stats);
            }
        }
    }
    Table { runner: runner.name(), cells }
}

impl Table {
    /// Per-class, per-phase rows: states, ran, failed (not implemented, or blocked on pair types not ported), then, over the
    /// states that ran (all of them when none ran): discrete-identical, bit-identical, differing words, worst field.
    pub fn render(&self) -> String {
        let mut s = format!("phase test, runner: {}\n", self.runner);
        s.push_str(&format!(
            "{:<17} {:<4} {:>6} {:>6} {:>7} {:>9} {:>9} {:>9} {:>9} {:>9}  {}\n",
            "class", "ph", "states", "ran", "failed", "discrete%", "bitexact%", "int diffs", "flt diffs", "max sulp", "worst field (max ulp)"
        ));
        for ((class, k), c) in &self.cells {
            let st = if c.ran > 0 { &c.ok } else { &c.stats };
            let n = st.states.max(1) as f64;
            let (d, f) = st.diff_words();
            let worst = st.worst().map_or("-".to_string(), |(name, u)| {
                if u == u64::MAX { format!("{name} (discrete)") } else { format!("{name} ({u})") }
            });
            let sulp = st.worst_scaled().map_or("-".to_string(), |(_, v)| format!("{v:.2}"));
            s.push_str(&format!(
                "{:<17} {:<4} {:>6} {:>6} {:>7} {:>8.2}% {:>8.2}% {:>9} {:>9} {:>9}  {}\n",
                class,
                PHASE_NAMES[*k],
                c.stats.states,
                c.ran,
                c.failed,
                100.0 * st.discrete_equal as f64 / n,
                100.0 * st.bit_equal as f64 / n,
                d,
                f,
                sulp,
                worst
            ));
        }
        s
    }

    /// The same columns, one row per phase (the classes added up), then the field table of the differences per phase.
    pub fn render_by_phase(&self, fields: usize) -> String {
        let mut s = format!("{}: totals per phase
", self.runner);
        s.push_str(&format!(
            "{:<4} {:>7} {:>7} {:>7} {:>9} {:>9} {:>9} {:>9} {:>9}  {}
",
            "ph", "states", "ran", "failed", "discrete%", "bitexact%", "int diffs", "flt diffs", "max sulp", "worst field (max ulp)"
        ));
        let by = self.by_phase();
        for (k, c) in &by {
            let st = if c.ran > 0 { &c.ok } else { &c.stats };
            let n = st.states.max(1) as f64;
            let (d, f) = st.diff_words();
            let worst = st.worst().map_or("-".to_string(), |(name, u)| if u == u64::MAX { format!("{name} (discrete)") } else { format!("{name} ({u})") });
            let sulp = st.worst_scaled().map_or("-".to_string(), |(_, v)| format!("{v:.2}"));
            s.push_str(&format!(
                "{:<4} {:>7} {:>7} {:>7} {:>8.2}% {:>8.2}% {:>9} {:>9} {:>9}  {}
",
                PHASE_NAMES[*k],
                c.stats.states,
                c.ran,
                c.failed,
                100.0 * st.discrete_equal as f64 / n,
                100.0 * st.bit_equal as f64 / n,
                d,
                f,
                sulp,
                worst
            ));
        }
        for (k, c) in &by {
            let st = if c.ran > 0 { &c.ok } else { &c.stats };
            if st.diff_words() != (0, 0) && c.ran > 0 {
                s.push_str(&format!("
{} differences, states that ran, all classes:
{}", PHASE_NAMES[*k], st.field_table(fields)));
            }
        }
        s
    }

    pub fn total(&self) -> Cell {
        let mut t = Cell::default();
        for c in self.cells.values() {
            t.stats.merge(&c.stats);
            t.ok.merge(&c.ok);
            t.ran += c.ran;
            t.failed += c.failed;
        }
        t
    }

    /// Per-phase totals over the classes.
    pub fn by_phase(&self) -> BTreeMap<usize, Cell> {
        let mut m: BTreeMap<usize, Cell> = BTreeMap::new();
        for ((_, k), c) in &self.cells {
            let t = m.entry(*k).or_default();
            t.stats.merge(&c.stats);
            t.ok.merge(&c.ok);
            t.ran += c.ran;
            t.failed += c.failed;
        }
        m
    }
}


// ---- the suite: platform host, correctly rounded host, device, and the attribution of the device's differences ----------------
/// The libm functions of `lf::`, in the bit order of `rscuda_phase_mixed`'s mask.
pub const LIBM_FUNCS: [&str; 7] = ["sin", "cos", "tan", "atan2", "pow", "atan2_64", "hypot64"];

/// The words in which the device differs from V3 at one phase, attributed to the libm call that causes them: a word is
/// attributed to function f when making f alone the platform library (the other four stay correctly rounded) changes the word;
/// to `combo` when no single function changes it but making all five the platform library does (two functions together);
/// otherwise the difference is not the libm's (`other`).
#[derive(Default, Clone)]
pub struct Attribution {
    /// States that ran on the device.
    pub ran: u64,
    /// ... of which differ from V3 in some word of the verdict fields.
    pub states_diff: u64,
    pub words: u64,
    /// Words changed by each function alone (a word that two functions change counts under both).
    pub by_func: [u64; 7],
    pub combo: u64,
    pub other: u64,
    pub examples: Vec<String>,
}

/// The words in which `a` and `b` differ, outside the excluded fields.
pub fn diff_words(cmp: &Comparator, a: &[u32], b: &[u32]) -> Vec<usize> {
    (0..a.len()).filter(|&w| a[w] != b[w] && !cmp.is_excluded_word(w)).collect()
}

impl Attribution {
    /// Adds one state whose device result `dev` differs from V3's `v3` in the words `diff` (`diff_words`, not empty). From the same
    /// input, `cr` and `plat` are the correctly rounded and the platform host builds' results and `single[f]` the mixed build's with
    /// function f alone on the platform library (None: that run did not reach this state, e.g. it stopped at a goal earlier, which
    /// counts as a change). `label` names the state in an example.
    #[allow(clippy::too_many_arguments)]
    pub fn add(&mut self, cmp: &Comparator, label: &str, diff: &[usize], dev: &[u32], v3: &[u32], cr: &[u32], plat: &[u32], single: &[Option<&[u32]>]) {
        self.states_diff += 1;
        for &w in diff {
            self.words += 1;
            let hits: Vec<usize> = (0..LIBM_FUNCS.len()).filter(|&f| single[f].is_none_or(|x| x[w] != cr[w])).collect();
            if !hits.is_empty() {
                for f in hits {
                    self.by_func[f] += 1;
                }
            } else if plat[w] != cr[w] {
                self.combo += 1;
            } else {
                self.other += 1;
                if self.examples.len() < 8 {
                    self.examples.push(format!("{label}: word {w} ({}) device {:08x} V3 {:08x} platform host {:08x}", cmp.group_name(w), dev[w], v3[w], plat[w]));
                }
            }
        }
    }
}

/// The attribution table: one row per key (`col` names the key column).
pub fn render_attribution_rows<'a>(title: &str, col: &str, rows: impl IntoIterator<Item = (String, &'a Attribution)>) -> String {
    let mut s = format!("{title}
");
    s.push_str("(a word counts under every function that changes it when it alone is the platform library; combo: only changed by two or more together; other: not the libm)
");
    s.push_str(&format!(
        "{:<17} {:>8} {:>10} {:>9}  {}  {:>7} {:>7}
",
        col,
        "ran",
        "states!=V3",
        "words!=V3",
        LIBM_FUNCS.iter().map(|f| format!("{f:>7}")).collect::<Vec<_>>().join(" "),
        "combo",
        "other"
    ));
    for (key, a) in rows {
        s.push_str(&format!(
            "{:<17} {:>8} {:>10} {:>9}  {}  {:>7} {:>7}
",
            key,
            a.ran,
            a.states_diff,
            a.words,
            a.by_func.iter().map(|n| format!("{n:>7}")).collect::<Vec<_>>().join(" "),
            a.combo,
            a.other
        ));
    }
    s
}

pub struct Suite {
    /// Each runner against V3, per class and phase.
    pub platform: Table,
    pub cr: Table,
    pub device: Table,
    /// Host (correctly rounded) against the device: must be empty everywhere.
    pub cr_vs_device: Table,
    /// Per phase.
    pub attribution: BTreeMap<usize, Attribution>,
}

fn cell_add(cells: &mut BTreeMap<(String, usize), Cell>, cmp: &Comparator, class: &str, k: usize, st: i32, got: &ArenaState, want: &ArenaState) {
    let cell = cells.entry((class.to_string(), k)).or_default();
    if st == 0 {
        cell.ran += 1;
        cmp.compare(got, want, &mut cell.ok);
    } else {
        cell.failed += 1;
    }
    cmp.compare(got, want, &mut cell.stats);
}

/// For every sample and every phase k (or those in `only`): from V3's boundary k run phase k on the platform host, the
/// correctly rounded host and the device; compare each with V3's boundary k + 1 and the last two with each other; and attribute
/// the device's differences from V3 to libm calls (a third host build, one function at a time on the platform library).
pub fn run_phase_suite(samples: &[Sample], configs: &mut BTreeMap<u64, Config>, dev: &mut DevicePhase, only: &[usize]) -> Suite {
    let cmp = Comparator::new();
    let (mut platform, mut cr, mut device, mut pair): (BTreeMap<_, Cell>, BTreeMap<_, Cell>, BTreeMap<_, Cell>, BTreeMap<_, Cell>) = Default::default();
    let mut attribution: BTreeMap<usize, Attribution> = BTreeMap::new();
    let (mut host_plat, mut host_cr) = (HostPhase(HostLibm::Platform), HostPhase(HostLibm::Cr));
    for k in 0..PHASES {
        if !only.is_empty() && !only.contains(&k) {
            continue;
        }
        let mut by_cfg: BTreeMap<u64, Vec<&Sample>> = BTreeMap::new();
        for s in samples {
            by_cfg.entry(s.config).or_default().push(s);
        }
        for (hash, group) in by_cfg {
            let cfg = configs.get_mut(&hash).unwrap();
            let inputs: Vec<&ArenaState> = group.iter().map(|s| &*s.bounds[k]).collect();
            let (op, sp) = host_plat.run(k, cfg, &inputs);
            let (oc, sc) = host_cr.run(k, cfg, &inputs);
            let (od, sd_) = dev.run(k, cfg, &inputs);
            let attr = attribution.entry(k).or_default();
            for (i, s) in group.iter().enumerate() {
                let want = &s.bounds[k + 1];
                assert_eq!((sp[i], sc[i]), (sd_[i], sd_[i]), "{} phase {}: the host builds and the device report different statuses", s.class, PHASE_NAMES[k]);
                cell_add(&mut platform, &cmp, &s.class, k, sp[i], &op[i], want);
                cell_add(&mut cr, &cmp, &s.class, k, sc[i], &oc[i], want);
                cell_add(&mut device, &cmp, &s.class, k, sd_[i], &od[i], want);
                cell_add(&mut pair, &cmp, &s.class, k, sc[i], &oc[i], &od[i]);
                if sd_[i] != 0 {
                    continue;
                }
                attr.ran += 1;
                let (wd, wt) = (od[i].words(), want.words());
                let diff = diff_words(&cmp, wd, wt);
                if diff.is_empty() {
                    continue;
                }
                // one function at a time on the platform library; mask 0 and mask 127 must reproduce the two builds
                let run_mask = |mask: u32| {
                    let mut x = Box::new(*inputs[i]);
                    let st = rocketsim_cuda::ffi::phase_host_mixed(k, &mut x, &cfg.sd, mask);
                    assert_eq!(st, 0);
                    x
                };
                let (m0, m_all) = (run_mask(0), run_mask(127));
                assert!(m0.words() == oc[i].words(), "mixed build, mask 0, differs from the correctly rounded build");
                assert!(m_all.words() == op[i].words(), "mixed build, all functions on the platform library, differs from the platform build");
                let single: Vec<Box<ArenaState>> = (0..7).map(|f| run_mask(1 << f)).collect();
                let single: Vec<Option<&[u32]>> = single.iter().map(|x| Some(x.words())).collect();
                attr.add(&cmp, &format!("{} {}", s.class, PHASE_NAMES[k]), &diff, wd, wt, oc[i].words(), op[i].words(), &single);
            }
        }
    }
    let table = |runner: &str, cells| Table { runner: runner.to_string(), cells };
    Suite {
        platform: table(&host_plat.name(), platform),
        cr: table(&host_cr.name(), cr),
        device: table(&dev.name(), device),
        cr_vs_device: table("host (correctly rounded libm) against device phase_k", pair),
        attribution,
    }
}

impl Suite {
    pub fn render_attribution(&self) -> String {
        render_attribution_rows(
            "device phase_k against V3: the differing words, attributed to the libm call that causes them",
            "ph",
            self.attribution.iter().map(|(k, a)| (PHASE_NAMES[*k].to_string(), a)),
        )
    }
}
