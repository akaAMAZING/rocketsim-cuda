//! The field-by-field comparator of two `ArenaState`s. Discrete words (integers, flags, indices, counts) must be equal;
//! float words get an ulp distance, collected per field (the dotted field path with array indices removed) and per state
//! class. Nothing is hidden: fields that V3 recomputes are listed in `EXCLUDED` and reported apart from the verdict.
//!
//! Beside the plain ulp distance every differing float word gets a scale-aware one (`sulp`): the difference in units of the ulp
//! of max(|a|, |b|, scale), where the scale is the natural size of the field family. An entry near zero then does not read as
//! 1e9 ulp when its neighbours are of order 1. The scales (`scale_rule`):
//!   * rotation matrices (`rot_mat`, the 3x3 part of a transform), quaternions, unit vectors (normals, axle directions): 1;
//!   * the translation part of a transform and every other 3-vector (positions, velocities, angular velocities, contact
//!     points, impulses, torques) and each corner of a box: the largest |component| of that vector in the two states;
//!   * inertia tensors (`inv_inertia_tensor_world`): the largest |entry| of that 3x3 tensor in the two states;
//!   * boost: 100 (its range); every other scalar (timers in seconds, friction and restitution coefficients, angles,
//!     lengths in uu, controls): 1.

use std::collections::BTreeMap;

use crate::layout::{ArenaState, Kind, arena_leaves};

/// Fields kept out of the pass/fail verdict (still compared and reported, marked excluded).
///   overflow: the cap counters of the port; V3 has none, so they are 0 in every V3 state and a port that overflows must not be
///   hidden by... itself: they are reported, and a non-zero counter fails the run in its own assertion (tests/phases.rs).
///   events: the port's record of the tick's V3 events (EventBit bits). A V3 dump carries no events,
///   so a V3 state reads 0 there; the word is held to V3's own event list by tests/step.rs, not by this comparator.
/// The contact records (boundaries 5 to 8, P3b to P7): V3 lists every record; the port stores only the car-ball and car-car ones
/// (`records`, `n_records`, in V3's order) and folds the rest into `fold_events` and `fold_car_normal`. A V3
/// state is mapped the same way before it reaches this comparator: `dump::fold_records` (the importer, used for every V3 dump
/// including `tests/common`'s `snap`) is the documented mapping, and `dump::unfold_records` its inverse for the writer. So every
/// word of the folded fields is compared like any other (discrete for the bits and counts, ulp for the normals).
/// Fields V3 recomputes and the port does not store (companion_id, the manifold thresholds, applied_impulse, the island
/// memo ...) are not in `ArenaState` at all; the list is in cuda/sim/arena_state.cuh.
pub const EXCLUDED: &[&str] = &["overflow", "events"];

/// ulp buckets: equal value with different bits (+0 against -0), 1, 2-4, 5-16, 17-1024, over 1024, NaN mismatch.
pub const BUCKETS: [&str; 7] = ["+-0", "1", "2-4", "5-16", "17-1k", ">1k", "nan"];
/// Scale-aware ulp buckets: up to 1, 4, 16, 1024, over 1024 (and NaN against a number counts as over 1024).
pub const SBUCKETS: [&str; 5] = ["<=1", "<=4", "<=16", "<=1k", ">1k"];

/// Where the scale of a float word comes from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Scale {
    Fixed(f32),
    /// The largest |word| over words `start..start + len` of the state (both states are looked at).
    Block { start: usize, len: usize },
}

#[derive(Clone, Default)]
pub struct FieldStat {
    pub words: u64,
    pub discrete_diff: u64,
    pub float_diff: u64,
    pub max_ulp: u64,
    pub hist: [u64; 7],
    /// Largest scale-aware ulp distance (see the module doc) and its histogram.
    pub max_sulp: f64,
    pub shist: [u64; 5],
    /// First differing example: (state number within the stats, word index, a, b).
    pub first: Option<(u64, usize, u32, u32)>,
}

impl FieldStat {
    fn merge(&mut self, o: &FieldStat) {
        self.words += o.words;
        self.discrete_diff += o.discrete_diff;
        self.float_diff += o.float_diff;
        self.max_ulp = self.max_ulp.max(o.max_ulp);
        self.max_sulp = self.max_sulp.max(o.max_sulp);
        for i in 0..7 {
            self.hist[i] += o.hist[i];
        }
        for i in 0..5 {
            self.shist[i] += o.shist[i];
        }
        if self.first.is_none() {
            self.first = o.first;
        }
    }
}

#[derive(Clone, Default)]
pub struct Stats {
    pub states: u64,
    /// States with no discrete difference outside the excluded fields.
    pub discrete_equal: u64,
    /// States with no difference at all outside the excluded fields (every word bit-equal).
    pub bit_equal: u64,
    pub fields: BTreeMap<String, FieldStat>,
}

pub struct Comparator {
    /// group id of each word
    group: Vec<u32>,
    kind: Vec<Kind>,
    names: Vec<String>,
    excluded: Vec<bool>,
    scale: Vec<Scale>,
}

fn ordered(u: u32) -> i64 {
    if u >> 31 == 1 { -i64::from(u & 0x7fff_ffff) } else { i64::from(u) }
}

/// ulp distance of two float bit patterns; None when either is NaN and the bits differ.
pub fn ulp(x: u32, y: u32) -> Option<u64> {
    if f32::from_bits(x).is_nan() || f32::from_bits(y).is_nan() {
        return if x == y { Some(0) } else { None };
    }
    Some((ordered(x) - ordered(y)).unsigned_abs())
}

/// ulp distance of two float bit patterns in units of the ulp of max(|a|, |b|, scale); infinity when either is NaN (and the
/// bits differ) or an infinity against a different value.
pub fn sulp(x: u32, y: u32, scale: f32) -> f64 {
    let (a, b) = (f32::from_bits(x), f32::from_bits(y));
    if x == y {
        return 0.0;
    }
    if !a.is_finite() || !b.is_finite() {
        return f64::INFINITY;
    }
    let m = a.abs().max(b.abs()).max(scale.abs());
    let e = (m.to_bits() >> 23) & 0xff; // the ulp of a subnormal or zero is 2^-149, as for the smallest normals
    let ulp = 2f64.powi(e.max(1) as i32 - 150);
    (f64::from(a) - f64::from(b)).abs() / ulp
}

/// The scale of the float word `idx` of a run of `run_len` words of the field `name`, which starts at word `run_start`.
fn scale_rule(name: &str, run_len: usize, idx: usize, run_start: usize) -> Scale {
    if name.ends_with("inv_inertia_tensor_world") && run_len % 9 == 0 {
        return Scale::Block { start: run_start + idx / 9 * 9, len: 9 };
    }
    match run_len {
        12 => {
            // a transform: 3x3 basis, then the translation
            if idx % 12 < 9 { Scale::Fixed(1.0) } else { Scale::Block { start: run_start + idx / 12 * 12 + 9, len: 3 } }
        }
        9 | 4 => Scale::Fixed(1.0),
        3 | 6 if name.contains("normal") || name.ends_with("axle_dir") => Scale::Fixed(1.0),
        3 | 6 => Scale::Block { start: run_start + idx / 3 * 3, len: 3 },
        1 if name.ends_with("boost") => Scale::Fixed(100.0),
        _ => Scale::Fixed(1.0),
    }
}

fn is_excluded(name: &str) -> bool {
    EXCLUDED.iter().any(|e| name == *e || name.strip_prefix(e).is_some_and(|r| r.starts_with('#')))
}

impl Default for Comparator {
    fn default() -> Self {
        Self::new()
    }
}

impl Comparator {
    pub fn new() -> Self {
        let leaves = arena_leaves();
        let mut names: Vec<String> = Vec::new();
        let mut ids: BTreeMap<String, u32> = BTreeMap::new();
        let mut group = Vec::with_capacity(leaves.len());
        let mut kind = Vec::with_capacity(leaves.len());
        for (name, k) in leaves {
            let id = *ids.entry(name.clone()).or_insert_with(|| {
                names.push(name.clone());
                (names.len() - 1) as u32
            });
            group.push(id);
            kind.push(k);
        }
        let excluded = names.iter().map(|n| is_excluded(n)).collect();
        // runs of consecutive words of one field: a vector is a run of 3, a matrix of 9, a transform of 12 ...
        let mut scale = Vec::with_capacity(group.len());
        let mut start = 0;
        while start < group.len() {
            let mut end = start;
            while end < group.len() && group[end] == group[start] {
                end += 1;
            }
            for w in start..end {
                scale.push(scale_rule(&names[group[start] as usize], end - start, w - start, start));
            }
            start = end;
        }
        Self { group, kind, names, excluded, scale }
    }

    pub fn group_name(&self, word: usize) -> &str {
        &self.names[self.group[word] as usize]
    }

    /// True for the words of an excluded field (reported, never part of a verdict).
    pub fn is_excluded_word(&self, word: usize) -> bool {
        self.excluded[self.group[word] as usize]
    }

    /// The scale-aware ulp distance of word `w` between two states (see the module doc).
    pub fn sulp_at(&self, w: usize, wa: &[u32], wb: &[u32]) -> f64 {
        let scale = match self.scale[w] {
            Scale::Fixed(f) => f,
            Scale::Block { start, len } => (start..start + len).fold(0f32, |m, i| m.max(f32::from_bits(wa[i]).abs()).max(f32::from_bits(wb[i]).abs())),
        };
        sulp(wa[w], wb[w], scale)
    }

    /// Adds the comparison of `a` against `b` to `stats`. Returns (discrete_equal, bit_equal) outside the excluded fields.
    pub fn compare(&self, a: &ArenaState, b: &ArenaState, stats: &mut Stats) -> (bool, bool) {
        let (wa, wb) = (a.words(), b.words());
        let (mut discrete_ok, mut bit_ok) = (true, true);
        let state_no = stats.states;
        let mut local: BTreeMap<u32, FieldStat> = BTreeMap::new();
        for w in 0..wa.len() {
            let g = self.group[w];
            let (x, y) = (wa[w], wb[w]);
            let e = local.entry(g).or_default();
            e.words += 1;
            if x == y {
                continue;
            }
            let excluded = self.excluded[g as usize];
            match self.kind[w] {
                Kind::Int => {
                    e.discrete_diff += 1;
                    if !excluded {
                        discrete_ok = false;
                        bit_ok = false;
                    }
                }
                Kind::Float => {
                    e.float_diff += 1;
                    let d = ulp(x, y);
                    let bucket = match d {
                        None => 6,
                        Some(0) => 0,
                        Some(1) => 1,
                        Some(2..=4) => 2,
                        Some(5..=16) => 3,
                        Some(17..=1024) => 4,
                        Some(_) => 5,
                    };
                    e.hist[bucket] += 1;
                    e.max_ulp = e.max_ulp.max(d.unwrap_or(u64::MAX));
                    let sd = self.sulp_at(w, wa, wb);
                    e.max_sulp = e.max_sulp.max(sd);
                    e.shist[match sd {
                        v if v <= 1.0 => 0,
                        v if v <= 4.0 => 1,
                        v if v <= 16.0 => 2,
                        v if v <= 1024.0 => 3,
                        _ => 4,
                    }] += 1;
                    if !excluded {
                        bit_ok = false;
                    }
                }
            }
            if e.first.is_none() {
                e.first = Some((state_no, w, x, y));
            }
        }
        stats.states += 1;
        stats.discrete_equal += u64::from(discrete_ok);
        stats.bit_equal += u64::from(bit_ok);
        for (g, l) in local {
            stats.fields.entry(self.names[g as usize].clone()).or_default().merge(&l);
        }
        (discrete_ok, bit_ok)
    }
}

impl Stats {
    pub fn merge(&mut self, o: &Stats) {
        self.states += o.states;
        self.discrete_equal += o.discrete_equal;
        self.bit_equal += o.bit_equal;
        for (k, v) in &o.fields {
            self.fields.entry(k.clone()).or_default().merge(v);
        }
    }

    /// The worst verdict field: (name, max ulp, or u64::MAX for a discrete difference).
    pub fn worst(&self) -> Option<(String, u64)> {
        self.fields
            .iter()
            .filter(|(k, f)| !is_excluded(k) && f.discrete_diff + f.float_diff > 0)
            .map(|(k, f)| (k.clone(), if f.discrete_diff > 0 { u64::MAX } else { f.max_ulp }))
            .max_by_key(|&(_, u)| u)
    }

    /// The worst scale-aware ulp over the verdict fields: (name, sulp).
    pub fn worst_scaled(&self) -> Option<(String, f64)> {
        self.fields
            .iter()
            .filter(|(k, f)| !is_excluded(k) && f.float_diff > 0)
            .map(|(k, f)| (k.clone(), f.max_sulp))
            .max_by(|a, b| a.1.total_cmp(&b.1))
    }

    /// Words that differ outside the excluded fields: (discrete, float).
    pub fn diff_words(&self) -> (u64, u64) {
        self.fields.iter().filter(|(k, _)| !is_excluded(k)).fold((0, 0), |(d, f), (_, s)| (d + s.discrete_diff, f + s.float_diff))
    }

    /// A per-field table of the differences (verdict fields first, then the excluded ones).
    pub fn field_table(&self, max_rows: usize) -> String {
        let mut rows: Vec<(&String, &FieldStat)> = self.fields.iter().filter(|(_, f)| f.discrete_diff + f.float_diff > 0).collect();
        rows.sort_by_key(|(k, f)| (is_excluded(k), std::cmp::Reverse(f.discrete_diff + f.float_diff)));
        let mut s = format!("{:<44} {:>10} {:>9} {:>9} {:>9} {:>9}  {}  | sulp {}\n", "field", "words", "discrete", "float", "max ulp", "max sulp", BUCKETS.join(" "), SBUCKETS.join(" "));
        for (k, f) in rows.into_iter().take(max_rows) {
            let mark = if is_excluded(k) { " (excluded)" } else { "" };
            let ulp = if f.float_diff > 0 { f.max_ulp.to_string() } else { "-".into() };
            let h: Vec<String> = f.hist.iter().map(u64::to_string).collect();
            let sh: Vec<String> = f.shist.iter().map(u64::to_string).collect();
            let su = if f.float_diff > 0 { format!("{:.2}", f.max_sulp) } else { "-".into() };
            s.push_str(&format!("{:<44} {:>10} {:>9} {:>9} {:>9} {:>9}  {}  | {}{mark}\n", k, f.words, f.discrete_diff, f.float_diff, ulp, su, h.join(" "), sh.join(" ")));
        }
        s
    }
}
