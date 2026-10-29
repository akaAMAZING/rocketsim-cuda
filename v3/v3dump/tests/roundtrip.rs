//! Completeness of the rocketsim-dump patch. Run in release with the meshes available:
//!   ROCKETSIM_MESHES=<dir with soccar/*.cmf> cargo test --release -p v3dump -- --nocapture

use std::{fs, path::PathBuf};

use rocketsim::{
    Arena, ArenaConfig, CarBodyConfig, CarControls, GameMode, Team,
    dump::{ArenaDump, StaticDump},
};
use v3dump::{
    meshes::init_meshes,
    scenario::{self, Kind, Report, specs},
};

fn out_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
}

fn fresh_arena() -> Arena {
    init_meshes();
    let mut a = Arena::new_with_config(ArenaConfig::new(GameMode::Soccar).with_rng_seed(7));
    a.add_car(Team::Blue, CarBodyConfig::OCTANE);
    a.add_car(Team::Orange, CarBodyConfig::DOMINUS);
    a
}

/// The reference must not fuse multiply-adds.
#[test]
fn fma_is_off() {
    assert!(
        !cfg!(target_feature = "fma"),
        "built with target feature fma: unset RUSTFLAGS target-cpu=native"
    );
}

/// The static dump reloads with equal counts and hash, is stable between builds, and has V3's shape.
#[test]
fn static_dump_reloads() {
    let a = fresh_arena();
    let d = a.export_static();
    let path = out_dir().join("static.v3s");
    fs::write(&path, d.as_bytes()).unwrap();
    let back = StaticDump::from_bytes(fs::read(&path).unwrap()).expect("static dump reloads");
    assert_eq!(back, d);
    assert_eq!(back.hash(), d.hash());
    assert_eq!(back.counts(), d.counts());

    // a second arena (and a second export) gives the same bytes
    let again = fresh_arena().export_static();
    assert_eq!(again.hash(), d.hash());
    assert_eq!(again.as_bytes(), d.as_bytes());

    // counts: [bodies, dynamic, meshes, triangles, wide nodes, leaves, tri infos, planes, cars, pads, grid cells, ray-cell entries]
    let c = d.counts();
    println!("static dump: {} bytes, hash {:016x}, counts {c:?}", d.as_bytes().len(), d.hash());
    assert_eq!(c[0], 3 + 16 + 4, "ball + 2 cars + 16 meshes + 4 planes");
    assert_eq!(c[1], 3);
    assert_eq!(c[2], 16);
    assert_eq!(c[7], 4);
    assert_eq!(c[8], 2);
    assert_eq!(c[9], 34);
    assert_eq!(c[3], c[6], "one triangle info per triangle");
    assert_eq!(c[3], c[5], "single-triangle leaves");
    assert!(c[3] > 5000 && c[4] > 0);

    // a corrupted payload is refused
    let mut bad = d.as_bytes().to_vec();
    let n = bad.len();
    bad[n / 2] ^= 1;
    assert!(StaticDump::from_bytes(bad).is_err());
    // the config hash separates different configurations
    let mut other = Arena::new_with_config(ArenaConfig::new(GameMode::Soccar).with_rng_seed(7));
    other.add_car(Team::Blue, CarBodyConfig::PLANK);
    other.add_car(Team::Orange, CarBodyConfig::DOMINUS);
    assert_ne!(other.config_hash(), a.config_hash());
}

/// Writing a dump and reading it back is byte-identical; broken dumps are refused.
#[test]
fn binary_roundtrip() {
    let mut a = fresh_arena();
    a.reset_to_random_kickoff(Some(3));
    for t in 0..400 {
        a.set_car_controls(0, CarControls::default().with_throttle(1.0).with_boost(true).with_steer(((t / 50) % 3) as f32 - 1.0));
        a.set_car_controls(1, CarControls::default().with_throttle(1.0).with_boost(true));
        a.step_tick();
    }
    let d = a.export_state();
    let path = out_dir().join("arena.v3d");
    fs::write(&path, d.as_bytes()).unwrap();
    let back = ArenaDump::from_bytes(fs::read(&path).unwrap()).unwrap();
    assert_eq!(back, d);
    let mut b = fresh_arena();
    b.import_state(&back).unwrap();
    assert_eq!(b.export_state().as_bytes(), d.as_bytes());
    println!("arena dump: {} bytes, counts {:?}", d.as_bytes().len(), d.counts());

    // refusals
    let bytes = d.as_bytes();
    assert!(ArenaDump::from_bytes(bytes[..10].to_vec()).is_err());
    let mut wrong_magic = bytes.to_vec();
    wrong_magic[0] ^= 1;
    assert!(ArenaDump::from_bytes(wrong_magic).is_err());
    let mut wrong_version = bytes.to_vec();
    wrong_version[8] ^= 1;
    assert!(ArenaDump::from_bytes(wrong_version).is_err());
    let truncated = ArenaDump(bytes[..bytes.len() - 5].to_vec());
    assert!(fresh_arena().import_state(&truncated).is_err());
    let mut trailing = bytes.to_vec();
    trailing.push(0);
    assert!(fresh_arena().import_state(&ArenaDump(trailing)).is_err());
    // another car body: the config hash differs
    let mut other = Arena::new_with_config(ArenaConfig::new(GameMode::Soccar).with_rng_seed(7));
    other.add_car(Team::Blue, CarBodyConfig::HYBRID);
    other.add_car(Team::Orange, CarBodyConfig::DOMINUS);
    assert!(other.import_state(&d).is_err());
}

/// The key test: export A at random ticks, import into a fresh B, clone A, step all three with the same controls
/// and require byte-identical exports at every compare point.
#[test]
fn completeness_roundtrip() {
    let all = specs();
    assert!(all.len() >= 60, "{} scenarios", all.len());
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let t0 = std::time::Instant::now();
    let reports = scenario::run_all(&all, threads);
    let wall = t0.elapsed().as_secs_f32();

    let mut total = reports[0].clone();
    total.name = "ALL".into();
    total.main_ticks = 0;
    total.total_ticks = 0;
    total.checkpoints = 0;
    total.comparisons = 0;
    total.cov = Default::default();
    for r in &reports {
        total.add_totals(r);
        total.merge_max(r);
    }
    let count = |f: fn(&Kind) -> bool| all.iter().filter(|s| f(&s.kind)).count();
    let (kick, play, collide, corner, pads) = (
        count(|k| matches!(k, Kind::Kickoff { .. })),
        count(|k| matches!(k, Kind::Play)),
        count(|k| matches!(k, Kind::Collide)),
        count(|k| matches!(k, Kind::Corner)),
        count(|k| matches!(k, Kind::Pads)),
    );
    let mut text = scenario::table(&reports);
    text.push_str(&format!(
        "
{} scenarios ({kick} kickoff, {play} play, {collide} collide, {corner} corner, {pads} pads), {} main ticks, {} ticks stepped in all, {} checkpoints, {} arena comparisons, {} divergences, {wall:.1}s on {threads} threads
",
        all.len(),
        total.main_ticks,
        total.total_ticks,
        total.checkpoints,
        total.comparisons,
        reports.iter().map(|r| r.divergences.len()).sum::<usize>(),
    ));
    text.push_str(&format!("coverage: {:?}\n", total.cov));
    text.push_str(&format!("solver sizes: {:?}\n", total.stats));
    text.push_str(&format!(
        "max header counts [bodies, manifolds, active, points, manifold table, handles, pairs, ray-cell entries, cars, pads, records, ball-world records]: {:?}\nlargest arena dump: {} bytes\n",
        total.max_counts, total.max_dump_len
    ));
    println!("{text}");
    fs::write(out_dir().join("v3dump-report.txt"), &text).unwrap();

    let failed: Vec<&Report> = reports.iter().filter(|r| !r.divergences.is_empty()).collect();
    assert!(failed.is_empty(), "{} scenarios diverged:\n{}", failed.len(), scenario::table(&reports));

    // the play must have exercised what the dump has to carry
    let c = &total.cov;
    assert!(c.car_ball_hits > 100, "car-ball contacts");
    assert!(c.bumps > 100, "bumps");
    assert!(c.demos >= 5, "demos");
    assert!(c.car_world_hits > 100, "car-world contacts");
    assert!(c.ball_world_hits > 100, "ball-world contacts");
    assert!(c.pad_pickups > 20, "pad pickups");
    assert!(c.wall_ticks > 1000, "wall driving");
    assert!(c.ceiling_ticks > 100, "ceiling driving");
    assert!(c.flips > 20, "flips");
    assert!(c.double_jumps > 5, "double jumps");
    assert!(c.ball_sleeps > 20 && c.ball_wakes > 20, "ball sleep and wake-up (788c382)");
    for (r, s) in reports.iter().zip(&all) {
        if matches!(s.kind, Kind::Play) {
            assert!(r.main_ticks >= 20_000, "{}: {} ticks", r.name, r.main_ticks);
        } else {
            assert!(r.main_ticks >= 4_000, "{}: {} ticks", r.name, r.main_ticks);
        }
        // every window length of the spec is exercised (1, 8, 120, 1000 ticks)
        for w in [1, 8, 120, 1000] {
            assert!(s.checkpoints.iter().any(|&(_, win)| win == w), "{}: no window of {w}", r.name);
        }
    }
}

/// What an export and an import cost (printed, not asserted).
#[test]
fn export_import_cost() {
    let mut a = fresh_arena();
    a.reset_to_random_kickoff(Some(5));
    for t in 0..300 {
        a.set_car_controls(0, CarControls::default().with_throttle(1.0).with_boost(true).with_steer((t % 40) as f32 / 20.0 - 1.0));
        a.set_car_controls(1, CarControls::default().with_throttle(1.0).with_boost(true));
        a.step_tick();
    }
    let n = 2000;
    let t0 = std::time::Instant::now();
    let mut d = a.export_state();
    for _ in 1..n {
        d = a.export_state();
    }
    let export_us = t0.elapsed().as_secs_f64() * 1e6 / f64::from(n);
    let mut b = fresh_arena();
    let t0 = std::time::Instant::now();
    for _ in 0..n {
        b.import_state(&d).unwrap();
    }
    let import_us = t0.elapsed().as_secs_f64() * 1e6 / f64::from(n);
    let t0 = std::time::Instant::now();
    for _ in 0..n {
        let _ = a.clone();
    }
    let clone_us = t0.elapsed().as_secs_f64() * 1e6 / f64::from(n);
    let t0 = std::time::Instant::now();
    for _ in 0..n {
        a.step_tick();
    }
    let step_us = t0.elapsed().as_secs_f64() * 1e6 / f64::from(n);
    let t0 = std::time::Instant::now();
    let s = a.export_static();
    let static_ms = t0.elapsed().as_secs_f64() * 1e3;
    println!(
        "per call: export_state {export_us:.1} us, import_state {import_us:.1} us, Arena::clone {clone_us:.1} us, step_tick {step_us:.1} us; export_static {static_ms:.1} ms ({} bytes)",
        s.as_bytes().len()
    );
}
