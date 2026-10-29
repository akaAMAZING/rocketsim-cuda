//! Scenarios for the completeness round trip: build an arena, play it with the random drivers, and at chosen ticks
//! export arena A, import the dump into a fresh arena B, clone A, and step all three with identical controls.

use std::{
    fmt::Write as _,
    sync::atomic::{AtomicUsize, Ordering},
};

use rocketsim::{
    Arena, ArenaConfig, ArenaEvent, BallState, BoostPadState, CarBodyConfig, CarControls, CarCore, GameMode, Mat3A,
    PhysState, Team, Vec3A, consts,
    dump::{ARENA_COUNTS, ArenaDump, SolverStats},
};

use crate::{
    bots::{Bot, MODES, Mode},
    meshes::init_meshes,
    rng::Rng,
};

pub const BODIES: [CarBodyConfig; 7] = [
    CarBodyConfig::OCTANE,
    CarBodyConfig::DOMINUS,
    CarBodyConfig::PLANK,
    CarBodyConfig::BREAKOUT,
    CarBodyConfig::HYBRID,
    CarBodyConfig::MERC,
    CarBodyConfig::PSYCLOPS,
];

/// Ticks after a checkpoint at which the three arenas are compared.
/// Event-triggered checkpoints per scenario, at most.
pub const MAX_TRIGGERS: u32 = 120;

pub const COMPARE_AT: [u32; 11] = [1, 2, 4, 8, 16, 32, 64, 120, 250, 500, 1000];

#[derive(Clone, Debug)]
pub enum Kind {
    /// Kickoff spawn `spawn` (index into V3's five Soccar kickoff spots).
    Kickoff { spawn: usize, reset_seed: u64 },
    /// Random placement of ball and cars, then long play.
    Play,
    /// Two cars start facing each other on the ground at speed, both chase the other car: bumps and demos.
    Collide,
    /// The ball in a corner, on a post or in the goal mouth with both cars on it: pinches and piles (sizing stress).
    Corner,
    /// Both cars on one small pad with low boost, car 1 first in the pad's arrival list and held by its hitbox box only
    /// (outside the pickup cylinder): V3 0.3.3's pad membership decides who gets the boost.
    Pads,
}

#[derive(Clone, Debug)]
pub struct Spec {
    pub id: usize,
    pub kind: Kind,
    /// The first car added is blue (car 0 is blue) or orange.
    pub blue_first: bool,
    /// Body preset of car 0 and car 1.
    pub bodies: [usize; 2],
    pub modes: [Mode; 2],
    pub seed: u64,
    /// Main ticks the scenario plays at least.
    pub ticks: u32,
    /// (main tick, window): export at the tick, then step A, B and the clone `window` ticks.
    pub checkpoints: Vec<(u32, u32)>,
}

/// The scenario kinds' labels, in the order of [`Kind`]'s variants.
pub const KIND_LABELS: [&str; 5] = ["kickoff", "play", "collide", "corner", "pads"];

impl Kind {
    /// The label of this kind (an entry of [`KIND_LABELS`]).
    pub fn label(&self) -> &'static str {
        match self {
            Kind::Kickoff { .. } => KIND_LABELS[0],
            Kind::Play => KIND_LABELS[1],
            Kind::Collide => KIND_LABELS[2],
            Kind::Corner => KIND_LABELS[3],
            Kind::Pads => KIND_LABELS[4],
        }
    }
}

impl Spec {
    pub fn name(&self) -> String {
        let kind = match &self.kind {
            Kind::Kickoff { spawn, .. } => format!("kickoff{spawn}"),
            Kind::Play => "play".into(),
            Kind::Collide => "collide".into(),
            Kind::Corner => "corner".into(),
            Kind::Pads => "pads".into(),
        };
        format!(
            "{:02}-{kind}-{}-{:?}/{:?}-b{}{}",
            self.id,
            if self.blue_first { "BO" } else { "OB" },
            self.modes[0],
            self.modes[1],
            self.bodies[0],
            self.bodies[1]
        )
    }
}

/// A seed for `reset_to_random_kickoff` that puts the pair on kickoff spot `spawn`.
fn seed_for_spawn(spawn: usize) -> u64 {
    let spot = consts::car::spawn::get_kickoff_spawn_locations(GameMode::Soccar)[spawn];
    (0..10_000u64)
        .find(|&seed| {
            let mut a = build(&ArenaSpec {
                blue_first: true,
                bodies: [0, 0],
                rng_seed: 1,
            });
            a.reset_to_random_kickoff(Some(seed));
            let p = a.get_car_state(0).phys.pos;
            (p.x - spot.x).abs() < 1.0 && (p.y - spot.y).abs() < 1.0
        })
        .expect("no seed found for kickoff spot")
}

/// `V3DUMP_SCALE=n` multiplies the play length of the long scenarios (default 1: 20k ticks).
pub fn specs() -> Vec<Spec> {
    init_meshes();
    let scale: u32 = std::env::var("V3DUMP_SCALE").ok().and_then(|v| v.parse().ok()).unwrap_or(1).max(1);
    let mut out = Vec::new();
    // 5 spawns x 2 car orders x 3 seeds = 30 kickoff scenarios
    for spawn in 0..5 {
        let kick_seed = seed_for_spawn(spawn);
        for blue_first in [true, false] {
            for k in 0..3u64 {
                let id = out.len();
                let seed = 0x1000 + (id as u64) * 7919 + k;
                let mut rng = Rng(seed);
                let modes = match k {
                    0 => [Mode::ChaseBall, Mode::ChaseBall],
                    1 => [Mode::ChaseBall, Mode::ChaseCar],
                    _ => [Mode::ChaseCar, Mode::Random],
                };
                let mut checkpoints = vec![(0, 1), (1, 8), (9, 120), (129, 1000)];
                let mut t = 1129 + rng.int(50, 600);
                for _ in 0..2 {
                    checkpoints.push((t, 1000));
                    t += 1000 + rng.int(100, 1200);
                }
                out.push(Spec {
                    id,
                    kind: Kind::Kickoff { spawn, reset_seed: kick_seed },
                    blue_first,
                    bodies: [rng.int(0, 6) as usize, rng.int(0, 6) as usize],
                    modes,
                    seed,
                    ticks: t + 500,
                    checkpoints,
                });
            }
        }
    }
    // long random-control play: 40 scenarios of at least 20k ticks
    for j in 0..40usize {
        let id = out.len();
        let seed = 0x9000 + (id as u64) * 104_729;
        let mut rng = Rng(seed);
        let modes = [MODES[j % 5], MODES[(j / 5 + j % 5 + 1 + (j / 10)) % 5]];
        let mut checkpoints = Vec::new();
        let mut t = rng.int(10, 400);
        for window in [1, 8, 120, 1000, 1000, 1000, 1000] {
            checkpoints.push((t, window));
            t += window + rng.int(50, 1500);
        }
        out.push(Spec {
            id,
            kind: Kind::Play,
            blue_first: rng.chance(0.5),
            bodies: [rng.int(0, 6) as usize, rng.int(0, 6) as usize],
            modes,
            seed,
            ticks: (20_000 * scale).max(t + 500),
            checkpoints,
        });
    }
    // head-on drives: bumps and demos
    for j in 0..10usize {
        let id = out.len();
        let seed = 0x5000 + (id as u64) * 15_485_863;
        let mut rng = Rng(seed);
        let mut checkpoints = vec![(0, 1), (1, 8), (9, 120), (129, 1000)];
        let mut t = 1129 + rng.int(50, 600);
        for _ in 0..2 {
            checkpoints.push((t, 1000));
            t += 1000 + rng.int(100, 1200);
        }
        out.push(Spec {
            id,
            kind: Kind::Collide,
            blue_first: j % 2 == 0,
            bodies: [rng.int(0, 6) as usize, rng.int(0, 6) as usize],
            modes: [Mode::ChaseCar, if j % 3 == 0 { Mode::ChaseBall } else { Mode::ChaseCar }],
            seed,
            ticks: t + 500,
            checkpoints,
        });
    }
    // pinches in corners and goal mouths
    for j in 0..10usize {
        let id = out.len();
        let seed = 0x7000 + (id as u64) * 32_452_843;
        let mut rng = Rng(seed);
        let mut checkpoints = vec![(0, 1), (1, 8), (9, 120), (129, 1000)];
        let mut t = 1129 + rng.int(50, 600);
        for _ in 0..2 {
            checkpoints.push((t, 1000));
            t += 1000 + rng.int(100, 1200);
        }
        out.push(Spec {
            id,
            kind: Kind::Corner,
            blue_first: j % 2 == 1,
            bodies: [rng.int(0, 6) as usize, rng.int(0, 6) as usize],
            modes: [Mode::ChaseBall, if j % 2 == 0 { Mode::ChaseBall } else { Mode::ChaseCar }],
            seed,
            ticks: t + 500,
            checkpoints,
        });
    }
    // pad membership: arrival order and box-only retention
    for j in 0..12usize {
        let id = out.len();
        let seed = 0x3000 + (id as u64) * 49_979_687;
        let mut rng = Rng(seed);
        let mut checkpoints = vec![(0, 1), (1, 8), (9, 120), (129, 1000)];
        let mut t = 1129 + rng.int(50, 600);
        for _ in 0..2 {
            checkpoints.push((t, 1000));
            t += 1000 + rng.int(100, 1200);
        }
        out.push(Spec {
            id,
            kind: Kind::Pads,
            blue_first: j % 2 == 0,
            bodies: [rng.int(0, 6) as usize, rng.int(0, 6) as usize],
            modes: [MODES[j % 5], MODES[(j + 2) % 5]],
            seed,
            ticks: t + 500,
            checkpoints,
        });
    }
    out
}

struct ArenaSpec {
    blue_first: bool,
    bodies: [usize; 2],
    rng_seed: u64,
}

/// A fresh 1v1 Soccar arena.
fn build(a: &ArenaSpec) -> Arena {
    let mut arena =
        Arena::new_with_config(ArenaConfig::new(GameMode::Soccar).with_rng_seed(a.rng_seed));
    let teams = if a.blue_first {
        [Team::Blue, Team::Orange]
    } else {
        [Team::Orange, Team::Blue]
    };
    for i in 0..2 {
        arena.add_car(teams[i], BODIES[a.bodies[i]]);
    }
    arena
}

pub fn build_arena(spec: &Spec) -> Arena {
    init_meshes();
    build(&ArenaSpec {
        blue_first: spec.blue_first,
        bodies: spec.bodies,
        rng_seed: spec.seed,
    })
}

fn rot(yaw: f32, pitch: f32, roll: f32) -> Mat3A {
    let (sy, cy) = yaw.sin_cos();
    let (sp, cp) = pitch.sin_cos();
    let (sr, cr) = roll.sin_cos();
    // R = Rz(yaw) * Ry(pitch) * Rx(roll); columns are forward, right, up
    Mat3A::from_cols(
        Vec3A::new(cy * cp, sy * cp, -sp),
        Vec3A::new(cy * sp * sr - sy * cr, sy * sp * sr + cy * cr, cp * sr),
        Vec3A::new(cy * sp * cr + sy * sr, sy * sp * cr - cy * sr, cp * cr),
    )
}

fn random_dir(rng: &mut Rng) -> Vec3A {
    loop {
        let v = Vec3A::new(rng.range(-1.0, 1.0), rng.range(-1.0, 1.0), rng.range(-1.0, 1.0));
        if v.length_squared() > 0.05 && v.length_squared() <= 1.0 {
            return v.normalize();
        }
    }
}

/// Random ball and car placement for the play scenarios.
fn randomize(arena: &mut Arena, rng: &mut Rng) {
    arena.reset_to_random_kickoff(Some(rng.next_u64()));
    let mut ball = BallState::DEFAULT;
    ball.phys.pos = Vec3A::new(rng.range(-3500.0, 3500.0), rng.range(-4300.0, 4300.0), rng.range(95.0, 1200.0));
    ball.phys.vel = random_dir(rng) * rng.range(0.0, 2500.0);
    ball.phys.ang_vel = random_dir(rng) * rng.range(0.0, 4.0);
    let ball_pos = ball.phys.pos;
    arena.set_ball_state(ball);
    for i in 0..2 {
        let yaw = rng.range(-3.14, 3.14);
        let (pos, rotm, vel, grounded) = match rng.int(0, 4) {
            0 => {
                let r = rot(yaw, 0.0, 0.0);
                let p = Vec3A::new(rng.range(-3500.0, 3500.0), rng.range(-4200.0, 4200.0), 17.0);
                (p, r, r.x_axis * rng.range(0.0, 2000.0), true)
            }
            1 => {
                let r = rot(yaw, rng.range(-0.6, 0.6), rng.range(-1.5, 1.5));
                let p = Vec3A::new(rng.range(-3500.0, 3500.0), rng.range(-4200.0, 4200.0), rng.range(300.0, 1500.0));
                (p, r, random_dir(rng) * rng.range(0.0, 1500.0), false)
            }
            2 => {
                let side = if rng.chance(0.5) { 1.0 } else { -1.0 };
                let yaw = if side > 0.0 { 0.0 } else { 3.14159 } + rng.range(-0.4, 0.4);
                let r = rot(yaw, rng.range(-0.3, 0.3), 0.0);
                let p = Vec3A::new(side * rng.range(3700.0, 4040.0), rng.range(-3000.0, 3000.0), rng.range(100.0, 800.0));
                (p, r, Vec3A::new(side * rng.range(500.0, 1800.0), 0.0, rng.range(0.0, 600.0)), false)
            }
            3 => {
                let r = rot(yaw, rng.range(-0.6, 0.6), rng.range(-1.5, 1.5));
                let p = Vec3A::new(rng.range(-3000.0, 3000.0), rng.range(-3500.0, 3500.0), rng.range(1700.0, 1950.0));
                (p, r, Vec3A::new(rng.range(-800.0, 800.0), rng.range(-800.0, 800.0), rng.range(500.0, 1500.0)), false)
            }
            _ => {
                let r = rot(yaw, 0.0, 0.0);
                let off = Vec3A::new(rng.range(-900.0, 900.0), rng.range(-900.0, 900.0), 0.0);
                let mut p = ball_pos + off;
                p.z = 17.0;
                p.x = p.x.clamp(-3800.0, 3800.0);
                p.y = p.y.clamp(-4600.0, 4600.0);
                (p, r, r.x_axis * rng.range(0.0, 1500.0), true)
            }
        };
        let phys = PhysState {
            pos,
            rot_mat: rotm,
            vel,
            ang_vel: if grounded { Vec3A::ZERO } else { random_dir(rng) * rng.range(0.0, 3.0) },
        };
        let boost = if rng.chance(0.3) { 0.0 } else { rng.range(0.0, 100.0) };
        let core = if grounded { CarCore::grounded(phys, boost) } else { CarCore::aerial(phys) };
        arena.set_car_state(i, core.with_boost(boost));
    }
    if rng.chance(0.5) {
        arena.clear_persistent_manifolds();
    }
}

/// Two cars on the ground, facing each other at speed with full boost.
fn collide(arena: &mut Arena, rng: &mut Rng) {
    arena.reset_to_random_kickoff(Some(rng.next_u64()));
    for i in 0..2 {
        let sign = if i == 0 { -1.0 } else { 1.0 };
        let yaw = if i == 0 { std::f32::consts::FRAC_PI_2 } else { -std::f32::consts::FRAC_PI_2 };
        let r = rot(yaw + rng.range(-0.1, 0.1), 0.0, 0.0);
        let phys = PhysState {
            pos: Vec3A::new(rng.range(-700.0, 700.0), sign * rng.range(1500.0, 2600.0), 17.0),
            rot_mat: r,
            vel: r.x_axis * rng.range(900.0, 1800.0),
            ang_vel: Vec3A::ZERO,
        };
        arena.set_car_state(i, CarCore::grounded(phys, 100.0));
    }
}

/// The ball in a back corner, against a post or on the goal line, both cars driving at it from close range.
fn corner(arena: &mut Arena, rng: &mut Rng) {
    arena.reset_to_random_kickoff(Some(rng.next_u64()));
    let (sx, sy) = (if rng.chance(0.5) { 1.0 } else { -1.0 }, if rng.chance(0.5) { 1.0 } else { -1.0 });
    let ball_pos = match rng.int(0, 2) {
        0 => Vec3A::new(sx * rng.range(3600.0, 3950.0), sy * rng.range(4500.0, 4950.0), rng.range(93.0, 300.0)),
        1 => Vec3A::new(sx * rng.range(700.0, 1100.0), sy * rng.range(4900.0, 5100.0), rng.range(93.0, 700.0)),
        _ => Vec3A::new(sx * rng.range(0.0, 600.0), sy * rng.range(5100.0, 5500.0), rng.range(93.0, 500.0)),
    };
    let mut ball = BallState::DEFAULT;
    ball.phys.pos = ball_pos;
    ball.phys.vel = random_dir(rng) * rng.range(0.0, 600.0);
    arena.set_ball_state(ball);
    for i in 0..2 {
        let to_ball = Vec3A::new(rng.range(-1.0, 1.0), rng.range(-1.0, 1.0), 0.0).normalize_or_zero();
        let mut pos = ball_pos + to_ball * rng.range(250.0, 700.0);
        pos.z = 17.0;
        pos.x = pos.x.clamp(-3900.0, 3900.0);
        pos.y = pos.y.clamp(-5000.0, 5000.0);
        let yaw = (-to_ball.y).atan2(-to_ball.x);
        let r = rot(yaw, 0.0, 0.0);
        let phys = PhysState {
            pos,
            rot_mat: r,
            vel: r.x_axis * rng.range(300.0, 1400.0),
            ang_vel: Vec3A::ZERO,
        };
        arena.set_car_state(i, CarCore::grounded(phys, rng.range(30.0, 100.0)));
    }
}

fn pads(arena: &mut Arena, rng: &mut Rng) {
    arena.reset_to_random_kickoff(Some(rng.next_u64()));
    let small: Vec<usize> = (0..arena.num_boost_pads()).filter(|&i| !arena.get_boost_pad_config(i).is_big).collect();
    let pad = small[rng.int(0, small.len() as u32 - 1) as usize];
    let centre = arena.get_boost_pad_config(pad).pos;
    // car 1 just outside the 144 uu pickup cylinder, its hitbox over the 96 uu pad box; car 0 inside the cylinder
    let a = rng.range(-3.14159, 3.14159);
    let places = [
        (centre + Vec3A::new(rng.range(-60.0, 60.0), rng.range(-60.0, 60.0), 0.0), rng.range(-3.14159, 3.14159)),
        (centre + Vec3A::new(a.cos(), a.sin(), 0.0) * rng.range(146.0, 160.0), a + 3.14159),
    ];
    for (i, (mut pos, yaw)) in places.into_iter().enumerate() {
        pos.z = 17.0;
        let phys = PhysState { pos, rot_mat: rot(yaw, 0.0, 0.0), vel: Vec3A::ZERO, ang_vel: Vec3A::ZERO };
        arena.set_car_state(i, CarCore::grounded(phys, rng.range(0.0, 60.0)));
    }
    arena.set_boost_pad_state(pad, BoostPadState { touching_car_idxs: vec![1, 0], cooldown: 0.0 });
}

/// What the scenario made V3 do (counted on arena A only).
#[derive(Clone, Debug, Default)]
pub struct Coverage {
    pub car_ball_hits: u64,
    pub bumps: u64,
    pub demos: u64,
    pub car_world_hits: u64,
    pub ball_world_hits: u64,
    pub pad_pickups: u64,
    pub landings: u64,
    pub flips: u64,
    pub double_jumps: u64,
    pub auto_flips: u64,
    pub wall_ticks: u64,
    pub ceiling_ticks: u64,
    pub supersonic_ticks: u64,
    pub boost_ticks: u64,
    pub demoed_ticks: u64,
    pub air_ball_ticks: u64,
    pub resets: u64,
    /// V3's ball sleep (788c382): ticks that end with the ball asleep, falls asleep, wake-ups.
    pub ball_sleep_ticks: u64,
    pub ball_sleeps: u64,
    pub ball_wakes: u64,
}

impl Coverage {
    fn add(&mut self, o: &Self) {
        macro_rules! a { ($($f:ident),*) => { $(self.$f += o.$f;)* } }
        a!(
            car_ball_hits, bumps, demos, car_world_hits, ball_world_hits, pad_pickups, landings, flips, double_jumps,
            auto_flips, wall_ticks, ceiling_ticks, supersonic_ticks, boost_ticks, demoed_ticks, air_ball_ticks, resets,
            ball_sleep_ticks, ball_sleeps, ball_wakes
        );
    }
}

/// The previous tick's flags `observe` compares against: per car (flipping, double-jumped, auto-flipping, supersonic),
/// and the ball's activation code.
#[derive(Clone, Copy, Debug, Default)]
pub struct Prev {
    pub cars: [(bool, bool, bool, bool); 2],
    pub ball_activation: u8,
}

/// The ball's activation code for "asleep".
const SLEEPING: u8 = 1;

#[derive(Clone, Debug)]
pub struct Report {
    pub name: String,
    pub main_ticks: u32,
    /// All ticks stepped over A and its three copies (fresh import, used import, clone).
    pub total_ticks: u64,
    pub checkpoints: u32,
    /// Arena pairs compared: A against each of its three copies counts three at every compare point.
    pub comparisons: u64,
    pub divergences: Vec<String>,
    pub cov: Coverage,
    pub stats: SolverStats,
    /// Largest header counts seen in any export.
    pub max_counts: [u32; ARENA_COUNTS],
    pub max_dump_len: usize,
    pub secs: f32,
}

impl Report {
    pub fn merge_max(&mut self, other: &Self) {
        for i in 0..ARENA_COUNTS {
            self.max_counts[i] = self.max_counts[i].max(other.max_counts[i]);
        }
        let (a, b) = (&mut self.stats, &other.stats);
        a.max_live_manifolds = a.max_live_manifolds.max(b.max_live_manifolds);
        a.max_active_manifolds = a.max_active_manifolds.max(b.max_active_manifolds);
        a.max_active_points = a.max_active_points.max(b.max_active_points);
        a.max_contact_rows = a.max_contact_rows.max(b.max_contact_rows);
        a.max_friction_rows = a.max_friction_rows.max(b.max_friction_rows);
        a.max_solver_bodies = a.max_solver_bodies.max(b.max_solver_bodies);
        self.max_dump_len = self.max_dump_len.max(other.max_dump_len);
    }

    pub fn add_totals(&mut self, other: &Self) {
        self.main_ticks += other.main_ticks;
        self.total_ticks += other.total_ticks;
        self.checkpoints += other.checkpoints;
        self.comparisons += other.comparisons;
        self.cov.add(&other.cov);
    }

    pub fn line(&self) -> String {
        let c = &self.cov;
        format!(
            "{:<34} main {:>6} total {:>7} cp {:>2} cmp {:>4} div {} | hit {:>4} bump {:>3} demo {:>2} wall {:>5} ceil {:>4} flip {:>3} dj {:>3} pad {:>3} reset {:>2} sleep {:>2}/{:>2} | man {:>2} rows {:>2}/{:>2} | {:>4.1}s",
            self.name,
            self.main_ticks,
            self.total_ticks,
            self.checkpoints,
            self.comparisons,
            self.divergences.len(),
            c.car_ball_hits,
            c.bumps,
            c.demos,
            c.wall_ticks,
            c.ceiling_ticks,
            c.flips,
            c.double_jumps,
            c.pad_pickups,
            c.resets,
            c.ball_sleeps,
            c.ball_wakes,
            self.stats.max_live_manifolds,
            self.stats.max_contact_rows,
            self.stats.max_friction_rows,
            self.secs
        )
    }
}

pub fn out_of_world(a: &Arena) -> bool {
    let p = a.get_ball_state().phys.pos;
    !(p.x.abs() < 6000.0 && p.y.abs() < 6600.0 && p.z > -100.0 && p.z < 2400.0)
}

/// Sets the controls, steps one tick, restarts the play after a goal or an escape. Identical for all three arenas.
fn step_arena(a: &mut Arena, ctrls: &[CarControls; 2], cov: &mut Coverage) -> bool {
    step_arena_with(a, ctrls, cov, false)
}

/// `step_arena`, advancing by `step_tick` or (`phased`, feature `trace`) by the nine phases in order.
pub fn step_arena_with(a: &mut Arena, ctrls: &[CarControls; 2], cov: &mut Coverage, phased: bool) -> bool {
    a.set_car_controls(0, ctrls[0]);
    a.set_car_controls(1, ctrls[1]);
    let mut interesting = false;
    if phased {
        for k in 0..rocketsim::PHASES {
            a.step_phase(k);
        }
    } else {
        a.step_tick();
    }
    for e in a.get_last_step_events() {
        match e {
            ArenaEvent::CarHitBall(_) => (cov.car_ball_hits, interesting) = (cov.car_ball_hits + 1, true),
            ArenaEvent::CarHitCar(h) if h.is_demo => (cov.demos, interesting) = (cov.demos + 1, true),
            ArenaEvent::CarHitCar(_) => (cov.bumps, interesting) = (cov.bumps + 1, true),
            ArenaEvent::CarHitWorld(_) => cov.car_world_hits += 1,
            ArenaEvent::BallHitWorld(_) => cov.ball_world_hits += 1,
            ArenaEvent::CarPickupBoost(_) => (cov.pad_pickups, interesting) = (cov.pad_pickups + 1, true),
            ArenaEvent::CarLanded(_) => (cov.landings, interesting) = (cov.landings + 1, true),
        }
    }
    if a.is_ball_scored() || out_of_world(a) {
        a.reset_to_random_kickoff(Some(a.tick_count()));
        cov.resets += 1;
        interesting = true;
    }
    interesting
}

/// State-derived coverage of arena A after a tick.
pub fn observe(a: &Arena, prev: &mut Prev, cov: &mut Coverage) -> bool {
    let mut interesting = false;
    let (ball_activation, _) = a.ball_activation();
    if (ball_activation == SLEEPING) != (prev.ball_activation == SLEEPING) {
        if ball_activation == SLEEPING {
            cov.ball_sleeps += 1;
        } else {
            cov.ball_wakes += 1;
        }
        interesting = true;
    }
    prev.ball_activation = ball_activation;
    cov.ball_sleep_ticks += u64::from(ball_activation == SLEEPING);
    for (i, p) in prev.cars.iter_mut().enumerate() {
        let st = a.get_car_state(i);
        if st.is_flipping && !p.0 {
            cov.flips += 1;
            interesting = true;
        }
        if st.has_double_jumped && !p.1 {
            cov.double_jumps += 1;
            interesting = true;
        }
        if st.is_auto_flipping && !p.2 {
            cov.auto_flips += 1;
            interesting = true;
        }
        if st.is_supersonic != p.3 {
            interesting = true;
        }
        *p = (st.is_flipping, st.has_double_jumped, st.is_auto_flipping, st.is_supersonic);
        let up = st.phys.rot_mat.z_axis;
        if st.is_on_ground() && up.z.abs() < 0.3 {
            cov.wall_ticks += 1;
        }
        if st.is_on_ground() && up.z < -0.7 {
            cov.ceiling_ticks += 1;
        }
        cov.supersonic_ticks += u64::from(st.is_supersonic);
        cov.boost_ticks += u64::from(st.is_boosting);
        cov.demoed_ticks += u64::from(st.is_demoed);
    }
    cov.air_ball_ticks += u64::from(a.get_ball_state().phys.pos.z > 300.0);
    interesting
}

/// First difference between two dumps, with the section it falls in.
pub fn diff(a: &ArenaDump, b: &ArenaDump) -> Option<String> {
    let (x, y) = (a.as_bytes(), b.as_bytes());
    if x == y {
        return None;
    }
    let off = x.iter().zip(y).position(|(p, q)| p != q).unwrap_or(x.len().min(y.len()));
    let mut section = "header";
    let mut at = ArenaDump::HEADER_LEN;
    for t in [b"ARNA", b"BALL", b"CARS", b"PADS", b"CONT", b"BODY", b"DISP", b"BROA", b"ISLD"] {
        if let Some(p) = x[at..].windows(4).position(|w| w == t) {
            if at + p <= off {
                section = std::str::from_utf8(t).unwrap();
                at += p;
            }
        }
    }
    Some(format!(
        "first difference at byte {off} (section {section}, +{}); lengths {} vs {}",
        off - at,
        x.len(),
        y.len()
    ))
}

fn note_export(rep: &mut Report, d: &ArenaDump) {
    let c = d.counts();
    for i in 0..ARENA_COUNTS {
        rep.max_counts[i] = rep.max_counts[i].max(c[i]);
    }
    rep.max_dump_len = rep.max_dump_len.max(d.as_bytes().len());
}

fn absorb_stats(rep: &mut Report, a: &Arena) {
    let (s, t) = (&mut rep.stats, a.dump_stats());
    s.max_live_manifolds = s.max_live_manifolds.max(t.max_live_manifolds);
    s.max_active_manifolds = s.max_active_manifolds.max(t.max_active_manifolds);
    s.max_active_points = s.max_active_points.max(t.max_active_points);
    s.max_contact_rows = s.max_contact_rows.max(t.max_contact_rows);
    s.max_friction_rows = s.max_friction_rows.max(t.max_friction_rows);
    s.max_solver_bodies = s.max_solver_bodies.max(t.max_solver_bodies);
}

/// A used arena of the same configuration: other play, other pose, hidden state left over from 200-400 ticks.
fn dirty_arena(spec: &Spec, seed: u64) -> Arena {
    let mut d = build_arena(spec);
    let mut rng = Rng(seed);
    randomize(&mut d, &mut rng);
    let mut bots = [
        Bot::new(MODES[rng.int(0, 4) as usize], rng.next_u64()),
        Bot::new(MODES[rng.int(0, 4) as usize], rng.next_u64()),
    ];
    let mut cov = Coverage::default();
    for _ in 0..rng.int(200, 400) {
        let ctrls = [bots[0].controls(&d, 0), bots[1].controls(&d, 1)];
        step_arena(&mut d, &ctrls, &mut cov);
    }
    d
}

/// One checkpoint: export A, import into a fresh B and into a used D, clone A, step all four `window` ticks.
fn checkpoint(
    spec: &Spec,
    a: &mut Arena,
    bots: &mut [Bot; 2],
    window: u32,
    prev: &mut Prev,
    rep: &mut Report,
) {
    let tick = a.tick_count();
    let d = a.export_state();
    note_export(rep, &d);

    // the binary round trip: bytes -> ArenaDump -> bytes
    match ArenaDump::from_bytes(d.as_bytes().to_vec()) {
        Ok(back) if back == d => {}
        _ => rep.divergences.push(format!("tick {tick}: dump bytes do not read back")),
    }

    let mut arms: Vec<(&str, Arena)> = vec![
        ("fresh", build_arena(spec)),
        ("used", dirty_arena(spec, spec.seed ^ tick)),
    ];
    for (name, arena) in &mut arms {
        if let Err(e) = arena.import_state(&d) {
            rep.divergences.push(format!("tick {tick}: import into {name} failed: {e}"));
            return;
        }
    }
    arms.push(("clone", a.clone()));
    rep.checkpoints += 1;
    for (name, arena) in &arms {
        rep.comparisons += 1;
        if let Some(m) = diff(&d, &arena.export_state()) {
            rep.divergences.push(format!("tick {tick}+0 {name}: {m}"));
            return;
        }
    }

    let mut cov = Coverage::default();
    for n in 1..=window {
        let ctrls = [bots[0].controls(a, 0), bots[1].controls(a, 1)];
        step_arena(a, &ctrls, &mut rep.cov);
        for (_, arena) in &mut arms {
            step_arena(arena, &ctrls, &mut cov);
        }
        observe(a, prev, &mut rep.cov);
        rep.total_ticks += 1 + arms.len() as u64;
        if COMPARE_AT.contains(&n) {
            let ea = a.export_state();
            note_export(rep, &ea);
            for (name, arena) in &arms {
                rep.comparisons += 1;
                if let Some(m) = diff(&ea, &arena.export_state()) {
                    rep.divergences.push(format!("tick {tick}+{n} {name}: {m}"));
                    for (_, arena) in &arms {
                        absorb_stats(rep, arena);
                    }
                    return;
                }
            }
        }
    }
    for (_, arena) in &arms {
        absorb_stats(rep, arena);
    }
}

impl Report {
    pub fn new(name: String) -> Self {
        Self {
            name,
            main_ticks: 0,
            total_ticks: 0,
            checkpoints: 0,
            comparisons: 0,
            divergences: Vec::new(),
            cov: Coverage::default(),
            stats: SolverStats::default(),
            max_counts: [0; ARENA_COUNTS],
            max_dump_len: 0,
            secs: 0.0,
        }
    }
}

/// The arena a scenario starts from: built, placed (kickoff or randomised) and checked.
pub fn start_arena(spec: &Spec) -> Arena {
    let mut a = build_arena(spec);
    match spec.kind.clone() {
        Kind::Kickoff { reset_seed, .. } => a.reset_to_random_kickoff(Some(reset_seed)),
        Kind::Play => randomize(&mut a, &mut Rng(spec.seed ^ 0xabcd)),
        Kind::Collide => collide(&mut a, &mut Rng(spec.seed ^ 0xabcd)),
        Kind::Corner => corner(&mut a, &mut Rng(spec.seed ^ 0xabcd)),
        Kind::Pads => pads(&mut a, &mut Rng(spec.seed ^ 0xabcd)),
    }
    if let Kind::Kickoff { spawn, .. } = spec.kind {
        // the spawn is fixed by the seed (checked once at build of the specs); verify it here
        let spot = consts::car::spawn::get_kickoff_spawn_locations(GameMode::Soccar)[spawn];
        let blue = (0..2).find(|&i| a.get_car_info(i).team == Team::Blue).unwrap();
        let p = a.get_car_state(blue).phys.pos;
        assert!((p.x - spot.x).abs() < 1.0 && (p.y - spot.y).abs() < 1.0, "kickoff spot mismatch");
    }
    a
}

pub fn run(spec: &Spec) -> Report {
    let t0 = std::time::Instant::now();
    let mut rep = Report::new(spec.name());
    let mut a = start_arena(spec);
    let mut bots = [
        Bot::new(spec.modes[0], spec.seed ^ 0x11),
        Bot::new(spec.modes[1], spec.seed ^ 0x22),
    ];
    let mut prev = Prev::default();
    let mut cps = spec.checkpoints.iter().copied().peekable();
    let mut rng = Rng(spec.seed ^ 0x5eed);
    // event-triggered checkpoints: a few ticks after a bump, demo, ball touch, landing, pad pickup, flip or reset,
    // so that the short-lived state (cooldowns, wheel caches, fresh manifolds) is exported while it is live
    let (mut trigger_at, mut triggers, mut cooldown_until) = (None, 0u32, 0u32);
    let mut t = 0u32;
    while t < spec.ticks {
        while cps.peek().is_some_and(|&(at, _)| at < t) {
            cps.next();
        }
        let scheduled = cps.peek().copied().filter(|&(at, _)| at == t);
        let triggered = trigger_at.filter(|&at| at <= t).map(|_| (t, [8, 120][rng.int(0, 1) as usize]));
        if let Some((_, window)) = scheduled.or(triggered) {
            if scheduled.is_some() {
                cps.next();
            } else {
                trigger_at = None;
                triggers += 1;
            }
            checkpoint(spec, &mut a, &mut bots, window, &mut prev, &mut rep);
            t += window;
            cooldown_until = t + 40;
            continue;
        }
        let ctrls = [bots[0].controls(&a, 0), bots[1].controls(&a, 1)];
        let hit = step_arena(&mut a, &ctrls, &mut rep.cov);
        let moved = observe(&a, &mut prev, &mut rep.cov);
        rep.total_ticks += 1;
        t += 1;
        if (hit || moved)
            && trigger_at.is_none()
            && triggers < MAX_TRIGGERS
            && t >= cooldown_until
            && rng.chance(0.5)
        {
            trigger_at = Some(t + rng.int(0, 3));
        }
    }
    absorb_stats(&mut rep, &a);
    rep.main_ticks = t;
    rep.secs = t0.elapsed().as_secs_f32();
    rep
}

/// Plays `spec` with its bots, exactly as [`run`] does, and exports the arena before the first tick and then after every
/// `every` ticks: `count` dumps at most (fewer when the scenario ends first). The dumps are start states for tests of
/// another implementation; the scenario's kind ([`Kind::label`]) says what sort of situation they hold.
pub fn sample_states(spec: &Spec, every: u32, count: usize) -> Vec<ArenaDump> {
    let mut a = start_arena(spec);
    let mut bots = [Bot::new(spec.modes[0], spec.seed ^ 0x11), Bot::new(spec.modes[1], spec.seed ^ 0x22)];
    let mut cov = Coverage::default();
    let mut out = Vec::with_capacity(count);
    for t in 0..spec.ticks {
        if t % every == 0 {
            out.push(a.export_state());
            if out.len() == count {
                break;
            }
        }
        let ctrls = [bots[0].controls(&a, 0), bots[1].controls(&a, 1)];
        step_arena(&mut a, &ctrls, &mut cov);
    }
    out
}

/// Runs all specs on `threads` threads; reports come back in spec order.
pub fn run_all(specs: &[Spec], threads: usize) -> Vec<Report> {
    let next = AtomicUsize::new(0);
    let mut out: Vec<Option<Report>> = vec![None; specs.len()];
    let results = std::sync::Mutex::new(&mut out);
    std::thread::scope(|s| {
        for _ in 0..threads.max(1) {
            s.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= specs.len() {
                        break;
                    }
                    // a panic inside V3 (a bad import can break its invariants) is a failed scenario, not a crash
                    let r = std::panic::catch_unwind(|| run(&specs[i])).unwrap_or_else(|e| {
                        let mut r = Report::new(specs[i].name());
                        let msg = e
                            .downcast_ref::<String>()
                            .cloned()
                            .or_else(|| e.downcast_ref::<&str>().map(|s| (*s).to_string()))
                            .unwrap_or_default();
                        r.divergences.push(format!("PANIC {msg}"));
                        r
                    });
                    results.lock().unwrap()[i] = Some(r);
                }
            });
        }
    });
    out.into_iter().map(Option::unwrap).collect()
}

/// A multi-line table of the reports.
pub fn table(reports: &[Report]) -> String {
    let mut s = String::new();
    for r in reports {
        let _ = writeln!(s, "{}", r.line());
        for d in &r.divergences {
            let _ = writeln!(s, "    DIVERGENCE {d}");
        }
    }
    s
}
