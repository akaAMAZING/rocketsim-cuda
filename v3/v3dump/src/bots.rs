//! Random-control drivers. They are not good players: they exist to push the physics through car-ball and car-car
//! contacts, bumps and demos, wall and ceiling driving, flips and double jumps and pad pickups.

use std::collections::VecDeque;

use rocketsim::{Arena, CarControls, Vec3A};

use crate::rng::Rng;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Every control held for a random 3-45 ticks.
    Random,
    /// Drives at the ball with boost and flips into it.
    ChaseBall,
    /// Drives at the other car (bumps, supersonic demos).
    ChaseCar,
    /// Drives up a side wall onto the ceiling and jumps off again.
    WallRider,
    /// Jumps, boosts and spins with random air control; double jumps and flips at random.
    Aerial,
}

pub const MODES: [Mode; 5] = [
    Mode::Random,
    Mode::ChaseBall,
    Mode::ChaseCar,
    Mode::WallRider,
    Mode::Aerial,
];

pub struct Bot {
    pub mode: Mode,
    rng: Rng,
    hold: u32,
    cur: CarControls,
    queue: VecDeque<CarControls>,
    wall_side: f32,
    age: u32,
}

fn clamp1(x: f32) -> f32 {
    x.clamp(-1.0, 1.0)
}

impl Bot {
    pub fn new(mode: Mode, seed: u64) -> Self {
        let mut rng = Rng(seed);
        let wall_side = if rng.chance(0.5) { 1.0 } else { -1.0 };
        Self {
            mode,
            rng,
            hold: 0,
            cur: CarControls::default(),
            queue: VecDeque::new(),
            wall_side,
            age: 0,
        }
    }

    fn axis(&mut self) -> f32 {
        match self.rng.int(0, 3) {
            0 => 0.0,
            1 => 1.0,
            2 => -1.0,
            _ => self.rng.range(-1.0, 1.0),
        }
    }

    /// A jump press of `first` ticks, `gap` released ticks, then a two-tick dodge in the given direction.
    fn queue_dodge(&mut self, first: u32, gap: u32, pitch: f32, yaw: f32, roll: f32) {
        let press = CarControls::default().with_jump(true).with_boost(true);
        for _ in 0..first {
            self.queue.push_back(press);
        }
        for _ in 0..gap {
            self.queue.push_back(CarControls::default().with_boost(true));
        }
        let dodge = CarControls::default()
            .with_jump(true)
            .with_pitch(pitch)
            .with_yaw(yaw)
            .with_roll(roll);
        for _ in 0..2 {
            self.queue.push_back(dodge);
        }
    }

    fn random_dodge(&mut self) {
        let (first, gap) = (self.rng.int(1, 30), self.rng.int(3, 30));
        let (p, y, r) = (self.axis(), self.axis(), self.axis());
        self.queue_dodge(first, gap, p, y, r);
    }

    /// Controls for car `idx` this tick.
    pub fn controls(&mut self, arena: &Arena, idx: usize) -> CarControls {
        self.age += 1;
        if let Some(c) = self.queue.pop_front() {
            return c;
        }
        let me = arena.get_car_state(idx);
        let pos = me.phys.pos;
        let fwd = me.phys.rot_mat.x_axis;
        let right = me.phys.rot_mat.y_axis;
        let up = me.phys.rot_mat.z_axis;
        let on_ground = me.is_on_ground();
        let speed = me.phys.vel.length();
        let ball = arena.get_ball_state().phys.pos;
        let other = arena.get_car_state(1 - idx).phys.pos;

        match self.mode {
            Mode::Random => {
                if self.hold == 0 {
                    self.hold = self.rng.int(3, 45);
                    self.cur = CarControls {
                        throttle: self.axis(),
                        steer: self.axis(),
                        pitch: self.axis(),
                        yaw: self.axis(),
                        roll: self.axis(),
                        jump: self.rng.chance(0.25),
                        boost: self.rng.chance(0.5),
                        handbrake: self.rng.chance(0.1),
                    };
                }
                self.hold -= 1;
                self.cur
            }
            Mode::ChaseBall | Mode::ChaseCar => {
                let target = if self.mode == Mode::ChaseBall { ball } else { other };
                let to = target - pos;
                let flat = Vec3A::new(to.x, to.y, 0.0);
                let dist = flat.length().max(1.0);
                let dir = flat / dist;
                let side = right.dot(dir);
                let ahead = fwd.dot(dir);
                let mut c = CarControls {
                    throttle: if ahead > -0.3 || dist > 1500.0 { 1.0 } else { -1.0 },
                    steer: clamp1(side * 4.0 + self.rng.range(-0.15, 0.15)),
                    boost: ahead > 0.85 && dist > 500.0 && self.rng.chance(0.9),
                    handbrake: ahead < 0.2 && dist > 600.0 && self.rng.chance(0.2),
                    ..CarControls::default()
                };
                if on_ground && dist < 1100.0 && speed > 400.0 && self.rng.chance(0.05) {
                    // flip into the target: forward, slightly sideways
                    let (first, gap) = (self.rng.int(1, 6), self.rng.int(2, 14));
                    self.queue_dodge(first, gap, -1.0, clamp1(side * 0.5), 0.0);
                } else if on_ground && dist > 1500.0 && self.rng.chance(0.004) {
                    let gap = self.rng.int(2, 10);
                    self.queue_dodge(1, gap, -1.0, 0.0, 0.0);
                } else if !on_ground && self.rng.chance(0.03) {
                    c.pitch = self.axis();
                    c.yaw = self.axis();
                    c.roll = self.axis();
                }
                c
            }
            Mode::WallRider => {
                let mut c = CarControls::default().with_throttle(1.0).with_boost(true);
                if on_ground && up.z > 0.9 {
                    // aim at the wall
                    let dir = Vec3A::new(self.wall_side, 0.0, 0.0);
                    c.steer = clamp1(right.dot(dir) * 3.0 + self.rng.range(-0.2, 0.2));
                } else if on_ground {
                    // on a wall or the ceiling: keep climbing, wiggle, jump off now and then
                    c.steer = self.rng.range(-0.3, 0.3);
                    if self.age % 600 == 599 || self.rng.chance(0.002) {
                        let (p, y, r) = (self.axis(), self.axis(), self.axis());
                        self.queue_dodge(2, 6, p, y, r);
                    } else if pos.z > 1900.0 && self.rng.chance(0.01) {
                        c.jump = true;
                    }
                } else {
                    c.pitch = self.axis();
                    c.roll = self.axis();
                    c.boost = self.rng.chance(0.5);
                }
                c
            }
            Mode::Aerial => {
                if on_ground {
                    if self.rng.chance(0.08) {
                        self.random_dodge();
                    }
                    return CarControls::default()
                        .with_throttle(1.0)
                        .with_steer(self.rng.range(-0.5, 0.5))
                        .with_boost(self.rng.chance(0.5));
                }
                if self.hold == 0 {
                    self.hold = self.rng.int(5, 30);
                    self.cur = CarControls {
                        pitch: self.axis(),
                        yaw: self.axis(),
                        roll: self.axis(),
                        boost: self.rng.chance(0.7),
                        ..CarControls::default()
                    };
                }
                self.hold -= 1;
                if self.rng.chance(0.02) {
                    let (p, y, r) = (self.axis(), self.axis(), self.axis());
                    self.queue_dodge(1, 0, p, y, r);
                }
                self.cur.with_throttle(1.0)
            }
        }
    }
}
