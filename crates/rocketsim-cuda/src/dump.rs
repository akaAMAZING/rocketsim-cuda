//! V3 dump bytes (the export of the patched V3, v3/rocketsim-dump.patch) <-> `ArenaState` / `StaticData`.
//!
//! One traversal per section serves both directions, as in the patched V3 (`dump.rs` there): `io_arena` reads the stream
//! into an `ArenaState` or writes an `ArenaState` back as a V3 arena dump, field by field in the same order, so the two
//! directions cannot differ. Writing is what proves that `ArenaState` holds everything V3 needs: the dump rebuilt from an
//! `ArenaState` (with the constants of `StaticData` and a `Template` for what the state leaves out) is imported into V3,
//! which finishes the tick to the same state (tests/harness.rs).
//!
//! What the importer checks and refuses (it never silently drops state): a list over its cap, any Heatseeker, Dropshot or
//! Snowday field that is not at its default, a ball-world contact record, a non-empty free list, a wrong config.

use crate::layout::*;

const ARENA_MAGIC: &[u8; 8] = b"V3ARENA\0";
const STATIC_MAGIC: &[u8; 8] = b"V3STATIC";
pub const ARENA_VERSION: u32 = 2;
pub const STATIC_VERSION: u32 = 3;
pub const ARENA_HEADER_LEN: usize = 8 + 4 + 8 + 4 * 12;
pub const STATIC_HEADER_LEN: usize = 8 + 4 + 8 + 8 + 4 * 12;
/// Bytes of one `BODY` record.
pub const BODY_RECORD_LEN: usize = 219;
/// The `companion_id` option (1 + 4 bytes) sits at bytes 1..6 of a body record.
const COMPANION_AT: usize = 1;

fn tag(t: &[u8; 4]) -> u32 {
    u32::from_le_bytes(*t)
}

/// FNV-1a 64.
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h = (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

// ---- the byte stream, both directions ------------------------------------------------------------------------------------
pub trait Io {
    const READ: bool;
    fn u8(&mut self, v: &mut u8);
    fn u16(&mut self, v: &mut u16);
    fn u32(&mut self, v: &mut u32);
    fn u64(&mut self, v: &mut u64);
    fn f32(&mut self, v: &mut f32);
    fn raw(&mut self, v: &mut [u8]);
    fn fail(&mut self, msg: &str);
    /// Element count of a sequence.
    fn count(&mut self, n: usize) -> usize;

    fn tag(&mut self, t: &[u8; 4]) {
        let mut x = tag(t);
        self.u32(&mut x);
        if x != tag(t) {
            self.fail(&format!("section tag {} expected", String::from_utf8_lossy(t)));
        }
    }
    /// A V3 bool as one byte, held as a u32 flag.
    fn flag(&mut self, v: &mut u32) {
        let mut b = u8::from(*v != 0);
        self.u8(&mut b);
        *v = u32::from(b);
    }
    fn vec3(&mut self, v: &mut Vec3) {
        self.f32(&mut v.x);
        self.f32(&mut v.y);
        self.f32(&mut v.z);
    }
    fn mat3(&mut self, m: &mut Mat3) {
        self.vec3(&mut m.x_axis);
        self.vec3(&mut m.y_axis);
        self.vec3(&mut m.z_axis);
    }
    fn aff3(&mut self, t: &mut Affine3) {
        self.mat3(&mut t.matrix3);
        self.vec3(&mut t.translation);
    }
    fn quat(&mut self, q: &mut Quat) {
        self.f32(&mut q.x);
        self.f32(&mut q.y);
        self.f32(&mut q.z);
        self.f32(&mut q.w);
    }
    fn aabb(&mut self, a: &mut Aabb) {
        self.vec3(&mut a.min);
        self.vec3(&mut a.max);
    }
    fn u64s(&mut self, v: &mut U64) {
        let mut x = v.get();
        self.u64(&mut x);
        *v = U64::set(x);
    }
    /// `opt<T>` of the schema: a flag byte, then the payload always at its full size.
    fn opt<T: Copy>(&mut self, v: &mut Opt<T>, mut f: impl FnMut(&mut Self, &mut T)) {
        self.flag(&mut v.some);
        f(self, &mut v.v);
    }
    fn opt_idx(&mut self, v: &mut Opt<u32>) {
        self.opt(v, |io, x| io.u32(x));
    }
    fn opt_u64(&mut self, v: &mut Opt<U64>) {
        self.opt(v, |io, x| io.u64s(x));
    }
    fn opt_vec3(&mut self, v: &mut Opt<Vec3>) {
        self.opt(v, |io, x| io.vec3(x));
    }
}

#[derive(Default)]
pub struct Wr {
    pub buf: Vec<u8>,
    pub err: Option<String>,
}

impl Io for Wr {
    const READ: bool = false;
    fn u8(&mut self, v: &mut u8) {
        self.buf.push(*v);
    }
    fn u16(&mut self, v: &mut u16) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn u32(&mut self, v: &mut u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: &mut u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn f32(&mut self, v: &mut f32) {
        self.buf.extend_from_slice(&v.to_bits().to_le_bytes());
    }
    fn raw(&mut self, v: &mut [u8]) {
        self.buf.extend_from_slice(v);
    }
    fn fail(&mut self, msg: &str) {
        self.err.get_or_insert_with(|| msg.to_string());
    }
    fn count(&mut self, n: usize) -> usize {
        self.buf.extend_from_slice(&(n as u32).to_le_bytes());
        n
    }
}

pub struct Rd<'a> {
    b: &'a [u8],
    p: usize,
    pub err: Option<String>,
}

impl<'a> Rd<'a> {
    pub fn new(b: &'a [u8]) -> Self {
        Self { b, p: 0, err: None }
    }
    pub fn remaining(&self) -> usize {
        self.b.len() - self.p
    }
    fn take<const N: usize>(&mut self) -> [u8; N] {
        if self.remaining() < N {
            self.err.get_or_insert_with(|| "dump truncated".to_string());
            self.p = self.b.len();
            return [0; N];
        }
        let mut o = [0u8; N];
        o.copy_from_slice(&self.b[self.p..self.p + N]);
        self.p += N;
        o
    }
}

impl Io for Rd<'_> {
    const READ: bool = true;
    fn u8(&mut self, v: &mut u8) {
        *v = self.take::<1>()[0];
    }
    fn u16(&mut self, v: &mut u16) {
        *v = u16::from_le_bytes(self.take());
    }
    fn u32(&mut self, v: &mut u32) {
        *v = u32::from_le_bytes(self.take());
    }
    fn u64(&mut self, v: &mut u64) {
        *v = u64::from_le_bytes(self.take());
    }
    fn f32(&mut self, v: &mut f32) {
        *v = f32::from_bits(u32::from_le_bytes(self.take()));
    }
    fn raw(&mut self, v: &mut [u8]) {
        if self.remaining() < v.len() {
            self.err.get_or_insert_with(|| "dump truncated".to_string());
            self.p = self.b.len();
            return;
        }
        v.copy_from_slice(&self.b[self.p..self.p + v.len()]);
        self.p += v.len();
    }
    fn fail(&mut self, msg: &str) {
        self.err.get_or_insert_with(|| msg.to_string());
    }
    fn count(&mut self, _n: usize) -> usize {
        let mut x = 0u32;
        self.u32(&mut x);
        if x as usize > self.remaining() {
            self.err.get_or_insert_with(|| "count exceeds the bytes left".to_string());
            return 0;
        }
        x as usize
    }
}

// ---- the template: what an ArenaState leaves out ---------------------------------------------------------------------------
/// What the writer needs besides `ArenaState` and `StaticData` to rebuild a V3 arena dump: the raw records of the static
/// bodies (their poses and inertias never change; their `companion_id` is cleared) and the broadphase counters. Taken from any dump of the same configuration.
#[derive(Clone)]
pub struct Template {
    pub config_hash: u64,
    /// `BODY` record of every body; the dynamic bodies' entries are unused.
    pub static_body: Vec<[u8; BODY_RECORD_LEN]>,
    pub num_handles: u16,
    pub next_new_handle: u16,
    pub manifold_stride: u32,
    /// The Heatseeker and Dropshot ball info (24 bytes; Soccar never changes it) and whether one was read.
    pub ball_other: [u8; 24],
    pub ball_other_set: bool,
}

impl Template {
    fn empty() -> Self {
        Self {
            config_hash: 0,
            static_body: vec![[0; BODY_RECORD_LEN]; NUM_BODIES],
            num_handles: 0,
            next_new_handle: 0,
            manifold_stride: 0,
            ball_other: [0; 24],
            ball_other_set: false,
        }
    }
}

// ---- the arena traversal ----------------------------------------------------------------------------------------------------
fn io_controls<I: Io>(io: &mut I, c: &mut Controls) {
    io.f32(&mut c.throttle);
    io.f32(&mut c.steer);
    io.f32(&mut c.pitch);
    io.f32(&mut c.yaw);
    io.f32(&mut c.roll);
    io.flag(&mut c.jump);
    io.flag(&mut c.boost);
    io.flag(&mut c.handbrake);
}

fn io_hit<I: Io>(io: &mut I, h: &mut HitInfo) {
    io.vec3(&mut h.hit_point);
    io.vec3(&mut h.hit_normal);
    io.f32(&mut h.hit_fraction);
    let mut u = h.user_info as u8;
    io.u8(&mut u);
    h.user_info = u32::from(u);
}

fn io_point<I: Io>(io: &mut I, p: &mut ManifoldPoint) {
    io.vec3(&mut p.local_point_a);
    io.vec3(&mut p.local_point_b);
    io.vec3(&mut p.pos_world_on_b);
    io.vec3(&mut p.pos_world_on_a);
    io.vec3(&mut p.normal_world_on_b);
    io.f32(&mut p.distance_1);
    io.f32(&mut p.combined_friction);
    io.f32(&mut p.combined_restitution);
    // applied_impulse: V3 never writes it back (always 0); not in ArenaState
    let mut applied = 0.0f32;
    io.f32(&mut applied);
    if I::READ && applied.to_bits() != 0 {
        io.fail("manifold point with a non-zero applied_impulse");
    }
    io.flag(&mut p.is_special);
}

fn io_wheel<I: Io>(io: &mut I, w: &mut Wheel) {
    io.opt(&mut w.raycast_info, |io, r| {
        io.vec3(&mut r.contact_normal);
        io.vec3(&mut r.contact_point);
        io.u32(&mut r.ground_body_idx);
        io.f32(&mut r.suspension_length);
        io.vec3(&mut r.impulse);
        io.opt_vec3(&mut r.ground_stick);
        io.flag(&mut r.is_in_contact_with_world);
        io.f32(&mut r.clipped_inv_contact_dot_suspension);
        io.f32(&mut r.suspension_relative_vel);
    });
    io.vec3(&mut w.hard_point);
    io.vec3(&mut w.axle_dir);
    io.f32(&mut w.engine_force);
    io.f32(&mut w.brake);
    io.f32(&mut w.steer_angle);
    io.vec3(&mut w.vel_at_contact_point);
    io.f32(&mut w.lat_friction);
    io.f32(&mut w.long_friction);
    io.f32(&mut w.extra_pushback);
}

fn io_car<I: Io>(io: &mut I, c: &mut Car) {
    io.f32(&mut c.dodge_deadzone);
    io.vec3(&mut c.vel_impulse_cache);
    io.flag(&mut c.sticky_gate_prev);
    io.vec3(&mut c.pos);
    io.mat3(&mut c.rot_mat);
    io.vec3(&mut c.vel);
    io.vec3(&mut c.ang_vel);
    io_controls(io, &mut c.controls);
    io.flag(&mut c.has_jumped);
    io.flag(&mut c.has_double_jumped);
    io.flag(&mut c.has_flipped);
    io.vec3(&mut c.flip_rel_torque);
    io.u32(&mut c.jump_ticks);
    io.f32(&mut c.flip_time);
    io.flag(&mut c.is_flipping);
    io.flag(&mut c.is_jumping);
    io.flag(&mut c.jump_hold_broken);
    io.f32(&mut c.air_time);
    io.f32(&mut c.air_time_since_jump);
    io.f32(&mut c.boost);
    io.f32(&mut c.handbrake_val);
    io.flag(&mut c.is_supersonic);
    io.f32(&mut c.supersonic_grace_timer);
    io.f32(&mut c.time_since_boosted);
    io.f32(&mut c.boosting_time);
    io.flag(&mut c.is_boosting);
    io.flag(&mut c.is_auto_flipping);
    io.f32(&mut c.auto_flip_timer);
    io.f32(&mut c.auto_flip_torque_scale);
    io.f32(&mut c.bump_cooldown_timer);
    io.opt_idx(&mut c.bump_cooldown_victim);
    io.opt_u64(&mut c.last_extra_hit_tick);
    io.flag(&mut c.is_demoed);
    io.f32(&mut c.demo_respawn_timer);
    io.flag(&mut c.is_on_ground);
    for h in &mut c.wheels_with_contact {
        io.opt(h, io_hit);
    }
    io.opt_vec3(&mut c.world_contact_normal);
    io_controls(io, &mut c.prev_controls);
    for w in &mut c.wheels {
        io_wheel(io, w);
    }
}

/// One body record of a dynamic body.
fn io_body<I: Io>(io: &mut I, b: &mut Body) {
    let mut flags = b.collision_flags as u8;
    io.u8(&mut flags);
    b.collision_flags = u32::from(flags);
    // companion_id: the solver-body link, reassigned by P5's setup before any read; not state (written as none)
    let mut companion = Opt::<u32>::default();
    io.opt_idx(&mut companion);
    let mut act = b.activation as u8;
    io.u8(&mut act);
    b.activation = u32::from(act);
    io.aff3(&mut b.world_trans);
    io.quat(&mut b.world_quat);
    io.aff3(&mut b.interp_world_trans);
    io.f32(&mut b.deactivation_time);
    io.vec3(&mut b.lin_vel);
    io.vec3(&mut b.ang_vel);
    io.mat3(&mut b.inv_inertia_tensor_world);
    io.vec3(&mut b.accum_lin_vel);
    io.vec3(&mut b.accum_ang_vel);
    io.f32(&mut b.cached_linear_damping);
    io.f32(&mut b.cached_damping_time_step);
    io.f32(&mut b.cached_linear_damping_value);
}

fn table_index(id0: u32, id1: u32) -> usize {
    let (lo, hi) = if id0 <= id1 { (id0, id1) } else { (id1, id0) };
    (hi as usize) * (hi as usize + 1) / 2 + lo as usize
}

// ---- the record fold ---------------------------------------------------------------------------------------
// EventBit positions (cuda/sim/arena_state.cuh).
const EV_CAR_HIT_WORLD: u32 = 10;
const EV_BALL_HIT_WORLD: u32 = 12;
const EV_BALL_WALL: u32 = 14;
// UserInfoType (sim/user_info_types.rs)
const UI_NONE: u32 = 0;
const UI_CAR: u32 = 1;
const UI_BALL: u32 = 2;

/// V3's contact records of a tick (every one, in call order) as ArenaState holds them: the same mapping as the device's
/// `tracker_push_record` (cuda/sim/arena_contact_tracker.cuh), written independently of it. A record with user a = Car and
/// user b = Ball or Car is stored in order (`records`); car-world: bit `EV_CAR_HIT_WORLD + car` of `fold_events`, and
/// `fold_car_normal[car]` = the normal of the last such record of that car; ball-world (user a = Ball, b = None): bit
/// `EV_BALL_HIT_WORLD`, plus `EV_BALL_WALL` if any has |normal.z| < 0.8; any other record is a no-op in P7 and is dropped.
/// This is the mapping the comparator relies on: a V3 state at boundaries 5 to 8 is folded the same way before it is compared.
/// Errors when more than `MAX_RECORDS` are stored.
pub fn fold_records(s: &mut ArenaState, sd: &StaticData, recs: &[ContactRecord]) -> Result<(), String> {
    s.fold_events = 0;
    s.fold_car_normal = [Vec3::default(); NUM_CARS];
    s.n_records = 0;
    s.records = unsafe { core::mem::zeroed() };
    for r in recs {
        let a = &sd.bodies[r.rb_idx_a as usize];
        let b = &sd.bodies[r.rb_idx_b as usize];
        if a.user_idx == UI_CAR && (b.user_idx == UI_BALL || b.user_idx == UI_CAR) {
            if s.n_records as usize >= MAX_RECORDS {
                return Err("stored contact records over the cap".into());
            }
            s.records[s.n_records as usize] = *r;
            s.n_records += 1;
        } else if a.user_idx == UI_CAR {
            let car = a.user_pointer as usize;
            s.fold_events |= 1 << (EV_CAR_HIT_WORLD + car as u32);
            s.fold_car_normal[car] = r.point.normal_world_on_b;
        } else if a.user_idx == UI_BALL && b.user_idx == UI_NONE {
            s.fold_events |= 1 << EV_BALL_HIT_WORLD;
            if r.point.normal_world_on_b.z.abs() < 0.8 {
                s.fold_events |= 1 << EV_BALL_WALL;
            }
        }
    }
    Ok(())
}

/// The inverse for writing a V3 dump: records that V3's P7 turns into the same state (one ball-world record whose normal gives
/// the same two bits, one car-world record per car with its normal, then the stored records). Order is free by the proof.
pub fn unfold_records(s: &ArenaState, sd: &StaticData) -> Vec<ContactRecord> {
    let mut out = Vec::new();
    let world = (0..NUM_BODIES as u32).find(|&i| sd.bodies[i as usize].user_idx == UI_NONE).expect("a static body");
    let rec = |a: u32, n: Vec3| {
        let mut r: ContactRecord = unsafe { core::mem::zeroed() };
        r.rb_idx_a = a;
        r.rb_idx_b = world;
        r.point.normal_world_on_b = n;
        r
    };
    if s.fold_events >> EV_BALL_HIT_WORLD & 1 != 0 {
        let n = if s.fold_events >> EV_BALL_WALL & 1 != 0 { Vec3 { x: 1.0, y: 0.0, z: 0.0 } } else { Vec3 { x: 0.0, y: 0.0, z: 1.0 } };
        out.push(rec(sd.dyn_body[0], n));
    }
    for car in 0..NUM_CARS {
        if s.fold_events >> (EV_CAR_HIT_WORLD + car as u32) & 1 != 0 {
            out.push(rec(sd.dyn_body[1 + car], s.fold_car_normal[car]));
        }
    }
    out.extend_from_slice(&s.records[..s.n_records as usize]);
    out
}

/// Reads or writes the sections after the 68-byte header. On write, `s` must already satisfy the caps.
fn io_arena<I: Io>(io: &mut I, s: &mut ArenaState, sd: &StaticData, t: &mut Template) {
    // ARNA
    io.tag(b"ARNA");
    io.u64s(&mut s.tick_count);
    io.u64s(&mut s.rng);
    // ball_only: always 0 in 1v1 (the importer refuses an arena without cars)
    let mut ball_only = 0u32;
    io.flag(&mut ball_only);
    if I::READ && ball_only != 0 {
        io.fail("a ball-only arena");
    }
    // BALL
    io.tag(b"BALL");
    io.vec3(&mut s.ball.pos);
    io.mat3(&mut s.ball.rot_mat);
    io.vec3(&mut s.ball.vel);
    io.vec3(&mut s.ball.ang_vel);
    // Heatseeker and Dropshot info (i8, f32, f32 | u8, f32, i8, opt<u64>): other modes, must be zero
    let mut other = t.ball_other;
    io.raw(&mut other);
    if I::READ {
        if !t.ball_other_set {
            (t.ball_other, t.ball_other_set) = (other, true);
        } else if other != t.ball_other {
            io.fail("Heatseeker or Dropshot ball info changed between dumps");
        }
    }
    io.u64s(&mut s.ball.tick_count_since_kickoff);
    let mut ground_stick = 0u8;
    io.u8(&mut ground_stick);
    if I::READ && ground_stick != 0 {
        io.fail("Snowday ground_stick_applied is set");
    }
    // CARS
    io.tag(b"CARS");
    let n = io.count(NUM_CARS);
    if n != NUM_CARS {
        io.fail("not a 2-car arena");
        return;
    }
    for c in &mut s.cars {
        io_car(io, c);
    }
    // PADS
    io.tag(b"PADS");
    let n = io.count(NUM_PADS);
    if n != NUM_PADS {
        io.fail("not a 34-pad arena");
        return;
    }
    for (gave, touching) in s.pad_gave_boost.iter_mut().zip(&mut s.pad_touching) {
        io.opt_u64(gave);
        // touching_car_idxs (arena dump version 2): arrival order; one pad per car, so at most NUM_CARS
        let n = io.count(touching.n as usize);
        if n > NUM_CARS {
            io.fail("a pad touched by more cars than the arena has");
            return;
        }
        touching.n = n as u32;
        for k in 0..n {
            idx(io, &mut touching.car_idx[k]);
        }
    }
    // CONT: V3 lists every record in call order; ArenaState folds the ball-world and car-world ones (fold_records)
    io.tag(b"CONT");
   let mut list: Vec<ContactRecord> = if I::READ { Vec::new() } else { unfold_records(s, sd) };
    let n = io.count(list.len());
    if I::READ {
        list = vec![unsafe { core::mem::zeroed() }; n];
    }
    for r in &mut list {
        io.flag(&mut r.is_swap);
        io.u32(&mut r.rb_idx_a);
        io.u32(&mut r.rb_idx_b);
        // P7 reads the two world points and the normal; the rest of V3's manifold point is not kept (zero on write)
        let mut p = ManifoldPoint {
            local_point_a: Vec3::default(),
            local_point_b: Vec3::default(),
            pos_world_on_b: r.point.pos_world_on_b,
            pos_world_on_a: r.point.pos_world_on_a,
            normal_world_on_b: r.point.normal_world_on_b,
            distance_1: 0.0,
            combined_friction: 0.0,
            combined_restitution: 0.0,
            is_special: 0,
        };
        io_point(io, &mut p);
        r.point = RecordPoint { pos_world_on_b: p.pos_world_on_b, pos_world_on_a: p.pos_world_on_a, normal_world_on_b: p.normal_world_on_b };
    }
    if I::READ {
        match fold_records(s, sd, &list) {
            Ok(()) => {}
            Err(e) => {
                s.overflow[2] += 1;
                io.fail(&e);
                return;
            }
        }
    }
    let nb = io.count(0); // ball-world records: ball_only arenas only
    if nb != 0 {
        io.fail("ball-world contact records present");
        return;
    }
    let mut ball_only_tracker = 0u32;
    io.flag(&mut ball_only_tracker);
    // BODY
    io.tag(b"BODY");
    let n = io.count(NUM_BODIES);
    if n != NUM_BODIES {
        io.fail("not a 23-body arena");
        return;
    }
    for i in 0..NUM_BODIES {
        if let Some(d) = sd.dyn_body.iter().position(|&b| b as usize == i) {
            io_body(io, &mut s.bodies[d]);
        } else if I::READ {
            let mut rec = [0u8; BODY_RECORD_LEN];
            io.raw(&mut rec);
            rec[COMPANION_AT..COMPANION_AT + 5].fill(0);
            // static bodies never change (checked by the round-trip tests); keep the first record seen
            if t.static_body[i].iter().all(|&b| b == 0) {
                t.static_body[i] = rec;
            }
        } else {
            let mut rec = t.static_body[i];
            io.raw(&mut rec);
        }
    }
    // DISP
    io.tag(b"DISP");
    let nm = io.count(s.n_manifolds as usize);
    if I::READ {
        if nm > MAX_MANIFOLDS {
            s.overflow[0] += 1;
            io.fail("retained manifolds over the cap");
            return;
        }
        s.n_manifolds = nm as u32;
    }
    for m in &mut s.manifolds[..nm] {
        io.u32(&mut m.body0_idx);
        io.u32(&mut m.body1_idx);
        let key = (u64::from(m.body0_idx.min(m.body1_idx)) << 32) | u64::from(m.body0_idx.max(m.body1_idx));
        let mut got = key;
        io.u64(&mut got);
        if I::READ && got != key {
            io.fail("manifold pair_key is not min << 32 | max");
        }
        // the two thresholds are the min over the manifold's two bodies (persistent_manifold.rs:60-65)
        let (b0, b1) = (m.body0_idx as usize, m.body1_idx as usize);
        if b0 >= NUM_BODIES || b1 >= NUM_BODIES {
            io.fail("manifold body index out of range");
            return;
        }
        let mut cbt = sd.bodies[b0].contact_breaking_threshold.min(sd.bodies[b1].contact_breaking_threshold);
        let mut cpt = sd.bodies[b0].contact_processing_threshold.min(sd.bodies[b1].contact_processing_threshold);
        let (want_cbt, want_cpt) = (cbt, cpt);
        io.f32(&mut cbt);
        io.f32(&mut cpt);
        if I::READ && (cbt.to_bits() != want_cbt.to_bits() || cpt.to_bits() != want_cpt.to_bits()) {
            io.fail("a manifold threshold is not the min over its two bodies");
            return;
        }
        io.u32(&mut m.point_count);
        if m.point_count as usize > MAX_POINTS {
            io.fail("manifold with more than 4 points");
            return;
        }
        for p in &mut m.points {
            io_point(io, p);
        }
    }
    let na = io.count(s.n_active as usize);
    if I::READ {
        if na > MAX_ACTIVE {
            s.overflow[1] += 1;
            io.fail("active manifolds over the cap");
            return;
        }
        s.n_active = na as u32;
    }
    for a in &mut s.active[..na] {
        io.u32(a);
    }
    // manifold_stride and manifold_table: rebuilt as searches, so only validated on read, rebuilt on write
    let mut stride = t.manifold_stride;
    if !I::READ {
        stride = if nm > 0 { 32 } else { t.manifold_stride };
    }
    io.u32(&mut stride);
    let tl = io.count((stride as usize) * (stride as usize));
    if I::READ {
        t.manifold_stride = stride;
        if tl != (stride as usize) * (stride as usize) {
            io.fail("manifold table length is not stride squared");
            return;
        }
        let mut expect = vec![0u32; tl];
        for (i, m) in s.manifolds[..nm].iter().enumerate() {
            let (lo, hi) = (m.body0_idx.min(m.body1_idx) as usize, m.body0_idx.max(m.body1_idx) as usize);
            if hi >= stride as usize {
                io.fail("manifold body index outside the table");
                return;
            }
            expect[lo * stride as usize + hi] = i as u32 + 1;
        }
        for e in expect {
            let mut got = 0u32;
            io.u32(&mut got);
            if got != e {
                io.fail("manifold table is not the search of the pool");
                return;
            }
        }
    } else {
        let mut table = vec![0u32; tl];
        for (i, m) in s.manifolds[..nm].iter().enumerate() {
            let (lo, hi) = (m.body0_idx.min(m.body1_idx) as usize, m.body0_idx.max(m.body1_idx) as usize);
            table[lo * stride as usize + hi] = i as u32 + 1;
        }
        for mut e in table {
            io.u32(&mut e);
        }
    }
    // BROA: dynamic ray-cell lists (cells ascending, each: cell, handles in list order)
    io.tag(b"BROA");
    if I::READ {
        let ncells = io.count(0);
        let mut entries: Vec<DynCell> = Vec::new();
        let mut last = None;
        for _ in 0..ncells {
            let mut cell = 0u32;
            io.u32(&mut cell);
            if last.is_some_and(|l| l >= cell) {
                io.fail("dynamic ray cells not ascending");
                return;
            }
            last = Some(cell);
            let nh = io.count(0);
            for _ in 0..nh {
                let mut h = 0u32;
                io.u32(&mut h);
                entries.push(DynCell { cell, handle: h });
            }
        }
        if entries.len() > MAX_DYN_CELLS {
            s.overflow[4] += 1;
            io.fail("dynamic ray-cell entries over the cap");
            return;
        }
        s.n_dyn_cells = entries.len() as u32;
        s.dyn_cells = [DynCell { cell: 0, handle: 0 }; MAX_DYN_CELLS];
        s.dyn_cells[..entries.len()].copy_from_slice(&entries);
    } else {
        let list = &s.dyn_cells[..s.n_dyn_cells as usize];
        let mut groups: Vec<(u32, Vec<u32>)> = Vec::new();
        for e in list {
            match groups.last_mut() {
                Some((c, hs)) if *c == e.cell => hs.push(e.handle),
                _ => groups.push((e.cell, vec![e.handle])),
            }
        }
        io.count(groups.len());
        for (mut c, hs) in groups {
            io.u32(&mut c);
            io.count(hs.len());
            for mut h in hs {
                io.u32(&mut h);
            }
        }
    }
    // sweep-and-prune handles: the static proxies are constants (StaticData.proxies), the dynamic ones are state
    let nh = io.count(NUM_HANDLES);
    if nh != NUM_HANDLES {
        io.fail("not a 24-handle broadphase");
        return;
    }
    for h in 0..NUM_HANDLES {
        let mut p = sd.proxies[h];
        let dyn_idx = sd.dyn_handle.iter().position(|&d| d as usize == h);
        io.u32(&mut p.client_obj_idx);
        let (mut g, mut m) = (p.filter_group as u8, p.filter_mask as u8);
        io.u8(&mut g);
        io.u8(&mut m);
        io.u32(&mut p.unique_id);
        if let Some(d) = dyn_idx {
            io.aabb(&mut s.dyn_proxy[d].aabb);
            io.u32(&mut s.dyn_proxy[d].cell_idx);
            for i in &mut s.dyn_proxy[d].indices {
                io.u32(i);
            }
        } else {
            let mut tmp = p;
            io.aabb(&mut tmp.aabb);
            io.u32(&mut tmp.cell_idx);
            for i in &mut tmp.indices {
                io.u32(i);
            }
            if I::READ && (tmp.aabb != p.aabb || tmp.cell_idx != p.cell_idx || tmp.indices != p.indices) {
                io.fail("a static broadphase proxy differs from StaticData.proxies");
                return;
            }
        }
        if I::READ
            && (p.client_obj_idx != sd.proxies[h].client_obj_idx
                || u32::from(g) != sd.proxies[h].filter_group
                || u32::from(m) != sd.proxies[h].filter_mask
                || p.unique_id != sd.proxies[h].unique_id)
        {
            io.fail("a broadphase proxy constant differs from StaticData.proxies");
            return;
        }
    }
    let nhe = io.count(NUM_HANDLES);
    if nhe != NUM_HANDLES {
        io.fail("handle_edges length");
        return;
    }
    for e in &mut s.handle_edges {
        for v in e.min.iter_mut().chain(e.max.iter_mut()) {
            io.u16(v);
        }
    }
    for axis in &mut s.edges {
        let n = io.count(NUM_EDGES);
        if n != NUM_EDGES {
            io.fail("edge array length");
            return;
        }
        for e in axis.iter_mut() {
            io.u16(&mut e.pos);
            io.u16(&mut e.handle);
        }
    }
    let mut nh16 = t.num_handles;
    io.u16(&mut nh16);
    let nfree = io.count(0);
    if nfree != 0 {
        io.fail("free_list is not empty");
        return;
    }
    let mut next = t.next_new_handle;
    io.u16(&mut next);
    if I::READ {
        t.num_handles = nh16;
        t.next_new_handle = next;
    }
    let np = io.count(s.n_pairs as usize);
    if I::READ {
        if np > MAX_PAIRS {
            s.overflow[3] += 1;
            io.fail("retained pairs over the cap");
            return;
        }
        s.n_pairs = np as u32;
    }
    for p in &mut s.pairs[..np] {
        io.u32(&mut p.proxy0);
        io.u32(&mut p.proxy1);
    }
    // pair_indices: the triangular table is a search over the pair array (length: largest index in use + 1)
    let want: Vec<(usize, u32)> = s.pairs[..np]
        .iter()
        .enumerate()
        .map(|(i, p)| (table_index(p.proxy0, p.proxy1), i as u32 + 1))
        .collect();
    if I::READ {
        let len = io.count(0);
        let mut got = vec![0u32; len];
        for g in &mut got {
            io.u32(g);
        }
        let mut seen = 0;
        for (i, &g) in got.iter().enumerate() {
            if g != 0 {
                seen += 1;
                if !want.iter().any(|&(ti, v)| ti == i && v == g) {
                    io.fail("pair index table is not the search of the pair array");
                    return;
                }
            }
        }
        if seen != np {
            io.fail("pair index table and pair array disagree");
            return;
        }
    } else {
        let len = want.iter().map(|&(i, _)| i + 1).max().unwrap_or(0);
        io.count(len);
        let mut table = vec![0u32; len];
        for (i, v) in want {
            table[i] = v;
        }
        for mut e in table {
            io.u32(&mut e);
        }
    }
}

/// The `ISLD` section: the island manager is a memo of the manifold sort. Skipped on read (two u32
/// sequences, two i64 sequences, two u32 caches, one bool), written empty.
fn io_island<I: Io>(io: &mut I) {
    io.tag(b"ISLD");
    for width in [4usize, 4, 8, 8, 4, 4] {
        let n = io.count(0);
        if I::READ {
            let mut junk = vec![0u8; n * width];
            io.raw(&mut junk);
        }
    }
    let mut ball_only = 0u32;
    io.flag(&mut ball_only);
}

// ---- public API: arena dumps ----------------------------------------------------------------------------------------------------
pub struct ArenaHeader {
    pub version: u32,
    pub config_hash: u64,
    pub counts: [u32; 12],
}

pub fn arena_header(bytes: &[u8]) -> Result<ArenaHeader, String> {
    if bytes.len() < ARENA_HEADER_LEN || &bytes[..8] != ARENA_MAGIC {
        return Err("not an arena dump (magic)".into());
    }
    let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    if version != ARENA_VERSION {
        return Err(format!("unsupported arena dump version {version}"));
    }
    let config_hash = u64::from_le_bytes(bytes[12..20].try_into().unwrap());
    let counts = core::array::from_fn(|i| u32::from_le_bytes(bytes[20 + 4 * i..24 + 4 * i].try_into().unwrap()));
    Ok(ArenaHeader { version, config_hash, counts })
}

/// A V3 arena dump (tick or phase boundary) as an `ArenaState`, plus the template of what it leaves out.
pub fn parse_arena(bytes: &[u8], sd: &StaticData) -> Result<(Box<ArenaState>, Template), String> {
    let h = arena_header(bytes)?;
    let mut s = ArenaState::zeroed();
    let mut t = Template::empty();
    t.config_hash = h.config_hash;
    let mut r = Rd::new(&bytes[ARENA_HEADER_LEN..]);
    io_arena(&mut r, &mut s, sd, &mut t);
    io_island(&mut r);
    if let Some(e) = r.err {
        return Err(e);
    }
    if r.remaining() != 0 {
        return Err(format!("{} trailing bytes", r.remaining()));
    }
    Ok((s, t))
}

/// An `ArenaState` as a V3 arena dump version 1 (import it with `Arena::import_state`).
pub fn write_arena(s: &ArenaState, sd: &StaticData, t: &Template) -> Result<Vec<u8>, String> {
    let mut s = Box::new(*s);
    let mut t = t.clone();
    let mut body = Wr::default();
    io_arena(&mut body, &mut s, sd, &mut t);
    io_island(&mut body);
    if let Some(e) = body.err {
        return Err(e);
    }
    let nm = s.n_manifolds as usize;
    let points: u32 = s.manifolds[..nm].iter().map(|m| m.point_count).sum();
    let stride = if nm > 0 { 32u32 } else { t.manifold_stride };
    let dyn_entries = s.n_dyn_cells;
    let counts: [u32; 12] = [
        NUM_BODIES as u32,
        nm as u32,
        s.n_active,
        points,
        stride * stride,
        NUM_HANDLES as u32,
        s.n_pairs,
        dyn_entries,
        NUM_CARS as u32,
        NUM_PADS as u32,
        unfold_records(&s, sd).len() as u32,
        0,
    ];
    let mut out = Vec::with_capacity(ARENA_HEADER_LEN + body.buf.len());
    out.extend_from_slice(ARENA_MAGIC);
    out.extend_from_slice(&ARENA_VERSION.to_le_bytes());
    out.extend_from_slice(&t.config_hash.to_le_bytes());
    for c in counts {
        out.extend_from_slice(&c.to_le_bytes());
    }
    out.extend_from_slice(&body.buf);
    Ok(out)
}

// ---- static dumps ---------------------------------------------------------------------------------------------------------------
pub struct StaticHeader {
    pub hash: u64,
    pub counts: [u32; 12],
}

pub fn static_header(bytes: &[u8]) -> Result<StaticHeader, String> {
    if bytes.len() < STATIC_HEADER_LEN || &bytes[..8] != STATIC_MAGIC {
        return Err("not a static dump (magic)".into());
    }
    let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    if version != STATIC_VERSION {
        return Err(format!("unsupported static dump version {version} (need {STATIC_VERSION}: re-vendor V3)"));
    }
    let hash = u64::from_le_bytes(bytes[12..20].try_into().unwrap());
    let len = u64::from_le_bytes(bytes[20..28].try_into().unwrap()) as usize;
    if len != bytes.len() - STATIC_HEADER_LEN {
        return Err("static dump payload length mismatch".into());
    }
    if fnv1a64(&bytes[STATIC_HEADER_LEN..]) != hash {
        return Err("static dump payload hash mismatch".into());
    }
    let counts = core::array::from_fn(|i| u32::from_le_bytes(bytes[28 + 4 * i..32 + 4 * i].try_into().unwrap()));
    Ok(StaticHeader { hash, counts })
}

fn expect<I: Io>(io: &mut I, got: usize, want: usize, what: &str) -> bool {
    if got != want {
        io.fail(&format!("{what}: {got}, StaticData holds {want}"));
        return false;
    }
    true
}

/// V3's static dump (version 2) as `StaticData`.
pub fn parse_static(bytes: &[u8]) -> Result<Box<StaticData>, String> {
    let h = static_header(bytes)?;
    let c = h.counts;
    let want = [
        (c[0], NUM_BODIES),
        (c[1], NUM_DYN),
        (c[2], NUM_MESHES),
        (c[3], MAX_TRIS),
        (c[4], MAX_NODES),
        (c[5], MAX_LEAVES),
        (c[6], MAX_TRIS),
        (c[7], NUM_PLANES),
        (c[8], NUM_CARS),
        (c[9], NUM_PADS),
        (c[10], NUM_GRID_CELLS),
        (c[11], MAX_CELL_ENTRIES),
    ];
    for (i, &(got, w)) in want.iter().enumerate() {
        if got as usize != w {
            return Err(format!("static dump count {i} is {got}; StaticData is sized for {w} (the project's 16 Soccar meshes)"));
        }
    }
    let mut sd = StaticData::zeroed();
    let mut r = Rd::new(&bytes[STATIC_HEADER_LEN..]);
    io_static(&mut r, &mut sd);
    if let Some(e) = r.err {
        return Err(e);
    }
    // V3's table_sin (car/base.rs:101), once: the platform f64 sin of this build, as V3 on this platform computes it
    for (i, t) in sd.table_sin.iter_mut().enumerate() {
        *t = (i as f64 * std::f64::consts::TAU / 16384.0).sin() as f32;
    }
    if r.remaining() != 0 {
        return Err(format!("{} trailing bytes in the static dump", r.remaining()));
    }
    Ok(sd)
}

fn idx<I: Io>(io: &mut I, v: &mut u32) {
    io.u32(v);
}

fn byte<I: Io>(io: &mut I, v: &mut u32) {
    let mut b = *v as u8;
    io.u8(&mut b);
    *v = u32::from(b);
}

fn io_static<I: Io>(io: &mut I, sd: &mut StaticData) {
    // WRLD
    io.tag(b"WRLD");
    byte(io, &mut sd.game_mode);
    io.flag(&mut sd.no_ball_rot);
    idx(io, &mut sd.ball_body);
    io.vec3(&mut sd.gravity_uu);
    for f in &mut sd.mut_f {
        io.f32(f);
    }
    io.flag(&mut sd.unlimited_flips);
    io.flag(&mut sd.unlimited_double_jumps);
    io.flag(&mut sd.recharge_boost_enabled);
    io.f32(&mut sd.recharge_boost_per_second);
    io.f32(&mut sd.recharge_boost_delay);
    byte(io, &mut sd.demo_mode);
    io.flag(&mut sd.enable_team_demos);
    io.f32(&mut sd.goal_base_threshold_y);
    io.flag(&mut sd.boost_pads_enabled);
    io.vec3(&mut sd.gravity_bt);
    let n = io.count(NUM_DYN);
    if !expect(io, n, NUM_DYN, "dynamic bodies") {
        return;
    }
    for d in &mut sd.dyn_body {
        idx(io, d);
    }
    io.vec3(&mut sd.grid_min_pos);
    io.vec3(&mut sd.grid_max_pos);
    io.f32(&mut sd.grid_cell_size);
    io.f32(&mut sd.grid_cell_size_sq);
    for v in &mut sd.grid_num_cells {
        idx(io, v);
    }
    io.vec3(&mut sd.sap_world_aabb_min);
    io.vec3(&mut sd.sap_quantize);
    for v in [&mut sd.sap_handle_mask, &mut sd.sap_sentinel, &mut sd.sap_max_handles] {
        let mut x = *v as u16;
        io.u16(&mut x);
        *v = u32::from(x);
    }
    io.f32(&mut sd.sap_world_half_extent);

    // BODY
    io.tag(b"BODY");
    let n = io.count(NUM_BODIES);
    if !expect(io, n, NUM_BODIES, "bodies") {
        return;
    }
    let mut mesh_tris = [0u32; NUM_BODIES];
    for (i, b) in sd.bodies.iter_mut().enumerate() {
        idx(io, &mut b.world_array_idx);
        byte(io, &mut b.user_idx);
        idx(io, &mut b.user_pointer);
        idx(io, &mut b.broadphase_handle);
        byte(io, &mut b.filter_group);
        byte(io, &mut b.filter_mask);
        io.f32(&mut b.mass);
        io.f32(&mut b.inv_mass);
        io.vec3(&mut b.inv_inertia_local);
        io.vec3(&mut b.inv_mass_splat);
        io.f32(&mut b.friction);
        io.f32(&mut b.restitution);
        io.f32(&mut b.contact_processing_threshold);
        io.f32(&mut b.contact_breaking_threshold);
        io.f32(&mut b.linear_damping);
        io.f32(&mut b.linear_sleeping_threshold);
        io.f32(&mut b.angular_sleeping_threshold);
        io.aff3(&mut b.pose);
        byte(io, &mut b.shape_kind);
        match b.shape_kind {
            0 => {
                idx(io, &mut b.mesh_tri_count);
                mesh_tris[i] = b.mesh_tri_count;
            }
            1 => {}
            2 => {
                io.vec3(&mut b.sphere_implicit_dim);
                io.f32(&mut b.sphere_margin);
            }
            3 => {
                io.vec3(&mut b.box_half_extents);
                io.f32(&mut b.box_margin);
                io.aff3(&mut b.child_trans);
                io.aabb(&mut b.compound_local_aabb);
            }
            _ => {
                io.fail("unknown shape kind");
                return;
            }
        }
    }
    for (d, &bi) in sd.dyn_body.clone().iter().enumerate() {
        if bi as usize >= NUM_BODIES {
            io.fail("dynamic body index out of range");
            return;
        }
        sd.dyn_handle[d] = sd.bodies[bi as usize].broadphase_handle;
    }

    // MESH
    io.tag(b"MESH");
    let n = io.count(NUM_MESHES);
    if !expect(io, n, NUM_MESHES, "meshes") {
        return;
    }
    let (mut tf, mut nf, mut lf) = (0usize, 0usize, 0usize);
    for m in 0..NUM_MESHES {
        let mi = &mut sd.meshes[m];
        idx(io, &mut mi.body);
        io.aabb(&mut mi.aabb_ident_cache);
        let nt = io.count(0);
        if tf + nt > MAX_TRIS {
            io.fail("triangles over the StaticData size");
            return;
        }
        mi.tri_first = tf as u32;
        mi.tri_count = nt as u32;
        for t in &mut sd.tris[tf..tf + nt] {
            io.vec3(&mut t.p0);
            io.vec3(&mut t.p1);
            io.vec3(&mut t.p2);
            io.vec3(&mut t.normal);
        }
        io.aabb(&mut sd.meshes[m].root_aabb);
        let nn = io.count(0);
        if nf + nn > MAX_NODES {
            io.fail("BVH nodes over the StaticData size");
            return;
        }
        sd.meshes[m].node_first = nf as u32;
        sd.meshes[m].node_count = nn as u32;
        for node in &mut sd.nodes[nf..nf + nn] {
            for v in [&mut node.min_x, &mut node.min_y, &mut node.min_z, &mut node.max_x, &mut node.max_y, &mut node.max_z] {
                for l in v.iter_mut() {
                    io.f32(l);
                }
            }
            for ch in &mut node.children {
                io.u32(ch);
            }
            byte(io, &mut node.child_count);
            io.u32(&mut node.valid_mask);
        }
        let nl = io.count(0);
        if lf + nl > MAX_LEAVES {
            io.fail("BVH leaves over the StaticData size");
            return;
        }
        sd.meshes[m].leaf_first = lf as u32;
        sd.meshes[m].leaf_count = nl as u32;
        for leaf in &mut sd.leaves[lf..lf + nl] {
            io.aabb(&mut leaf.bounds);
            idx(io, &mut leaf.leaf_idx);
        }
        let ni = io.count(0);
        if ni != nt {
            io.fail("one triangle info per triangle expected");
            return;
        }
        for info in &mut sd.infos[tf..tf + ni] {
            byte(io, &mut info.flags);
            io.f32(&mut info.edge_v0_v1_angle);
            io.f32(&mut info.edge_v1_v2_angle);
            io.f32(&mut info.edge_v2_v0_angle);
        }
        tf += nt;
        nf += nn;
        lf += nl;
    }
    if (tf, nf, lf) != (MAX_TRIS, MAX_NODES, MAX_LEAVES) {
        io.fail("mesh totals differ from StaticData sizes");
        return;
    }
    for m in &sd.meshes {
        if m.tri_count != mesh_tris[m.body as usize] {
            io.fail("mesh triangle count differs from its body record");
            return;
        }
    }

    // PLAN
    io.tag(b"PLAN");
    let n = io.count(NUM_PLANES);
    if !expect(io, n, NUM_PLANES, "planes") {
        return;
    }
    for p in &mut sd.planes {
        idx(io, &mut p.body);
        io.vec3(&mut p.plane_normal);
        io.flag(&mut p.is_single_axis);
        idx(io, &mut p.single_axis_idx);
        io.flag(&mut p.single_axis_backwards);
        io.aabb(&mut p.aabb_ident_cache);
        io.aabb(&mut p.aabb_cache);
    }

    // CELL: static ray-cell lists, indexed by cell
    io.tag(b"CELL");
    let m = io.count(0);
    let mut first = 0usize;
    for _ in 0..m {
        let (mut cell, mut cnt) = (0u32, 0u32);
        idx(io, &mut cell);
        idx(io, &mut cnt);
        if cell as usize >= NUM_GRID_CELLS || first + cnt as usize > MAX_CELL_ENTRIES {
            io.fail("static ray cells over the StaticData size");
            return;
        }
        sd.cells[cell as usize] = CellRange { first: first as u32, count: cnt };
        for h in &mut sd.cell_handles[first..first + cnt as usize] {
            idx(io, h);
        }
        first += cnt as usize;
    }
    if first != MAX_CELL_ENTRIES {
        io.fail("static ray-cell entries differ from StaticData size");
        return;
    }

    // CARS
    io.tag(b"CARS");
    let n = io.count(NUM_CARS);
    if !expect(io, n, NUM_CARS, "cars") {
        return;
    }
    for c in &mut sd.cars {
        idx(io, &mut c.car_idx);
        byte(io, &mut c.team);
        idx(io, &mut c.rigid_body_idx);
        io.flag(&mut c.three_wheels);
        io.vec3(&mut c.hitbox_size);
        io.vec3(&mut c.hitbox_pos_offset);
        for w in &mut c.wheels {
            io.vec3(&mut w.chassis_connection_point_cs);
            io.f32(&mut w.suspension_rest_length_1);
            io.f32(&mut w.wheels_radius);
            io.f32(&mut w.suspension_force_scale);
            io.f32(&mut w.real_ray_length);
        }
    }

    // PADS
    io.tag(b"PADS");
    let n = io.count(NUM_PADS);
    if !expect(io, n, NUM_PADS, "pads") {
        return;
    }
    io.f32(&mut sd.max_pad_z);
    for p in &mut sd.pads {
        io.vec3(&mut p.pos);
        io.flag(&mut p.is_big);
        io.f32(&mut p.box_radius);
        io.f32(&mut p.cyl_radius);
        io.f32(&mut p.max_cooldown);
        io.f32(&mut p.boost_amount);
        io.aabb(&mut p.aabb);
    }

    // PROX
    io.tag(b"PROX");
    let n = io.count(NUM_HANDLES);
    if !expect(io, n, NUM_HANDLES, "proxies") {
        return;
    }
    for p in &mut sd.proxies {
        idx(io, &mut p.client_obj_idx);
        byte(io, &mut p.filter_group);
        byte(io, &mut p.filter_mask);
        idx(io, &mut p.unique_id);
        io.aabb(&mut p.aabb);
        idx(io, &mut p.cell_idx);
        for i in &mut p.indices {
            idx(io, i);
        }
    }
}
