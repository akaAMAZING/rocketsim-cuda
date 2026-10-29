//! `#[repr(C)]` mirrors of `cuda/sim/arena_state.cuh` (`ArenaState`, `StaticData`), written by hand and checked field by
//! field against the C++ host shim (`tests/layout.rs`: offset and size of every field, size of every struct).
//!
//! Every struct is an array of 4-byte words with alignment 4 and no padding, so an `ArenaState` is `ARENA_WORDS` u32
//! words. Each type also describes its words (`Mirror::leaves`): which field a word belongs to and whether the field is
//! discrete or a float, which is what the comparator (`compare.rs`) needs.

use core::mem::{offset_of, size_of};

// ---- sizes (arena_state.cuh) ----------------------------------------------------------------------------------------
pub const NUM_BODIES: usize = 23;
pub const NUM_DYN: usize = 3;
pub const NUM_CARS: usize = 2;
pub const NUM_WHEELS: usize = 4;
pub const NUM_PADS: usize = 34;
pub const NUM_HANDLES: usize = 24;
pub const NUM_EDGES: usize = 2 * NUM_HANDLES;
pub const NUM_MESHES: usize = 16;
pub const NUM_PLANES: usize = 4;
pub const MAX_MANIFOLDS: usize = 16;
pub const MAX_POINTS: usize = 4;
pub const MAX_ACTIVE: usize = 16;
pub const MAX_RECORDS: usize = 16;
pub const MAX_PAIRS: usize = 16;
pub const MAX_DYN_CELLS: usize = 64;
pub const OVF_COUNT: usize = 6;
/// Names of the overflow counters (`enum Overflow`).
pub const OVF_NAMES: [&str; OVF_COUNT] = ["manifolds", "active", "records", "pairs", "dyn_cells", "pad_touching"];
pub const MAX_TRIS: usize = 8020;
pub const MAX_NODES: usize = 3754;
pub const MAX_LEAVES: usize = 8020;
pub const TABLE_SIN_LEN: usize = 16384;
pub const NUM_GRID_CELLS: usize = 242;
pub const MAX_CELL_ENTRIES: usize = 1210;
pub const NUM_MUT: usize = 20;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Float,
    Int,
}

/// A type made of 4-byte words that can say what each word is.
pub trait Mirror: Copy {
    /// Appends one `(group, kind)` per 4-byte word, in memory order. `name` is the dotted path of the field.
    fn leaves(name: &str, out: &mut Vec<(String, Kind)>);
}

impl Mirror for f32 {
    fn leaves(name: &str, out: &mut Vec<(String, Kind)>) {
        out.push((name.to_string(), Kind::Float));
    }
}
impl Mirror for u32 {
    fn leaves(name: &str, out: &mut Vec<(String, Kind)>) {
        out.push((name.to_string(), Kind::Int));
    }
}
impl<T: Mirror, const N: usize> Mirror for [T; N] {
    fn leaves(name: &str, out: &mut Vec<(String, Kind)>) {
        for _ in 0..N {
            T::leaves(name, out);
        }
    }
}

/// A struct whose words all belong to one named field (a vector, a matrix, a pair of u32).
macro_rules! leaf_struct {
    ($(#[$m:meta])* $name:ident, $kind:expr, { $($f:ident : $t:ty),* }) => {
        $(#[$m])*
        #[repr(C)]
        #[derive(Clone, Copy, Default, PartialEq, Debug)]
        pub struct $name { $(pub $f: $t),* }
        impl $name {
            pub const FIELDS: &'static [(&'static str, usize, usize)] =
                &[$((stringify!($f), offset_of!($name, $f), size_of::<$t>())),*];
        }
        impl Mirror for $name {
            fn leaves(name: &str, out: &mut Vec<(String, Kind)>) {
                for _ in 0..size_of::<$name>() / 4 {
                    out.push((name.to_string(), $kind));
                }
            }
        }
    };
}

/// A struct of fields, each a `Mirror`.
macro_rules! mirror {
    ($(#[$m:meta])* $name:ident { $($(#[$fm:meta])* $f:ident : $t:ty),* $(,)? }) => {
        $(#[$m])*
        #[repr(C)]
        #[derive(Clone, Copy)]
        pub struct $name { $($(#[$fm])* pub $f: $t),* }
        impl $name {
            pub const FIELDS: &'static [(&'static str, usize, usize)] =
                &[$((stringify!($f), offset_of!($name, $f), size_of::<$t>())),*];
        }
        impl Mirror for $name {
            fn leaves(name: &str, out: &mut Vec<(String, Kind)>) {
                $({
                    let n = if name.is_empty() { stringify!($f).to_string() } else { format!("{name}.{}", stringify!($f)) };
                    <$t as Mirror>::leaves(&n, out);
                })*
            }
        }
    };
}

leaf_struct!(Vec3, Kind::Float, { x: f32, y: f32, z: f32 });
leaf_struct!(Quat, Kind::Float, { x: f32, y: f32, z: f32, w: f32 });
leaf_struct!(Mat3, Kind::Float, { x_axis: Vec3, y_axis: Vec3, z_axis: Vec3 });
leaf_struct!(Affine3, Kind::Float, { matrix3: Mat3, translation: Vec3 });
leaf_struct!(Aabb, Kind::Float, { min: Vec3, max: Vec3 });
leaf_struct!(U64, Kind::Int, { lo: u32, hi: u32 });

impl U64 {
    pub fn get(self) -> u64 {
        (u64::from(self.hi) << 32) | u64::from(self.lo)
    }
    pub fn set(v: u64) -> Self {
        Self { lo: v as u32, hi: (v >> 32) as u32 }
    }
}

/// `Opt<T>` of the header: `some` is 0 or 1, `v` is all zero when absent.
#[repr(C)]
#[derive(Clone, Copy, Default, PartialEq, Debug)]
pub struct Opt<T: Copy> {
    pub some: u32,
    pub v: T,
}
impl<T: Mirror> Mirror for Opt<T> {
    fn leaves(name: &str, out: &mut Vec<(String, Kind)>) {
        out.push((format!("{name}#some"), Kind::Int));
        T::leaves(name, out);
    }
}
impl<T: Copy> Opt<T> {
    pub fn get(self) -> Option<T> {
        (self.some != 0).then_some(self.v)
    }
}

/// `uint16_t pos, handle` (axis_sweep.rs:20-21): one discrete word.
#[repr(C)]
#[derive(Clone, Copy, Default, PartialEq, Debug)]
pub struct Edge {
    pub pos: u16,
    pub handle: u16,
}
impl Edge {
    pub const FIELDS: &'static [(&'static str, usize, usize)] =
        &[("pos", offset_of!(Edge, pos), 2), ("handle", offset_of!(Edge, handle), 2)];
}
impl Mirror for Edge {
    fn leaves(name: &str, out: &mut Vec<(String, Kind)>) {
        out.push((name.to_string(), Kind::Int));
    }
}

/// `uint16_t min[3], max[3]` (axis_sweep.rs:33-34): three discrete words.
#[repr(C)]
#[derive(Clone, Copy, Default, PartialEq, Debug)]
pub struct HandleEdges {
    pub min: [u16; 3],
    pub max: [u16; 3],
}
impl HandleEdges {
    pub const FIELDS: &'static [(&'static str, usize, usize)] =
        &[("min", offset_of!(HandleEdges, min), 6), ("max", offset_of!(HandleEdges, max), 6)];
}
impl Mirror for HandleEdges {
    fn leaves(name: &str, out: &mut Vec<(String, Kind)>) {
        for _ in 0..3 {
            out.push((name.to_string(), Kind::Int));
        }
    }
}

// ---- ArenaState ---------------------------------------------------------------------------------------------------------
mirror!(Controls { throttle: f32, steer: f32, pitch: f32, yaw: f32, roll: f32, jump: u32, boost: u32, handbrake: u32 });
mirror!(HitInfo { hit_point: Vec3, hit_normal: Vec3, hit_fraction: f32, user_info: u32 });
mirror!(RaycastInfo {
    contact_normal: Vec3,
    contact_point: Vec3,
    ground_body_idx: u32,
    suspension_length: f32,
    impulse: Vec3,
    ground_stick: Opt<Vec3>,
    is_in_contact_with_world: u32,
    clipped_inv_contact_dot_suspension: f32,
    suspension_relative_vel: f32,
});
mirror!(Wheel {
    raycast_info: Opt<RaycastInfo>,
    hard_point: Vec3,
    axle_dir: Vec3,
    engine_force: f32,
    brake: f32,
    steer_angle: f32,
    vel_at_contact_point: Vec3,
    lat_friction: f32,
    long_friction: f32,
    extra_pushback: f32,
});
mirror!(Car {
    dodge_deadzone: f32,
    vel_impulse_cache: Vec3,
    sticky_gate_prev: u32,
    pos: Vec3,
    rot_mat: Mat3,
    vel: Vec3,
    ang_vel: Vec3,
    controls: Controls,
    has_jumped: u32,
    has_double_jumped: u32,
    has_flipped: u32,
    flip_rel_torque: Vec3,
    jump_ticks: u32,
    flip_time: f32,
    is_flipping: u32,
    is_jumping: u32,
    jump_hold_broken: u32,
    air_time: f32,
    air_time_since_jump: f32,
    boost: f32,
    handbrake_val: f32,
    is_supersonic: u32,
    supersonic_grace_timer: f32,
    time_since_boosted: f32,
    boosting_time: f32,
    is_boosting: u32,
    is_auto_flipping: u32,
    auto_flip_timer: f32,
    auto_flip_torque_scale: f32,
    bump_cooldown_timer: f32,
    bump_cooldown_victim: Opt<u32>,
    last_extra_hit_tick: Opt<U64>,
    is_demoed: u32,
    demo_respawn_timer: f32,
    is_on_ground: u32,
    wheels_with_contact: [Opt<HitInfo>; NUM_WHEELS],
    world_contact_normal: Opt<Vec3>,
    prev_controls: Controls,
    wheels: [Wheel; NUM_WHEELS],
});
mirror!(BallDyn { pos: Vec3, rot_mat: Mat3, vel: Vec3, ang_vel: Vec3, tick_count_since_kickoff: U64 });
mirror!(Body {
    collision_flags: u32,
    activation: u32,
    deactivation_time: f32,
    world_trans: Affine3,
    world_quat: Quat,
    interp_world_trans: Affine3,
    lin_vel: Vec3,
    ang_vel: Vec3,
    inv_inertia_tensor_world: Mat3,
    accum_lin_vel: Vec3,
    accum_ang_vel: Vec3,
    cached_linear_damping: f32,
    cached_damping_time_step: f32,
    cached_linear_damping_value: f32,
});
mirror!(ManifoldPoint {
    local_point_a: Vec3,
    local_point_b: Vec3,
    pos_world_on_b: Vec3,
    pos_world_on_a: Vec3,
    normal_world_on_b: Vec3,
    distance_1: f32,
    combined_friction: f32,
    combined_restitution: f32,
    is_special: u32,
});
mirror!(Manifold {
    body0_idx: u32,
    body1_idx: u32,
    point_count: u32,
    points: [ManifoldPoint; MAX_POINTS],
});
mirror!(RecordPoint { pos_world_on_b: Vec3, pos_world_on_a: Vec3, normal_world_on_b: Vec3 });
mirror!(ContactRecord { is_swap: u32, rb_idx_a: u32, rb_idx_b: u32, point: RecordPoint });
mirror!(DynProxy { aabb: Aabb, cell_idx: u32, indices: [u32; 3] });
mirror!(DynCell { cell: u32, handle: u32 });
mirror!(Pair { proxy0: u32, proxy1: u32 });

mirror!(PadTouching { n: u32, car_idx: [u32; NUM_CARS] });

mirror!(ArenaState {
    tick_count: U64,
    rng: U64,
    ball: BallDyn,
    cars: [Car; NUM_CARS],
    pad_gave_boost: [Opt<U64>; NUM_PADS],
    pad_touching: [PadTouching; NUM_PADS],
    bodies: [Body; NUM_DYN],
    n_manifolds: u32,
    manifolds: [Manifold; MAX_MANIFOLDS],
    n_dyn_cells: u32,
    dyn_cells: [DynCell; MAX_DYN_CELLS],
    dyn_proxy: [DynProxy; NUM_DYN],
    handle_edges: [HandleEdges; NUM_HANDLES],
    edges: [[Edge; NUM_EDGES]; 3],
    n_pairs: u32,
    pairs: [Pair; MAX_PAIRS],
    overflow: [u32; OVF_COUNT],
    // the last tick's V3 events as EventBit bits (cuda/sim/arena_state.cuh); an output, cleared by P0
    events: u32,
    // live between phases only: zero at a tick boundary
    n_active: u32,
    active: [u32; MAX_ACTIVE],
    // the folded ball-world and car-world records (cuda/sim/arena_state.cuh)
    fold_events: u32,
    fold_car_normal: [Vec3; NUM_CARS],
    n_records: u32,
    records: [ContactRecord; MAX_RECORDS],
});

pub const ARENA_WORDS: usize = size_of::<ArenaState>() / 4;
/// Words an ArenaState holds at a tick boundary; the rest (`n_active` on) is the mid-tick part, zero between ticks.
pub const TICK_WORDS: usize = offset_of!(ArenaState, n_active) / 4;
const _: () = assert!(size_of::<ArenaState>() % 4 == 0 && core::mem::align_of::<ArenaState>() == 4);

// ---- StaticData ---------------------------------------------------------------------------------------------------------
mirror!(MeshInfo {
    body: u32,
    tri_first: u32,
    tri_count: u32,
    node_first: u32,
    node_count: u32,
    leaf_first: u32,
    leaf_count: u32,
    aabb_ident_cache: Aabb,
    root_aabb: Aabb,
});
mirror!(Tri { p0: Vec3, p1: Vec3, p2: Vec3, normal: Vec3 });
mirror!(WideNode {
    min_x: [f32; 4],
    min_y: [f32; 4],
    min_z: [f32; 4],
    max_x: [f32; 4],
    max_y: [f32; 4],
    max_z: [f32; 4],
    children: [u32; 4],
    child_count: u32,
    valid_mask: u32,
});
mirror!(Leaf { bounds: Aabb, leaf_idx: u32 });
mirror!(TriInfo { flags: u32, edge_v0_v1_angle: f32, edge_v1_v2_angle: f32, edge_v2_v0_angle: f32 });
mirror!(Plane {
    body: u32,
    plane_normal: Vec3,
    is_single_axis: u32,
    single_axis_idx: u32,
    single_axis_backwards: u32,
    aabb_ident_cache: Aabb,
    aabb_cache: Aabb,
});
mirror!(BodyConst {
    world_array_idx: u32,
    user_idx: u32,
    user_pointer: u32,
    broadphase_handle: u32,
    filter_group: u32,
    filter_mask: u32,
    mass: f32,
    inv_mass: f32,
    inv_inertia_local: Vec3,
    inv_mass_splat: Vec3,
    friction: f32,
    restitution: f32,
    contact_processing_threshold: f32,
    contact_breaking_threshold: f32,
    linear_damping: f32,
    linear_sleeping_threshold: f32,
    angular_sleeping_threshold: f32,
    pose: Affine3,
    shape_kind: u32,
    sphere_implicit_dim: Vec3,
    sphere_margin: f32,
    box_half_extents: Vec3,
    box_margin: f32,
    child_trans: Affine3,
    compound_local_aabb: Aabb,
    mesh_tri_count: u32,
});
mirror!(WheelConst {
    chassis_connection_point_cs: Vec3,
    suspension_rest_length_1: f32,
    wheels_radius: f32,
    suspension_force_scale: f32,
    real_ray_length: f32,
});
mirror!(CarConst {
    car_idx: u32,
    team: u32,
    rigid_body_idx: u32,
    three_wheels: u32,
    hitbox_size: Vec3,
    hitbox_pos_offset: Vec3,
    wheels: [WheelConst; NUM_WHEELS],
});
mirror!(PadConst { pos: Vec3, is_big: u32, box_radius: f32, cyl_radius: f32, max_cooldown: f32, boost_amount: f32, aabb: Aabb });
mirror!(ProxyConst {
    client_obj_idx: u32,
    filter_group: u32,
    filter_mask: u32,
    unique_id: u32,
    aabb: Aabb,
    cell_idx: u32,
    indices: [u32; 3],
});
mirror!(CellRange { first: u32, count: u32 });

mirror!(StaticData {
    game_mode: u32,
    no_ball_rot: u32,
    ball_body: u32,
    gravity_uu: Vec3,
    mut_f: [f32; NUM_MUT],
    unlimited_flips: u32,
    unlimited_double_jumps: u32,
    recharge_boost_enabled: u32,
    recharge_boost_per_second: f32,
    recharge_boost_delay: f32,
    demo_mode: u32,
    enable_team_demos: u32,
    goal_base_threshold_y: f32,
    boost_pads_enabled: u32,
    gravity_bt: Vec3,
    dyn_body: [u32; NUM_DYN],
    grid_min_pos: Vec3,
    grid_max_pos: Vec3,
    grid_cell_size: f32,
    grid_cell_size_sq: f32,
    grid_num_cells: [u32; 3],
    sap_world_aabb_min: Vec3,
    sap_quantize: Vec3,
    sap_handle_mask: u32,
    sap_sentinel: u32,
    sap_max_handles: u32,
    sap_world_half_extent: f32,
    max_pad_z: f32,
    dyn_handle: [u32; NUM_DYN],
    bodies: [BodyConst; NUM_BODIES],
    planes: [Plane; NUM_PLANES],
    meshes: [MeshInfo; NUM_MESHES],
    cars: [CarConst; NUM_CARS],
    pads: [PadConst; NUM_PADS],
    proxies: [ProxyConst; NUM_HANDLES],
    cells: [CellRange; NUM_GRID_CELLS],
    cell_handles: [u32; MAX_CELL_ENTRIES],
    infos: [TriInfo; MAX_TRIS],
    tris: [Tri; MAX_TRIS],
    nodes: [WideNode; MAX_NODES],
    leaves: [Leaf; MAX_LEAVES],
    table_sin: [f32; TABLE_SIN_LEN],
});

pub const STATIC_WORDS: usize = size_of::<StaticData>() / 4;
const _: () = assert!(size_of::<StaticData>() % 4 == 0 && core::mem::align_of::<StaticData>() == 4);

/// An all-zero value on the heap (every field of these structs is valid as zero).
pub fn zeroed<T: Copy>() -> Box<T> {
    // SAFETY: only used for the plain-word structs of this module, for which zero bytes are a valid value.
    unsafe { Box::<T>::new_zeroed().assume_init() }
}

impl ArenaState {
    pub fn zeroed() -> Box<Self> {
        zeroed()
    }
    /// The state as its 4-byte words.
    pub fn words(&self) -> &[u32] {
        // SAFETY: repr(C), size a multiple of 4, alignment 4, no padding, every word initialised (plain data).
        unsafe { core::slice::from_raw_parts((self as *const Self).cast::<u32>(), ARENA_WORDS) }
    }
    pub fn words_mut(&mut self) -> &mut [u32] {
        // SAFETY: as `words`; every bit pattern is a valid value of these fields.
        unsafe { core::slice::from_raw_parts_mut((self as *mut Self).cast::<u32>(), ARENA_WORDS) }
    }
    pub fn as_bytes(&self) -> &[u8] {
        // SAFETY: as `words`.
        unsafe { core::slice::from_raw_parts((self as *const Self).cast::<u8>(), size_of::<Self>()) }
    }
}

impl StaticData {
    pub fn zeroed() -> Box<Self> {
        zeroed()
    }
    pub fn as_bytes(&self) -> &[u8] {
        // SAFETY: as `ArenaState::words`.
        unsafe { core::slice::from_raw_parts((self as *const Self).cast::<u8>(), size_of::<Self>()) }
    }
}

/// The group and kind of every word of an `ArenaState`, built once.
pub fn arena_leaves() -> Vec<(String, Kind)> {
    let mut v = Vec::with_capacity(ARENA_WORDS);
    ArenaState::leaves("", &mut v);
    assert_eq!(v.len(), ARENA_WORDS, "a field of ArenaState does not describe all its words");
    v
}
