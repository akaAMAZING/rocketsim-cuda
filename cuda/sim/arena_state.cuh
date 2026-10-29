// ArenaState and StaticData: the plain-struct state of one Soccar 1v1 V3 arena.
//
// ArenaState holds every field V3 needs between ticks and at every phase boundary P0..P8 (as the dump of the patched V3 exports it),
// except what the tests proved unnecessary or constant (list at the end of this comment). StaticData holds what never
// changes after the arena is built: V3's BVH exactly as V3 built it, triangles, triangle_info_map, planes, body
// constants, ray-grid geometry with its static cells, wheel constants, pads. It is uploaded once.
//
// LAYOUT RULES (the Rust mirror in crates/rocketsim-cuda/src/layout.rs repeats them and a test compares every offset):
//   * every member is a 4-byte word or a struct of them: no padding, alignment 4, so an ArenaState is an array of u32
//     words and the AoSoA-32 copy-in/out kernels move it word by word;
//   * bool and u8 fields are u32 (0 or 1); Option<T> is Opt<T> = {u32 some; T v} with v all zero when absent; u64 is U64 {lo, hi}; indices are u32, usize::MAX is 0xFFFFFFFF;
//   * a V3 list is a count plus a fixed array (the caps below, each with margin over the measured maximum), and an
//     overflow counter per list that the producing code must bump instead of writing past the cap.
// Structure: the manifold pool is 16 slots with a count; V3's 1,024-entry
// manifold_table and the triangular pair_indices table are searches (no field); the island manager's sort cache and
// scratch are dropped (a memo).
//
// NOT IN ArenaState (each shown unneeded by the dump round-trip tests, or constant in Soccar 1v1):
//   pair_key (= min << 32 | max of the body indices), the manifolds' two
//   contact thresholds (the min over their two bodies' thresholds), applied_impulse (always 0: V3 never writes solver
//   impulses back), the island manager, the manifold table and the pair index table, free_list (always empty), the
//   Heatseeker, Dropshot and Snowday fields (other modes; the importer asserts default values), ball-world contact
//   records and ball_only (ball-only arenas; the importer asserts none), companion_id (the solver-body link P5's setup
//   reassigns before any read), the contact records' local points, distance, friction, restitution and is_special (P7 reads
//   only the two world points and the normal), the static bodies' dynamic fields (a static body never moves: StaticData
//   holds its pose, the writer's Template its record).
// The test behind this: the dump rebuilt from an ArenaState alone finishes the tick bit-identically at all
// 10.6M phase boundaries of the 90 scenarios; zeroing any field that IS in ArenaState there changes the result (except the redundant
// ball copies ball.rot_mat and ball.ang_vel, which V3 rewrites from the body at P8, and the lists whose zeroing panics V3), while zeroing
// companion_id or the dropped record fields changes nothing.
//
// Every field names its V3 source (paths relative to rocketsim/src/ of rev ad8b38c) in the comment beside it.
#pragma once
#include <stddef.h>
#include <stdint.h>
#include "common/glam.cuh"

namespace lfsim {

// ---- sizes of the Soccar 1v1 arena (V3: sim/arena/base.rs setup_arena_collision_shapes) --------------------
constexpr int NUM_BODIES = 23;      // 16 meshes, 4 planes, ball, 2 cars: collision_objs order
constexpr int NUM_DYN = 3;          // dynamic bodies: ball (body 20), car 0 (21), car 1 (22): dynamic_body_idcs
constexpr int NUM_CARS = 2;
constexpr int NUM_WHEELS = 4;
constexpr int NUM_PADS = 34;        // boost_pad_grid.rs: 28 small + 6 big
constexpr int NUM_HANDLES = 24;     // axis_sweep.rs handles: sentinel + 23 bodies
constexpr int NUM_EDGES = 2 * NUM_HANDLES;  // edges per axis: 2 per handle, the sentinel's pair included
constexpr int NUM_MESHES = 16;
constexpr int NUM_PLANES = 4;

// ---- caps (measured maximum over 40.3M ticks plus the phase harness, with margin) ---------------------
constexpr int MAX_MANIFOLDS = 16;   // retained manifolds: measured 10
constexpr int MAX_POINTS = 4;       // MANIFOLD_CACHE_SIZE (persistent_manifold.rs)
constexpr int MAX_ACTIVE = 16;      // manifolds in a solve: measured 8 at a boundary, 10 retained
constexpr int MAX_RECORDS = 16;     // stored contact records, car-ball and car-car only (the ball-world and car-world ones are folded
                                    // into fold_events). Measured max 4 over the test scenarios; expected structural bound 12 (2 car-ball
                                    // pairs and 1 car-car pair, 4 points each); the overflow counter guards it
constexpr int MAX_PAIRS = 16;       // retained broadphase pairs: measured 10
constexpr int MAX_DYN_CELLS = 64;   // dynamic ray-cell entries: measured 54

enum Overflow { OVF_MANIFOLDS, OVF_ACTIVE, OVF_RECORDS, OVF_PAIRS, OVF_DYN_CELLS, OVF_PAD_TOUCHING, OVF_COUNT };

// ArenaState.events: the tick's V3 events (arena/base.rs:89 `events`, pushed at :658, :749, :1536, :1565, :1573, :1724) as
// bits; per car, bit EV_x + car index (the victim's index for CarHitCar).
enum EventBit : uint32_t {
  EV_CAR_HIT_BALL = 0,     // CarHitBall
  EV_BUMP = 2,             // CarHitCar, is_demo false
  EV_DEMO = 4,             // CarHitCar, is_demo true
  EV_PICKUP = 6,           // CarPickupBoost
  EV_LANDED = 8,           // CarLanded
  EV_CAR_HIT_WORLD = 10,   // CarHitWorld
  EV_BALL_HIT_WORLD = 12,  // BallHitWorld, any
  // bit 13 is TickEvents::code's goal (is_ball_scored after the tick): not an event, not set here
  EV_BALL_WALL = 14,       // BallHitWorld with |contact_normal.z| < 0.8 (wall, ceiling or ramp: TickEvents.ball_wall)
};

struct U64 { uint32_t lo, hi; };
template <class T> struct Opt { uint32_t some; T v; };

LF_HDI uint64_t get_u64(U64 x) { return ((uint64_t)x.hi << 32) | x.lo; }
LF_HDI U64 make_u64(uint64_t v) { return U64{(uint32_t)v, (uint32_t)(v >> 32)}; }

struct Aabb { Vec3A min, max; };  // shared/aabb.rs

// ---- cars (V3: sim/car/base.rs:43-49, car_state.rs:305-362, wheel_info.rs:39-54) --------------------------------------
struct Controls {                                   // car_controls.rs:12-19
  float throttle, steer, pitch, yaw, roll;
  uint32_t jump, boost, handbrake;
};
struct HitInfo {                                    // user_raycast.rs:20-29 RaycastHitInfo
  Vec3A hit_point, hit_normal;                      // uu
  float hit_fraction;
  uint32_t user_info;                               // 0 none, 1 car, 2 ball, 3 dropshot tile
};
struct RaycastInfo {                                // wheel_info.rs:16-26
  Vec3A contact_normal, contact_point;              // BT
  uint32_t ground_body_idx;
  float suspension_length;
  Vec3A impulse;
  Opt<Vec3A> ground_stick;
  uint32_t is_in_contact_with_world;
  float clipped_inv_contact_dot_suspension;
  float suspension_relative_vel;
};
struct Wheel {                                      // wheel_info.rs:39-54, dynamic fields; order FL FR BL BR
  Opt<RaycastInfo> raycast_info;                    // :40
  Vec3A hard_point, axle_dir;                       // :41-42
  float engine_force, brake, steer_angle;           // :46-48
  Vec3A vel_at_contact_point;                       // :49
  float lat_friction, long_friction;                // :50-51
  float extra_pushback;                             // :53
};
struct Car {
  float dodge_deadzone;                             // car_body_config.rs:87 (state: set_car_dodge_deadzone, base.rs:1029)
  Vec3A vel_impulse_cache;                          // car/base.rs:47, BT/s
  uint32_t sticky_gate_prev;                        // car/base.rs:49
  // CarState (car_state.rs:305-362)
  Vec3A pos;                                        // :306 phys.pos (uu)
  Mat3A rot_mat;                                    // :306 phys.rot_mat
  Vec3A vel, ang_vel;                               // :306 (uu/s, rad/s)
  Controls controls;                                // :308
  uint32_t has_jumped, has_double_jumped, has_flipped;  // :310-314
  Vec3A flip_rel_torque;                            // :316
  uint32_t jump_ticks;                              // :318
  float flip_time;                                  // :320
  uint32_t is_flipping, is_jumping, jump_hold_broken;   // :322-326
  float air_time, air_time_since_jump, boost, handbrake_val;  // :328-334
  uint32_t is_supersonic;                           // :336
  float supersonic_grace_timer, time_since_boosted, boosting_time;  // :338-342
  uint32_t is_boosting, is_auto_flipping;           // :344-346
  float auto_flip_timer, auto_flip_torque_scale, bump_cooldown_timer;  // :348-352
  Opt<uint32_t> bump_cooldown_victim;               // :354
  Opt<U64> last_extra_hit_tick;                     // :356
  uint32_t is_demoed;                               // :358
  float demo_respawn_timer;                         // :360
  uint32_t is_on_ground;                            // :361 cache.is_on_ground (car_state.rs:251)
  Opt<HitInfo> wheels_with_contact[NUM_WHEELS];     // :252
  Opt<Vec3A> world_contact_normal;                  // :253
  Controls prev_controls;                           // :254
  Wheel wheels[NUM_WHEELS];                         // car/base.rs:45, vehicle_rl.rs:25
};

struct PadTouching {                                // boost_pad/base.rs:14 touching_car_idxs (a car touches one pad at most)
  uint32_t n;
  uint32_t car_idx[NUM_CARS];                       // slots from n on are zero
};

struct BallDyn {                                    // sim/ball/base.rs:29-33, ball_state.rs:82-87
  Vec3A pos;                                        // phys_state.rs:16 (uu)
  Mat3A rot_mat;
  Vec3A vel, ang_vel;
  U64 tick_count_since_kickoff;                     // ball_state.rs:86
};

// ---- rigid bodies (V3: bullet/dynamics/rigid_body.rs:129-172), the three dynamic ones ----------------------------------
struct Body {
  uint32_t collision_flags;                         // :144 bit 3 (no contact response) set while a car is demolished
  uint32_t activation;                              // :133 0 active, 1 sleeping, 2 disabled, 3 wants deactivation
  float deactivation_time;                          // :134 seconds below the sleeping thresholds (only the ball can sleep)
  Affine3A world_trans;                             // :130 (BT)
  Quat world_quat;                                  // :131
  Affine3A interp_world_trans;                      // :136 predicted transform, live P2..P6
  Vec3A lin_vel, ang_vel;                           // :155-156 (BT/s, rad/s)
  Mat3A inv_inertia_tensor_world;                   // :154
  Vec3A accum_lin_vel, accum_ang_vel;               // :161-162, live P1..P6
  float cached_linear_damping;                      // :166 (1 - damping)^dt with the exporting platform's powf
  float cached_damping_time_step;                   // :167
  float cached_linear_damping_value;                // :168
};

// ---- narrowphase (V3: collision_dispatcher.rs:25-37, persistent_manifold.rs:45-52, manifold_point.rs:8-18) -------------
struct ManifoldPoint {
  Vec3A local_point_a, local_point_b, pos_world_on_b, pos_world_on_a, normal_world_on_b;  // BT
  float distance_1, combined_friction, combined_restitution;
  uint32_t is_special;
};
struct Manifold {
  uint32_t body0_idx, body1_idx;                    // persistent_manifold.rs:47-48 (pair_key = min << 32 | max; the two thresholds
                                                    // of :50-51 are the min over the two bodies, StaticData.bodies, :60-65)
  uint32_t point_count;                             // :46 point_cache.len()
  ManifoldPoint points[MAX_POINTS];                 // slots from point_count on are zero
};
struct RecordPoint { Vec3A pos_world_on_b, pos_world_on_a, normal_world_on_b; };  // the three fields P7 reads (manifold_point.rs:8-18)
struct ContactRecord {                              // arena_contact_tracker.rs:19-24
  uint32_t is_swap, rb_idx_a, rb_idx_b;
  RecordPoint point;
};

// ---- broadphase (V3: grid_broadphase.rs:173-177, axis_sweep.rs:41-51, broadphase_proxy.rs:74-82) -----------------------
struct DynProxy {                                   // the moving fields of the proxy of dynamic body d (handle StaticData.dyn_handle[d])
  Aabb aabb;                                        // swept box (BT)
  uint32_t cell_idx;
  uint32_t indices[3];
};
struct HandleEdges { uint16_t min[3], max[3]; };    // axis_sweep.rs:33-34
struct Edge { uint16_t pos, handle; };              // axis_sweep.rs:20-21
struct DynCell { uint32_t cell, handle; };          // GridCell.dyn_handles (grid_broadphase.rs:21): sorted by cell, list order within a cell
struct Pair { uint32_t proxy0, proxy1; };           // broadphase_proxy.rs:86-87, overlapping_pair_cache.rs:16 (dispatch order)

// ---- the arena ------------------------------------------------------------------------------------------------------
struct ArenaState {
  U64 tick_count;                                   // arena/base.rs:86
  U64 rng;                                          // base.rs:93 fastrand WyRand state
  BallDyn ball;
  Car cars[NUM_CARS];
  Opt<U64> pad_gave_boost[NUM_PADS];                // boost_pad/base.rs:13 gave_boost_tick_count (i64 >= 0)
  PadTouching pad_touching[NUM_PADS];               // boost_pad/base.rs:14 touching_car_idxs, in arrival order
  Body bodies[NUM_DYN];                             // dynamic bodies in dynamic_body_idcs order: ball, car 0, car 1
  uint32_t n_manifolds;                             // collision_dispatcher.rs:26 persistent_manifolds, pool order (swap_remove)
  Manifold manifolds[MAX_MANIFOLDS];
  uint32_t n_dyn_cells;
  DynCell dyn_cells[MAX_DYN_CELLS];
  DynProxy dyn_proxy[NUM_DYN];
  HandleEdges handle_edges[NUM_HANDLES];            // :43
  Edge edges[3][NUM_EDGES];                         // :44
  uint32_t n_pairs;
  Pair pairs[MAX_PAIRS];                            // overlapping_pair_cache.rs:16
  uint32_t overflow[OVF_COUNT];                     // incremented instead of writing past a cap; must stay 0
  uint32_t events;                                  // arena/base.rs:89 events, the last tick's, as EventBit bits: cleared by P0,
                                                    // set by P1, P7, P8; an output (no phase reads it; a V3 dump has none)
  // ---- live between phases only: all zero at a tick boundary, so a copy at a tick boundary stops before this line ----
  uint32_t n_active;                                // collision_dispatcher.rs:30 active_manifolds (P3b to P5)
  uint32_t active[MAX_ACTIVE];                      // pool indices in dispatch/sort order
  uint32_t fold_events;                             // folded ball-world and car-world records (P3b to P7): EventBit
                                                    // bits EV_BALL_HIT_WORLD, EV_BALL_WALL (|normal.z| < 0.8 in any record) and
                                                    // EV_CAR_HIT_WORLD + car; P7 ORs them into events
  Vec3A fold_car_normal[NUM_CARS];                  // the normal of the LAST car-world record of that car (zero unless its bit is set)
  uint32_t n_records;                               // arena_contact_tracker.rs:36 (P3b to P7): car-ball and car-car records only
  ContactRecord records[MAX_RECORDS];
};

constexpr int ARENA_WORDS = (int)(sizeof(ArenaState) / 4);
// Words an ArenaState holds at a tick boundary; everything after is the mid-tick part (zero between ticks).
constexpr int TICK_WORDS = (int)(offsetof(ArenaState, n_active) / 4);
static_assert(sizeof(ArenaState) % 4 == 0, "ArenaState is an array of u32 words");

// ---- StaticData -------------------------------------------------------------------------------------------------------
// Everything that never changes after the arena is built (the static dump of the patched V3). Sized for the
// 16 Soccar meshes; the importer refuses a dump whose counts differ.
constexpr int MAX_TRIS = 8020;          // static dump count 3
constexpr int MAX_NODES = 3754;         // count 4
constexpr int MAX_LEAVES = 8020;        // count 5 (one single-triangle leaf per triangle)
constexpr int TABLE_SIN_LEN = 16384;    // car/base.rs:101: the 14-bit angle table of the auto-flip roll rebuild
constexpr int NUM_GRID_CELLS = 242;     // count 10: the ray grid's cells (grid_broadphase.rs CellGrid)
constexpr int MAX_CELL_ENTRIES = 1210;  // count 11: entries over all static ray-cell lists
constexpr int NUM_MUT = 20;             // mutator floats

struct MeshInfo {                       // BvhTriangleMeshShape (bvh_triangle_mesh_shape.rs:17-22) and its Tree (shared/bvh.rs:17)
  uint32_t body, tri_first, tri_count, node_first, node_count, leaf_first, leaf_count;
  Aabb aabb_ident_cache;                // :21
  Aabb root_aabb;                       // Tree.aabb
};
struct Tri { Vec3A p0, p1, p2, normal; };                       // triangle_shape.rs:36-38, triangle_mesh.rs:5-7
struct WideNode {                       // bvh.rs:454-464; one lane per child; a children word has bit 31 set for a leaf
  float min_x[4], min_y[4], min_z[4], max_x[4], max_y[4], max_z[4];
  uint32_t children[4];
  uint32_t child_count;                 // 1-4
  uint32_t valid_mask;                  // (1 << child_count) - 1
};
struct Leaf { Aabb bounds; uint32_t leaf_idx; };                // bvh.rs:448-451 (the quantised triangle box, triangle index)
struct TriInfo { uint32_t flags; float edge_v0_v1_angle, edge_v1_v2_angle, edge_v2_v0_angle; };  // triangle_info_map.rs:31-36
struct Plane {                          // static_plane_shape.rs:11-20
  uint32_t body;
  Vec3A plane_normal;
  uint32_t is_single_axis, single_axis_idx, single_axis_backwards;
  Aabb aabb_ident_cache, aabb_cache;
};
struct BodyConst {                      // rigid_body.rs:129-172 construction constants, all 23 bodies (dump B.3)
  uint32_t world_array_idx, user_idx, user_pointer, broadphase_handle, filter_group, filter_mask;
  float mass, inv_mass;
  Vec3A inv_inertia_local, inv_mass_splat;
  float friction, restitution, contact_processing_threshold, contact_breaking_threshold;
  float linear_damping, linear_sleeping_threshold, angular_sleeping_threshold;
  Affine3A pose;                        // a static body's world transform (identity for dynamic bodies)
  uint32_t shape_kind;                  // 0 triangle mesh, 1 static plane, 2 sphere, 3 compound (car)
  Vec3A sphere_implicit_dim;            // kind 2 (x = radius)
  float sphere_margin;
  Vec3A box_half_extents;               // kind 3 child box
  float box_margin;
  Affine3A child_trans;
  Aabb compound_local_aabb;
  uint32_t mesh_tri_count;              // kind 0
};
struct WheelConst {                     // wheel_info.rs:43-45, 52, 54
  Vec3A chassis_connection_point_cs;    // BT, y already negated for the left wheels
  float suspension_rest_length_1, wheels_radius, suspension_force_scale, real_ray_length;
};
struct CarConst {                       // car/car_info.rs, car_body_config.rs:85
  uint32_t car_idx, team, rigid_body_idx, three_wheels;
  Vec3A hitbox_size, hitbox_pos_offset;  // car_body_config.rs:74-78 (uu): the pad box test (boost_pad_grid.rs hitbox_overlaps_pad_box)
  WheelConst wheels[NUM_WHEELS];
};
struct PadConst {                       // boost_pad/base.rs:7-12
  Vec3A pos;
  uint32_t is_big;
  float box_radius, cyl_radius, max_cooldown, boost_amount;
  Aabb aabb;
};
struct ProxyConst {                     // broadphase_proxy.rs:74-82; aabb, cell and indices are meaningful for static bodies only
  uint32_t client_obj_idx, filter_group, filter_mask, unique_id;
  Aabb aabb;
  uint32_t cell_idx, indices[3];
};
struct CellRange { uint32_t first, count; };   // GridCell.static_handles (grid_broadphase.rs:22), indexed by cell

struct StaticData {
  // WRLD
  uint32_t game_mode, no_ball_rot, ball_body;      // game_mode.rs:14, arena_config.rs:45, ball/base.rs:31
  Vec3A gravity_uu;                                // mutator_config.rs:28
  float mut_f[NUM_MUT];                            // mutator_config.rs:30-69 in order (car_mass .. ball_radius)
  uint32_t unlimited_flips, unlimited_double_jumps, recharge_boost_enabled;
  float recharge_boost_per_second, recharge_boost_delay;
  uint32_t demo_mode, enable_team_demos;
  float goal_base_threshold_y;
  uint32_t boost_pads_enabled;                     // mutator_config.rs:96
  Vec3A gravity_bt;                                // discrete_dynamics_world.rs:29
  uint32_t dyn_body[NUM_DYN];                      // :27 dynamic_body_idcs
  Vec3A grid_min_pos, grid_max_pos;                // grid_broadphase.rs:49-53 (BT)
  float grid_cell_size, grid_cell_size_sq;
  uint32_t grid_num_cells[3];
  Vec3A sap_world_aabb_min, sap_quantize;          // axis_sweep.rs:48-49
  uint32_t sap_handle_mask, sap_sentinel, sap_max_handles;   // axis_sweep.rs:13-15 (u16 values)
  float sap_world_half_extent;                     // :16
  float max_pad_z;                                 // boost_pad_grid.rs:51
  uint32_t dyn_handle[NUM_DYN];                    // broadphase handle of each dynamic body (BODY.broadphase_handle)
  BodyConst bodies[NUM_BODIES];
  Plane planes[NUM_PLANES];
  MeshInfo meshes[NUM_MESHES];
  CarConst cars[NUM_CARS];
  PadConst pads[NUM_PADS];
  ProxyConst proxies[NUM_HANDLES];                 // PROX (static dump version 2)
  CellRange cells[NUM_GRID_CELLS];
  uint32_t cell_handles[MAX_CELL_ENTRIES];
  TriInfo infos[MAX_TRIS];
  Tri tris[MAX_TRIS];
  WideNode nodes[MAX_NODES];
  Leaf leaves[MAX_LEAVES];
  float table_sin[TABLE_SIN_LEN];                  // car/base.rs:101 table_sin, filled by the host with the platform f64 sin
};

constexpr int STATIC_WORDS = (int)(sizeof(StaticData) / 4);
static_assert(sizeof(StaticData) % 4 == 0, "StaticData is an array of u32 words");

}  // namespace lfsim
