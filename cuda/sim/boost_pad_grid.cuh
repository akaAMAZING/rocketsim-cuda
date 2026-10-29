// V3: sim/boost_pad/boost_pad_grid.rs and base.rs (0.3.3: pad membership, one pad per car, box hitbox). Each pad keeps the cars
// touching it in arrival order (ArenaState.pad_touching); a pad gives boost only to the first of them that can take it. A car
// joins a pad through the pickup cylinder on its centre and stays a member while either the cylinder or the box test against its
// hitbox holds. V3 finds new pads with a BVH query of the car's centre whose leaf bounds are the pads' own boxes (combine = exact
// min / max), so the query is an exact box test per pad; the pads' cylinders do not overlap, so at most one pad can take a car and
// the scan order is immaterial. The grid's `winners` and `touching_pad_by_car` are scratch, rebuilt each tick before any read.
#pragma once
#include "common/libm.cuh"
#include "sim/arena_state.cuh"
#include "sim/consts.cuh"

namespace lfsim {

constexpr float PAD_BOX_HEIGHT = 64.0f;          // consts.rs:562 boost_pads::BOX_HEIGHT
constexpr float F32_EPSILON = 1.1920929e-7f;     // f32::EPSILON

// boost_pad_grid.rs:30-33 overlaps_values: the car centre inside the pad's pickup cylinder.
LF_HDI bool pad_cyl_overlaps(Vec3A pad_pos, float radius, Vec3A pos) {
  float dx = lf_sub(pad_pos.x, pos.x), dy = lf_sub(pad_pos.y, pos.y);  // Vec2::distance_squared = (self - rhs).length_squared()
  float dist_sq_2d = lf_add(lf_mul(dx, dx), lf_mul(dy, dy));
  return dist_sq_2d < lf::powi(radius, 2) && lf_abs(lf_sub(pos.z, pad_pos.z)) <= c::CYL_HEIGHT;
}

// boost_pad_grid.rs:35-41 can_receive_boost: not demolished, and boost below the maximum (or either is NaN).
LF_HDI bool pad_can_receive(const Car& car, float max_boost) { return !car.is_demoed && !(car.boost >= max_boost); }

// boost_pad_grid.rs:43-86 hitbox_overlaps_pad_box: separating-axis test of the car's hitbox against the pad's box.
LF_HDI bool pad_box_overlaps(Vec3A pad_pos, float box_radius, const Car& car, const CarConst& cc) {
  const Mat3A rot = car.rot_mat;
  const Vec3A center = car.pos + rot * cc.hitbox_pos_offset;
  const Vec3A car_axes[3] = {rot.x_axis, rot.y_axis, rot.z_axis};
  const Vec3A car_half = cc.hitbox_size * 0.5f;
  const Vec3A pad_half = {box_radius, box_radius, PAD_BOX_HEIGHT};
  const Vec3A delta = center - pad_pos;
  const Vec3A world_axes[3] = {{1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}};
  for (int k = 0; k < 15; ++k) {  // the three world axes, the three car axes, then each car axis crossed with each world axis
    const Vec3A axis = k < 3 ? world_axes[k] : k < 6 ? car_axes[k - 3] : cross(car_axes[(k - 6) / 3], world_axes[(k - 6) % 3]);
    if (length_squared(axis) <= F32_EPSILON) continue;
    const float pad_radius = dot(pad_half, Vec3A{lf_abs(dot(world_axes[0], axis)), lf_abs(dot(world_axes[1], axis)),
                                                 lf_abs(dot(world_axes[2], axis))});
    const float car_radius = dot(car_half, Vec3A{lf_abs(dot(car_axes[0], axis)), lf_abs(dot(car_axes[1], axis)),
                                                 lf_abs(dot(car_axes[2], axis))});
    if (lf_abs(dot(delta, axis)) > lf_add(pad_radius, car_radius)) return false;
  }
  return true;
}

// boost_pad_grid.rs:235-243 refresh_winner: the first member that can take boost, or -1.
LF_HDI int pad_winner(const ArenaState* s, int pad, float max_boost) {
  const PadTouching& t = s->pad_touching[pad];
  for (uint32_t k = 0; k < t.n; ++k)
    if (pad_can_receive(s->cars[t.car_idx[k]], max_boost)) return (int)t.car_idx[k];
  return -1;
}

// boost_pad_grid.rs:179-225 update_membership with refresh_winners (:227-233): touching_pad[car] and winner[pad] (-1 for none).
LF_HDI void pad_update_membership(ArenaState* s, const StaticData* sd, float max_boost, int touching_pad[NUM_CARS],
                                  int winner[NUM_PADS]) {
  for (int ci = 0; ci < NUM_CARS; ++ci) touching_pad[ci] = -1;
  // retention, in pad order (it resolves a car listed on two pads: the first keeps it)
  for (int i = 0; i < NUM_PADS; ++i) {
    const PadConst& pad = sd->pads[i];
    PadTouching& t = s->pad_touching[i];
    uint32_t kept = 0;
    for (uint32_t k = 0; k < t.n; ++k) {  // Vec::retain: keeps order
      const uint32_t idx = t.car_idx[k];
      const Car& car = s->cars[idx];
      if (!car.is_demoed && touching_pad[idx] < 0 &&
          (pad_cyl_overlaps(pad.pos, pad.cyl_radius, car.pos) || pad_box_overlaps(pad.pos, pad.box_radius, car, sd->cars[idx]))) {
        touching_pad[idx] = i;
        t.car_idx[kept++] = idx;
      }
    }
    for (uint32_t k = kept; k < t.n; ++k) t.car_idx[k] = 0;  // slots from n on are zero
    t.n = kept;
  }
  // discovery through the cylinder, for cars on no pad (MembershipProcessor, :15-28)
  for (int ci = 0; ci < NUM_CARS; ++ci) {
    const Car& car = s->cars[ci];
    if (car.is_demoed || car.pos.z > sd->max_pad_z) continue;
    if (touching_pad[ci] >= 0) continue;
    for (int i = 0; i < NUM_PADS; ++i) {
      const PadConst& pad = sd->pads[i];
      // leaf box test, Aabb::intersects(car centre box): self.min <= rhs.max && self.max >= rhs.min on all axes
      if (!(glam::all(glam::cmple(pad.aabb.min, car.pos)) && glam::all(glam::cmpge(pad.aabb.max, car.pos)))) continue;
      if (!pad_cyl_overlaps(pad.pos, pad.cyl_radius, car.pos)) continue;
      PadTouching& t = s->pad_touching[i];
      bool listed = false;
      for (uint32_t k = 0; k < t.n; ++k) listed |= t.car_idx[k] == (uint32_t)ci;
      if (!listed) {
        if (t.n < NUM_CARS) t.car_idx[t.n++] = (uint32_t)ci;
        else ++s->overflow[OVF_PAD_TOUCHING];
      }
      touching_pad[ci] = i;
      break;
    }
  }
  for (int i = 0; i < NUM_PADS; ++i) winner[i] = pad_winner(s, i, max_boost);
}

// boost_pad_grid.rs:246-279 maybe_give_car_boost. Returns the pad index or -1.
LF_HDI int pad_maybe_give_car_boost(ArenaState* s, const StaticData* sd, int car_idx, const int touching_pad[NUM_CARS],
                                    const int winner[NUM_PADS], float max_boost) {
  Car& car = s->cars[car_idx];
  if (car.is_demoed) return -1;           // the stored position remains after demolition
  if (car.boost >= max_boost) return -1;  // already full on boost
  const int i = touching_pad[car_idx];
  if (i < 0 || winner[i] != car_idx) return -1;
  const PadConst& pad = sd->pads[i];
  const int64_t tick = (int64_t)get_u64(s->tick_count);
  if (s->pad_gave_boost[i].some) {
    int64_t last = (int64_t)get_u64(s->pad_gave_boost[i].v);
    if (lf_mul(lf_ll2f(tick - last), c::TICK_TIME) < pad.max_cooldown) return -1;
  }
  car.boost = rs_min(lf_add(car.boost, pad.boost_amount), max_boost);
  s->pad_gave_boost[i].some = 1;
  s->pad_gave_boost[i].v = make_u64((uint64_t)tick);
  return i;
}

// arena/base.rs:832-861 (step_tick), the pad part: membership, then each car in index order, refreshing the pad's winner after a
// pickup.
LF_HDI void boost_pads_tick(ArenaState* s, const StaticData* sd) {
  const float max_boost = sd->mut_f[c::MF_CAR_MAX_BOOST_AMOUNT];
  int touching_pad[NUM_CARS], winner[NUM_PADS];
  pad_update_membership(s, sd, max_boost, touching_pad, winner);
  for (int ci = 0; ci < NUM_CARS; ++ci) {
    const int pad = pad_maybe_give_car_boost(s, sd, ci, touching_pad, winner, max_boost);
    if (pad >= 0) {
      s->events |= 1u << (EV_PICKUP + ci);  // CarPickupBoost
      winner[pad] = pad_winner(s, pad, max_boost);
    }
  }
}

}  // namespace lfsim
