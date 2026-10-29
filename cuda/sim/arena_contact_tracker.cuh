// V3: sim/arena/arena_contact_tracker.rs (the contact-added callback and the retained-contact report) and the callback's
// hand-over in sim/arena/base.rs. Car-ball and car-car records are appended to ArenaState.records in call order, the others
// are folded (below); V3's ball-only records are not kept (the importer asserts a Soccar 1v1 arena, never ball-only), and a
// record keeps only the three vectors P7 reads.
#pragma once
#include "sim/narrowphase.cuh"
#include "sim/internal_edge_utility.cuh"

namespace lfsim {

// sim/consts.rs:96-110 car::HIT_BALL_COEFS, HIT_WORLD_COEFS, HIT_CAR_COEFS
constexpr float HIT_BALL_FRICTION = 2.0f, HIT_BALL_RESTITUTION = 0.0f;
constexpr float HIT_WORLD_FRICTION = 0.3f, HIT_WORLD_RESTITUTION = 0.3f;
constexpr float HIT_CAR_FRICTION = 0.09f, HIT_CAR_RESTITUTION = 0.1f;

// arena_contact_tracker.rs:81-84 / 146-149: the user_idx order decides which body is "a"
LF_HDI bool tracker_should_swap(const NpCtx& c, uint32_t body_a, uint32_t body_b) {
  uint32_t ua = body_const(c, body_a).user_idx, ub = body_const(c, body_b).user_idx;
  return (ua != USER_NONE && ub != USER_NONE) ? ua > ub : ub != USER_NONE;
}

// V3: arena_contact_tracker.rs:114-120 and :166-171, push of a ContactRecord, FOLDED. P7 (base.rs:678-733)
// consumes the records in order, but in Soccar only the car-ball and car-car records have order-dependent effects. The rest are
// folded here into the state P7 would leave, which is identical because no P7 callback reads what they write:
//   * ball-world (user a = Ball, b = None; base.rs:1516-1540): ball.on_world_hit is a no-op outside Heatseeker and Snowday
//     (ball/base.rs:341-382), so the record only pushes BallHitWorld: bit EV_BALL_HIT_WORLD, and EV_BALL_WALL when |normal.z| < 0.8;
//   * car-world (user a = Car, b = None; base.rs:1572-1580): the CarHitWorld event bit of the car, and world_contact_normal := the
//     normal, so the LAST record of a car decides it (read only by the next tick's P1: car/base.rs:616, 795-810, 1006-1011);
//   * a record that P7 ignores (neither a Car nor a Ball-with-None as user a) is dropped.
// The stored records (car-ball, car-car) keep V3's call order. The signature and the call sites are V3's.
LF_HDI void tracker_push_record(NpCtx& c, bool is_swap, uint32_t rb_a, uint32_t rb_b, const ManifoldPoint& p) {
  ArenaState* s = c.s;
  const BodyConst& ba = body_const(c, rb_a);
  const BodyConst& bb = body_const(c, rb_b);
  if (ba.user_idx == USER_CAR) {
    if (bb.user_idx != USER_BALL && bb.user_idx != USER_CAR) {
      uint32_t car = ba.user_pointer;
      s->fold_events |= 1u << (EV_CAR_HIT_WORLD + car);
      s->fold_car_normal[car] = p.normal_world_on_b;
      return;
    }
  } else if (ba.user_idx == USER_BALL && bb.user_idx == USER_NONE) {
    s->fold_events |= 1u << EV_BALL_HIT_WORLD;
    if (lf_abs(p.normal_world_on_b.z) < 0.8f) s->fold_events |= 1u << EV_BALL_WALL;
    return;
  } else {
    return;
  }
  if (s->n_records >= (uint32_t)MAX_RECORDS) {
    s->overflow[OVF_RECORDS] += 1;
    return;
  }
  ContactRecord& r = s->records[s->n_records++];
  r.is_swap = is_swap ? 1u : 0u;
  r.rb_idx_a = rb_a;
  r.rb_idx_b = rb_b;
  r.point.pos_world_on_b = p.pos_world_on_b;
  r.point.pos_world_on_a = p.pos_world_on_a;
  r.point.normal_world_on_b = p.normal_world_on_b;
}

// V3: arena_contact_tracker.rs:77 callback(manifold_point, body_a, body_b, idx). `idx` is -1 for None.
LF_HDI void tracker_callback(NpCtx& c, ManifoldPoint* mp, uint32_t body_a, uint32_t body_b, int32_t idx) {
  bool should_swap = tracker_should_swap(c, body_a, body_b);
  if (should_swap) {
    uint32_t t = body_a;
    body_a = body_b;
    body_b = t;
  }

  uint32_t user_a = body_const(c, body_a).user_idx;
  uint32_t user_b = body_const(c, body_b).user_idx;

  if (user_a == USER_CAR) {
    if (user_b == USER_BALL) {
      mp->combined_friction = HIT_BALL_FRICTION;
      mp->combined_restitution = HIT_BALL_RESTITUTION;
    } else if (user_b == USER_CAR) {
      mp->combined_friction = HIT_CAR_FRICTION;
      mp->combined_restitution = HIT_CAR_RESTITUTION;
    } else {
      mp->combined_friction = HIT_WORLD_FRICTION;
      mp->combined_restitution = HIT_WORLD_RESTITUTION;
    }
  } else if (user_a == USER_BALL && user_b == USER_NONE && body_is_static(c, body_b)) {
    mp->is_special = 1;
  }

  // Record contact data before edge adjustment changes the manifold.
  tracker_push_record(c, should_swap, body_a, body_b, *mp);

  if (idx >= 0) adjust_internal_edge_contacts(c, mp, body_b, (uint32_t)idx);
}

// V3: arena_contact_tracker.rs:142 contact_processed: only car-world retained contacts are reported again.
LF_HDI void tracker_contact_processed(NpCtx& c, const ManifoldPoint& mp, uint32_t body_a, uint32_t body_b) {
  bool should_swap = tracker_should_swap(c, body_a, body_b);
  if (should_swap) {
    uint32_t t = body_a;
    body_a = body_b;
    body_b = t;
  }

  uint32_t user_a = body_const(c, body_a).user_idx;
  uint32_t user_b = body_const(c, body_b).user_idx;
  if (user_a != USER_CAR || user_b == USER_BALL || user_b == USER_CAR) return;

  tracker_push_record(c, should_swap, body_a, body_b, mp);
}

}  // namespace lfsim
