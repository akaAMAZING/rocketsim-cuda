// Bullet constants that several phases share, defined once.
#pragma once
#include <stdint.h>

namespace lfsim {

// V3: bullet/dynamics/rigid_body.rs:51-58 CollisionFlags.
constexpr uint32_t CF_STATIC = 1u, CF_CUSTOM_MATERIAL = 1u << 1, CF_CAN_SLEEP = 1u << 2, CF_NO_CONTACT_RESPONSE = 1u << 3,
                   CF_NO_WORLD_GRAVITY = 1u << 4, CF_NO_ANGULAR_MOTION = 1u << 5;

// V3: bullet/collision/narrowphase/persistent_manifold.rs:30
constexpr float CONTACT_BREAKING_THRESHOLD = 0.02f;

}  // namespace lfsim
