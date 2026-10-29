// Inputs of the libm tests, generated from an index so that the host, the device and the stand-alone checker draw the same
// cases without moving them (integer arithmetic only, hence identical on every side). atan2f and powf take the pair
// (a, b): atan2f(a, b) and powf(a, b).
#pragma once
#include "common/hd.cuh"

namespace lft {

LF_HDI uint64_t mix(uint64_t z) {  // splitmix64 finaliser
    z += 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

LF_HDI float make(uint32_t sign, uint32_t exp, uint32_t mant) { return lf_from_bits(sign << 31 | exp << 23 | (mant & 0x7FFFFFu)); }

// one binary32 across the binades: mostly uniform in the exponent (denormals, huge values) and a share in the range the
// simulation lives in; a share with a zero mantissa (exact powers of two)
LF_HDI float any_float(uint64_t r) {
    uint32_t sign = (uint32_t)(r >> 40) & 1, mant = (uint32_t)r, sel = (uint32_t)(r >> 32) % 100;
    uint32_t e = 1 + (uint32_t)(r >> 44) % 254;
    if (sel < 2) e = 0;                                    // denormals
    else if (sel < 50) e = 100 + (uint32_t)(r >> 44) % 50;  // 2^-27 .. 2^22
    if ((r >> 52) % 20 == 0) mant = 0;
    return make(sign, e, mant);
}

// a double for the f64 tests: half in the range of the auto-flip path (a binary32 value widened, or a double with an
// exponent in [-30, 30]), the rest across every binade of the format, including the subnormals and the largest values
LF_HDI double f64_case(uint64_t r) {
    uint32_t kind = (uint32_t)(r >> 56) % 10;
    uint64_t sign = (r >> 63) << 63, m = r & 0xFFFFFFFFFFFFFull, e;
    uint64_t u;
    if (kind < 3) e = 1023 - 30 + (r >> 52) % 61, u = sign | e << 52 | m;
    else if (kind < 6) {  // a binary32 value: sign, exponent 2^-12 .. 2^3, 24-bit significand
        uint64_t f32m = (r & 0x7FFFFFull) << 29;
        u = sign | (uint64_t)(1023 - 12 + (r >> 52) % 16) << 52 | f32m;
    } else if (kind < 9) u = sign | (1 + (r >> 52) % 2046) << 52 | m;
    else u = sign | ((r >> 52) & 1 ? (uint64_t)(r & 0xFFFFFull) : (0x7FEull << 52 | m));  // subnormal or near the top
#if defined(__CUDA_ARCH__)
    return __longlong_as_double((long long)u);
#else
    double d;
    memcpy(&d, &u, 8);
    return d;
#endif
}

enum { ATAN2 = 0, POW = 1 };

// the case `idx` of the random stream of `op`
LF_HDI void pair(int op, uint64_t idx, float* a, float* b) {
    uint64_t r = mix(idx * 2 + (uint64_t)op * 0x1234567ull), s = mix(r ^ 0xA5A5A5A5A5A5A5A5ull), t = mix(s);
    if (op == ATAN2) {
        uint32_t sel = (uint32_t)(r >> 8) % 100;
        float y = any_float(r), x = any_float(s);
        if (sel >= 50) {  // exponents of y and x within 30 of each other
            uint32_t ex = 1 + (uint32_t)(s >> 44) % 254;
            int ey = (int)ex + (int)((uint32_t)(r >> 44) % 61) - 30;
            ey = ey < 1 ? 1 : ey > 254 ? 254 : ey;
            x = make((uint32_t)(s >> 40) & 1, ex, (uint32_t)s);
            y = make((uint32_t)(r >> 40) & 1, (uint32_t)ey, (uint32_t)r);
        }
        *a = y;
        *b = x;
    } else {
        uint32_t sx = (uint32_t)(r >> 8) % 100, sy = (uint32_t)(s >> 8) % 100;
        uint32_t xs = (sx >= 85) ? 1u : 0u;  // 15% negative bases, with an integer exponent
        float x;
        if (sx < 30) x = make(0, 1 + (uint32_t)(r >> 44) % 254, (uint32_t)r);
        else if (sx < 65) x = make(0, 107 + (uint32_t)(r >> 44) % 41, (uint32_t)r);
        else x = make(0, 126 + (uint32_t)(r >> 44) % 2, (uint32_t)r);  // near 1
        float y;
        if (sy < 40 || xs) {
            int k = (int)((uint32_t)(t >> 20) % 61) - 30;  // integer in [-30, 30]
            y = lf_i2f(k);
            if (!xs && sy < 20) y = make((uint32_t)(t >> 40) & 1, 120 + (uint32_t)(t >> 44) % 15, (uint32_t)t);
            if (xs && sy % 3 == 0) y = lf_i2f(k * 7 + 1);
        } else if (sy < 75) y = make((uint32_t)(t >> 40) & 1, 120 + (uint32_t)(t >> 44) % 15, (uint32_t)t);
        else if (sy < 90) y = make((uint32_t)(t >> 40) & 1, 134 + (uint32_t)(t >> 44) % 7, (uint32_t)t);
        else y = any_float(t);
        *a = xs ? lf_from_bits(lf_bits(x) | 0x80000000u) : x;
        *b = y;
    }
}

// special values for the edge-case grid (every pair is tested)
constexpr int N_SPECIAL = 40;
LF_HDI float special(int i) {
    switch (i) {
        case 0: return lf_from_bits(0x00000000u);
        case 1: return lf_from_bits(0x80000000u);
        case 2: return lf_from_bits(0x3F800000u);  // 1
        case 3: return lf_from_bits(0xBF800000u);  // -1
        case 4: return lf_from_bits(0x7F800000u);  // inf
        case 5: return lf_from_bits(0xFF800000u);
        case 6: return lf_from_bits(0x7FC00000u);  // nan
        case 7: return lf_from_bits(0xFFC00001u);
        case 8: return lf_from_bits(0x7F800001u);  // signalling nan
        case 9: return lf_from_bits(0x00000001u);  // smallest denormal
        case 10: return lf_from_bits(0x80000001u);
        case 11: return lf_from_bits(0x007FFFFFu);  // largest denormal
        case 12: return lf_from_bits(0x00800000u);  // smallest normal
        case 13: return lf_from_bits(0x7F7FFFFFu);  // largest finite
        case 14: return lf_from_bits(0xFF7FFFFFu);
        case 15: return lf_from_bits(0x3F000000u);  // 0.5
        case 16: return lf_from_bits(0x40000000u);  // 2
        case 17: return lf_from_bits(0xC0000000u);  // -2
        case 18: return lf_from_bits(0x40400000u);  // 3
        case 19: return lf_from_bits(0xC0400000u);
        case 20: return lf_from_bits(0x40490FDBu);  // pi
        case 21: return lf_from_bits(0x3FC90FDBu);  // pi/2
        case 22: return lf_from_bits(0x3F7FFFFFu);  // 1 - 2^-24
        case 23: return lf_from_bits(0x3F800001u);  // 1 + 2^-23
        case 24: return lf_from_bits(0x4B800000u);  // 2^24
        case 25: return lf_from_bits(0x4F000000u);  // 2^31
        case 26: return lf_from_bits(0x5F000000u);  // 2^63
        case 27: return lf_from_bits(0x7E800000u);  // 2^126
        case 28: return lf_from_bits(0x2F800000u);  // 2^-34
        case 29: return lf_from_bits(0x33800000u);  // 2^-24
        case 30: return lf_from_bits(0x39000000u);  // 2^-13
        case 31: return lf_from_bits(0x3A800000u);  // 2^-12
        case 32: return lf_from_bits(0x42FE0000u);  // 127
        case 33: return lf_from_bits(0xC2FE0000u);
        case 34: return lf_from_bits(0x43000000u);  // 128
        case 35: return lf_from_bits(0x43160000u);  // 150
        case 36: return lf_from_bits(0xC3160000u);
        case 37: return lf_from_bits(0x3EAAAAABu);  // ~1/3
        case 38: return lf_from_bits(0x3FB504F3u);  // sqrt 2
        default: return lf_from_bits(0x3E800000u);  // 0.25
    }
}

}  // namespace lft
