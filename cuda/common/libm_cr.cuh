// Correctly rounded binary32 sin, cos, tan, atan2 and pow, identical bit for bit on the host and the device.
//
// Ported from the CORE-MATH project (https://core-math.gitlabpages.inria.fr/, git 0ce0cbc15935e24c656280e5a46c34e89f301b7a):
//   src/binary32/sin/sinf.c, cos/cosf.c, tan/tanf.c   Copyright (c) 2022-2025 Alexei Sibidanov
//   src/binary32/atan2/atan2f.c, pow/powf.c           Copyright (c) 2022-2025 Alexei Sibidanov (powf with Paul Zimmermann)
// The tables, polynomial coefficients, special-case databases and the order of operations are CORE-MATH's. The licence of
// those files (MIT) follows.
//
//   Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
//   documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
//   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
//   persons to whom the Software is furnished to do so, subject to the following conditions:
//
//   The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
//   Software.
//
//   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
//   WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
//   COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
//   OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//
// What differs from CORE-MATH, and why the result is the same on every platform:
//  * Every double operation is one IEEE operation (+, -, *, /) with round-to-nearest: the wrapper type D routes each to the
//    device intrinsic (__dadd_rn ...), which nvcc never contracts, or to the plain operator on the host, which does not
//    contract when the host build forbids it (MSVC /fp:precise without /arch:AVX2, gcc/clang -ffp-contract=off; see hd.cuh).
//    The only other double operations are floor and fabs (exact by definition), conversions and the exact scaling by 2^k.
//  * FMA. CORE-MATH calls fma in the accurate paths (muldd, the powf reduction, atan2f's refinement). Here it is the
//    software fma of Boldo and Melquiond (2008), "Emulation of a FMA and correctly rounded sums: proved algorithms using
//    rounding to odd": Dekker's exact product (Veltkamp split), TwoSum, one round-to-odd sum and one rounding. It uses
//    only the operations above, so host and device run the same sequence of correctly rounded operations and cannot
//    differ. It is the correctly rounded fma whenever the exact product and the sums neither overflow nor underflow
//    (Dekker's error term needs |a*b| > 2^-969); the arguments here are far from both limits (checked against hardware
//    fma on the paths they are used in). It is NOT the device's hardware fma, deliberately: a hardware
//    fma is correctly rounded too and equal wherever the software one is exact, but "equal where exact" is not a
//    proof for the limits; the same code on both sides is.
//  * The binary32 fma calls of the tiny-argument paths are replaced by what they compute: for sinf and tanf below the
//    thresholds the fma rounds to x itself (the correction is below half an ulp of x: |x|*|x| < 2^-25 |x| < ulp(x)/2), and
//    for cosf a binary32 fma emulated through the exact double product and a tie fix (fmaf_cr).
//  * 128-bit integers (rbig) are two 64-bit limbs; roundeven is the 1.5*2^52 addition; ctz/clz are loops (rare paths).
//  * No fenv, errno or exception flags. A NaN result is always the canonical quiet NaN 0x7FC00000 (hardware NaN payloads
//    and signs differ between x86 and the GPU; the physics never reads them).
#pragma once
#include "hd.cuh"

#if defined(__CUDA_ARCH__)
#define LF_TBL static __device__ const
#else
#define LF_TBL static const
#endif

namespace lfcr {

// ---- the double type: one IEEE operation per operator ----------------------------------------------------------------------
struct D {
    double v;
    LF_HD D() : v(0.0) {}
    LF_HD D(double x) : v(x) {}
};

#if defined(__CUDA_ARCH__)
LF_HDI D operator+(D a, D b) { return D(__dadd_rn(a.v, b.v)); }
LF_HDI D operator-(D a, D b) { return D(__dadd_rn(a.v, -b.v)); }
LF_HDI D operator*(D a, D b) { return D(__dmul_rn(a.v, b.v)); }
LF_HDI D operator/(D a, D b) { return D(__ddiv_rn(a.v, b.v)); }
LF_HDI uint64_t dbits(D a) { return (uint64_t)__double_as_longlong(a.v); }
LF_HDI D dfrom(uint64_t u) { return D(__longlong_as_double((long long)u)); }
LF_HDI D ll2d(int64_t v) { return D(__ll2double_rn((long long)v)); }
LF_HDI float d2f(D a) { return __double2float_rn(a.v); }
LF_HDI int64_t d2ll(D a) { return (int64_t)__double2ll_rz(a.v); }
LF_HDI D dfloor(D a) { return D(::floor(a.v)); }
LF_HDI D dsqrt(D a) { return D(__dsqrt_rn(a.v)); }
#else
LF_HDI D operator+(D a, D b) { return D(a.v + b.v); }
LF_HDI D operator-(D a, D b) { return D(a.v - b.v); }
LF_HDI D operator*(D a, D b) { return D(a.v * b.v); }
LF_HDI D operator/(D a, D b) { return D(a.v / b.v); }
LF_HDI uint64_t dbits(D a) { uint64_t u; memcpy(&u, &a.v, 8); return u; }
LF_HDI D dfrom(uint64_t u) { double d; memcpy(&d, &u, 8); return D(d); }
LF_HDI D ll2d(int64_t v) { return D((double)v); }
LF_HDI float d2f(D a) { return (float)a.v; }
LF_HDI int64_t d2ll(D a) { return (int64_t)a.v; }
LF_HDI D dfloor(D a) { return D(::floor(a.v)); }
LF_HDI D dsqrt(D a) { return D(::sqrt(a.v)); }
#endif
LF_HDI D operator-(D a) { return D(-a.v); }  // exact
LF_HDI D& operator+=(D& a, D b) { a = a + b; return a; }
LF_HDI D& operator-=(D& a, D b) { a = a - b; return a; }
LF_HDI D& operator*=(D& a, D b) { a = a * b; return a; }
LF_HDI bool operator<(D a, D b) { return a.v < b.v; }
LF_HDI bool operator>(D a, D b) { return a.v > b.v; }
LF_HDI bool operator<=(D a, D b) { return a.v <= b.v; }
LF_HDI bool operator>=(D a, D b) { return a.v >= b.v; }
LF_HDI bool operator==(D a, D b) { return a.v == b.v; }
LF_HDI D dabs(D a) { return dfrom(dbits(a) & 0x7FFFFFFFFFFFFFFFull); }
LF_HDI D f2d(float a) { return D((double)a); }  // exact
LF_HDI D i2d(int a) { return D((double)a); }    // exact

LF_HDI float fabs_f(float a) { return lf_from_bits(lf_bits(a) & 0x7FFFFFFFu); }
LF_HDI float nan_f() { return lf_from_bits(0x7FC00000u); }
LF_HDI float copysign_f(float a, float s) { return lf_from_bits((lf_bits(a) & 0x7FFFFFFFu) | (lf_bits(s) & 0x80000000u)); }

// ---- software fma (Boldo-Melquiond) ----------------------------------------------------------------------------------------
LF_HDI void two_sum(D a, D b, D* s, D* e) {
    D ss = a + b;
    D bb = ss - a;
    *e = (a - (ss - bb)) + (b - bb);
    *s = ss;
}
LF_HDI void two_prod(D a, D b, D* p, D* e) {
    D pp = a * b;
    D ca = a * D(134217729.0), ah = ca - (ca - a), al = a - ah;  // Veltkamp splits, 2^27 + 1
    D cb = b * D(134217729.0), bh = cb - (cb - b), bl = b - bh;
    *e = (((ah * bh - pp) + ah * bl) + al * bh) + al * bl;
    *p = pp;
}
// a + b rounded to odd: the RN sum, moved to the odd neighbour on the side of the error when it is inexact and even.
LF_HDI D add_odd(D a, D b) {
    D s, e;
    two_sum(a, b, &s, &e);
    if (e.v != 0.0) {
        uint64_t u = dbits(s);
        if (!(u & 1)) u += ((e.v > 0.0) == (s.v > 0.0)) ? 1 : (uint64_t)-1;
        s = dfrom(u);
    }
    return s;
}
LF_HDI D fma(D a, D b, D c) {
    D uh, ul, th, tl;
    two_prod(a, b, &uh, &ul);
    two_sum(c, uh, &th, &tl);
    return th + add_odd(tl, ul);
}
// binary32 fma for a result in the normal binary32 range: the exact double product, one double sum (RN) with its exact error,
// and the single case where converting that sum to binary32 would round twice (the sum sits on a binary32 midpoint while
// the error is not zero) pushed to the right side.
LF_HDI float fmaf_cr(float a, float b, float c) {
    D s, e;
    two_sum(f2d(a) * f2d(b), f2d(c), &s, &e);
    uint64_t u = dbits(s);
    if (e.v != 0.0 && (u & 0x1FFFFFFFull) == 0x10000000ull) {
        u += ((e.v > 0.0) == (s.v > 0.0)) ? 1 : (uint64_t)-1;
        s = dfrom(u);
    }
    return d2f(s);
}

// round to nearest even of |x| < 2^51 (every argument here is far below); larger doubles are integers already
LF_HDI D roundeven_finite(D x) {
    if (dabs(x) >= D(2251799813685248.0)) return x;
    D m = D(0x1.8p52);
    return (x + m) - m;
}

// ---- argument reduction shared by sinf and cosf ----------------------------------------------------------------------------
// m * k + cin as 128 bits (m < 2^24)
LF_HDI uint64_t mul_add(uint64_t m, uint64_t k, uint64_t cin, uint64_t* hi) {
    uint64_t a = (k & 0xFFFFFFFFull) * m, b = (k >> 32) * m;
    uint64_t lo = a + (b << 32), h = (b >> 32) + (lo < a ? 1u : 0u);
    uint64_t lo2 = lo + cin;
    h += (lo2 < lo ? 1u : 0u);
    *hi = h;
    return lo2;
}

// 2/pi in 256 bits, for the reduction of |x| >= 2^26
LF_TBL uint64_t ipi[] = {0xfe5163abdebbc562, 0xdb6295993c439041, 0xfc2757d1f534ddc0, 0xa2f9836e4e441529};
// ---- small tables of CORE-MATH (the large ones are below) ------------------------------------------------------------------
LF_TBL float sin_st[4][3] = {{0x1.33333p+13f, -0x1.63f4bap-2f, -0x1p-27f},
                             {0x1.75b8a2p-1f, 0x1.55688ap-1f, -0x1p-26f},
                             {0x1.4f0654p+0f, 0x1.ee836cp-1f, -0x1p-26f},
                             {0x1.2d97c8p+3f, -0x1.99bc5ap-26f, -0x1p-51f}};
LF_TBL float cos_st[5][3] = {{0x1.2d97c8p+2f, 0x1.99bc5cp-27f, -0x1p-52f},
                             {0x1.4555p+51f, 0x1.115d7ep-1f, -0x1p-26f},
                             {0x1.48a858p+54f, 0x1.f48148p-2f, 0x1p-27f},
                             {0x1.3170fp+63f, 0x1.fe2976p-1f, 0x1p-26f},
                             {0x1.2b9622p+67f, 0x1.f0285ep-1f, -0x1p-26f}};
LF_TBL float tan_st[8][3] = {{0x1.143ec4p+0f, 0x1.ddf9f6p+0f, -0x1.891d24p-52f},
                             {0x1.ada6aap+27f, 0x1.e80304p-3f, 0x1.419f46p-58f},
                             {0x1.af61dap+48f, 0x1.60d1c8p-2f, -0x1.2d6c3ap-55f},
                             {0x1.0088bcp+52f, 0x1.ca1edp+0f, 0x1.f6053p-53f},
                             {0x1.f90dfcp+72f, 0x1.597f9cp-1f, 0x1.925978p-53f},
                             {0x1.cc4e22p+85f, -0x1.f33584p+1f, 0x1.d7254ap-51f},
                             {0x1.a6ce12p+86f, -0x1.c5612ep-1f, -0x1.26c33ep-53f},
                             {0x1.6a0b76p+102f, -0x1.e42a1ep+0f, -0x1.1dc906p-52f}};
LF_TBL double at_off[8] = {0.0, 0x1.921fb54442d18p+0, 0x1.921fb54442d18p+1, 0x1.921fb54442d18p+0,
                           -0.0, -0x1.921fb54442d18p+0, -0x1.921fb54442d18p+1, -0x1.921fb54442d18p+0};
LF_TBL double at_offl[8] = {0.0, 0x1.1a62633145c07p-54, 2 * 0x1.1a62633145c07p-54, 0x1.1a62633145c07p-54,
                            -0.0, -0x1.1a62633145c07p-54, -2 * 0x1.1a62633145c07p-54, -0x1.1a62633145c07p-54};
LF_TBL uint32_t pw_xmax[] = {0, 0xffffff, 4095, 255, 63, 27, 15, 9, 7, 5, 5, 3, 3, 3, 3, 3};



// 2/pi * x = q + z for |x| >= 2^26 (sinf, cosf: e0 = 124) or |x| >= 2^28 (tanf: e0 = 127); returns z, q in *q
LF_HDI D rbig(uint32_t u, int* q, int e0) {
    int e = (int)((u >> 23) & 0xff);
    uint64_t m = (uint64_t)((u & (~0u >> 9)) | (1u << 23));
    uint64_t h0, h1, h2, h3;
    uint64_t l0 = mul_add(m, ipi[0], 0, &h0);
    uint64_t l1 = mul_add(m, ipi[1], h0, &h1);
    uint64_t l2 = mul_add(m, ipi[2], h1, &h2);
    uint64_t l3 = mul_add(m, ipi[3], h2, &h3);
    (void)l0;
    uint64_t p3h = h3, p3l = l3, p2l = l2, p1l = l1;
    int64_t a;
    int i;
    int k = e - e0, s = k - 23;
    // called with 155 <= e <= 254: 5 <= s <= 104
    if (s < 64) {
        i = (int)(uint32_t)(p3h << s | p3l >> (64 - s));
        a = (int64_t)(p3l << s | p2l >> (64 - s));
    } else if (s == 64) {
        i = (int)(uint32_t)p3l;
        a = (int64_t)p2l;
    } else {
        i = (int)(uint32_t)(p3l << (s - 64) | p2l >> (128 - s));
        a = (int64_t)(p2l << (s - 64) | p1l >> (128 - s));
    }
    int sgn = (int32_t)u >> 31;
    int64_t sm = a >> 63;
    i = (int)((int64_t)i - sm);
    D z = ll2d(a ^ (int64_t)sgn) * D(0x1p-64);
    i = (i ^ sgn) - sgn;
    *q = i;
    return z;
}

// low 32 bits of 1.5*2^52 + id, i.e. id as an int
LF_HDI int quo(D id) { return (int)(uint32_t)(dbits(D(0x1.8p52) + id) & 0xFFFFFFFFull); }

LF_HDI D rltl(float z, int* q) {
    D x = f2d(z);
    D idl = D(-0x1.b1bbead603d8bp-29) * x, idh = D(0x1.45f306ep+2) * x, id = roundeven_finite(idh);
    *q = quo(id);
    return (idh - id) + idl;
}
LF_HDI D rltl0(D x, int* q) {
    D idh = D(0x1.45f306dc9c883p+2) * x, id = roundeven_finite(idh);
    *q = quo(id);
    return idh - id;
}

// sin, cos: polynomial coefficients and the table of sin(j pi/16), j = 0..31
LF_TBL double sc_b[] = {0x1.3bd3cc9be45dcp-6, -0x1.03c1f081b0833p-14, 0x1.55d3c6fc9ac1fp-24, -0x1.e1d3ff281b40dp-35};
LF_TBL double sc_a[] = {0x1.921fb54442d17p-3, -0x1.4abbce6256a39p-10, 0x1.466bc5a518c16p-19, -0x1.32bdc61074ff6p-29};
LF_TBL double sc_tb[] = {0x0p+0, 0x1.8f8b83c69a60bp-3, 0x1.87de2a6aea963p-2, 0x1.1c73b39ae68c8p-1,
    0x1.6a09e667f3bcdp-1, 0x1.a9b66290ea1a3p-1, 0x1.d906bcf328d46p-1, 0x1.f6297cff75cbp-1,
    0x1p+0, 0x1.f6297cff75cbp-1, 0x1.d906bcf328d46p-1, 0x1.a9b66290ea1a3p-1,
    0x1.6a09e667f3bcdp-1, 0x1.1c73b39ae68c8p-1, 0x1.87de2a6aea963p-2, 0x1.8f8b83c69a60bp-3,
    0x0p+0, -0x1.8f8b83c69a60bp-3, -0x1.87de2a6aea963p-2, -0x1.1c73b39ae68c8p-1,
    -0x1.6a09e667f3bcdp-1, -0x1.a9b66290ea1a3p-1, -0x1.d906bcf328d46p-1, -0x1.f6297cff75cbp-1,
    -0x1p+0, -0x1.f6297cff75cbp-1, -0x1.d906bcf328d46p-1, -0x1.a9b66290ea1a3p-1,
    -0x1.6a09e667f3bcdp-1, -0x1.1c73b39ae68c8p-1, -0x1.87de2a6aea963p-2, -0x1.8f8b83c69a60bp-3};


// ---- sinf ------------------------------------------------------------------------------------------------------------------
LF_HD inline float sinf_database(float x, float r) {
    uint32_t ax = lf_bits(x) & (~0u >> 1);
    for (int i = 0; i < 4; i++)
        if (lf_bits(sin_st[i][0]) == ax) {
            float sgn = copysign_f(1.0f, x);
            return lf_add(lf_mul(sgn, sin_st[i][1]), lf_mul(sgn, sin_st[i][2]));
        }
    return r;
}

LF_HD inline float sinf_big(float x) {
    uint32_t u = lf_bits(x), ax = u << 1;
    if (ax >= 0xffu << 24) return nan_f();  // nan or +-inf
    int ia;
    D z = rbig(u, &ia, 124);
    D z2 = z * z, z4 = z2 * z2;
    D aa = (D(sc_a[0]) + z2 * D(sc_a[1])) + z4 * (D(sc_a[2]) + z2 * D(sc_a[3]));
    D bb = (D(sc_b[0]) + z2 * D(sc_b[1])) + z4 * (D(sc_b[2]) + z2 * D(sc_b[3]));
    D s0 = D(sc_tb[ia & 31]), c0 = D(sc_tb[(ia + 8u) & 31]);
    D r = s0 + z * (aa * c0 - bb * (z * s0));
    return d2f(r);
}

LF_HD inline float sinf_cr(float x) {
    uint32_t ax = lf_bits(x) << 1;
    int ia;
    D z0 = f2d(x), z;
    if (ax > 0x99000000u || ax < 0x73000000u) {  // |x| > 2^26 or |x| < 2^-12
        if (ax < 0x73000000u) {
            if (ax < 0x66000000u) return x;  // |x| < 2^-25 (and x = 0): fmaf(-x, |x|, x) is x
            return lf_add(lf_mul(lf_mul(-0x1.555556p-3f, x), lf_mul(x, x)), x);
        }
        return sinf_big(x);
    }
    if (ax < 0x822d97c8u) {
        if (ax == 0x7e75b8a2u || ax == 0x7f4f0654u) return sinf_database(x, 0.0f);
        z = rltl0(z0, &ia);
    } else {
        if (ax == 0x8c333330u) return sinf_database(x, 0.0f);
        z = rltl(x, &ia);
    }
    D z2 = z * z, z4 = z2 * z2;
    D aa = (D(sc_a[0]) + z2 * D(sc_a[1])) + z4 * (D(sc_a[2]) + z2 * D(sc_a[3]));
    D bb = (D(sc_b[0]) + z2 * D(sc_b[1])) + z4 * (D(sc_b[2]) + z2 * D(sc_b[3]));
    D s0 = D(sc_tb[ia & 31]), c0 = D(sc_tb[(ia + 8) & 31]);
    D r = s0 + aa * (z * c0) - bb * (z2 * s0);
    return d2f(r);
}

// ---- cosf ------------------------------------------------------------------------------------------------------------------
// cos(j pi/16) is sin((j + 8) pi/16): the cosine table of CORE-MATH is the sine table shifted by 8 (identical constants).
LF_HD inline float cosf_database(float x, float r) {
    uint32_t ax = lf_bits(x) & (~0u >> 1);
    for (int i = 0; i < 5; i++)
        if (lf_bits(cos_st[i][0]) == ax) return lf_add(cos_st[i][1], cos_st[i][2]);
    return r;
}

LF_HD inline float cosf_big(float x) {
    uint32_t u = lf_bits(x), ax = u << 1;
    if (ax >= 0xffu << 24) return nan_f();
    int ia;
    D z = rbig(u, &ia, 124);
    D z2 = z * z, z4 = z2 * z2;
    D aa = (D(sc_a[0]) + z2 * D(sc_a[1])) + z4 * (D(sc_a[2]) + z2 * D(sc_a[3]));
    D bb = (D(sc_b[0]) + z2 * D(sc_b[1])) + z4 * (D(sc_b[2]) + z2 * D(sc_b[3]));
    D s0 = D(sc_tb[(ia + 16u) & 31]), c0 = D(sc_tb[(ia + 8u) & 31]);
    D r = c0 + z * (aa * s0 - bb * (z * c0));
    uint64_t tail = (dbits(r) + 6) & (~(uint64_t)0 >> 36);
    if (tail <= 12) return cosf_database(x, d2f(r));
    return d2f(r);
}

LF_HD inline float cosf_cr(float x) {
    uint32_t ax = lf_bits(x) << 1;
    int ia;
    D z0 = f2d(x), z;
    if (ax > 0x99000000u || ax < 0x74000000u) {  // |x| > 2^26 or |x| < 2^-11
        if (ax < 0x74000000u) return lf_mul(0.5f, fmaf_cr(x, -x, 2.0f));
        return cosf_big(x);
    }
    if (ax < 0x82a41896u) {
        if (ax == 0x812d97c8u) return cosf_database(x, 0.0f);
        z = rltl0(z0, &ia);
    } else {
        z = rltl(x, &ia);
    }
    D z2 = z * z, z4 = z2 * z2;
    D aa = (D(sc_a[0]) + z2 * D(sc_a[1])) + z4 * (D(sc_a[2]) + z2 * D(sc_a[3]));
    D bb = (D(sc_b[0]) + z2 * D(sc_b[1])) + z4 * (D(sc_b[2]) + z2 * D(sc_b[3]));
    D c0 = D(sc_tb[(ia + 8) & 31]), s0 = D(sc_tb[(ia + 16) & 31]);
    D r = c0 + aa * (z * s0) - bb * (z2 * c0);
    return d2f(r);
}

// ---- tanf ------------------------------------------------------------------------------------------------------------------
LF_HD inline D rltl_tan(float z, int* q) {
    D x = f2d(z);
    D idl = D(-0x1.b1bbead603d8bp-32) * x, idh = D(0x1.45f306ep-1) * x, id = roundeven_finite(idh);
    *q = (int)d2ll(id);
    return (idh - id) + idl;
}

LF_TBL double tan_cn[] = {0x1.921fb54442d18p+0, -0x1.fd226e573289fp-2, 0x1.b7a60c8dac9f6p-6, -0x1.725beb40f33e5p-13};
LF_TBL double tan_cd[] = {0x1p+0, -0x1.2395347fb829dp+0, 0x1.2313660f29c36p-3, -0x1.9a707ab98d1c1p-9};


LF_HD inline float tanf_cr(float x) {
    uint32_t u = lf_bits(x);
    int e = (int)((u >> 23) & 0xff), i;
    D z;
    if (e < 127 + 28) {  // |x| < 2^28
        if (e < 115) return x;  // |x| < 2^-13 (and x = 0): tan x = x(1 + x^2/3 ...) and the fma of CORE-MATH rounds to x
        z = rltl_tan(x, &i);
    } else if (e < 0xff) {
        z = rbig(u, &i, 127);
    } else {
        return nan_f();  // nan or inf
    }
    D z2 = z * z, z4 = z2 * z2;
    D n = D(tan_cn[0]) + z2 * D(tan_cn[1]), n2 = D(tan_cn[2]) + z2 * D(tan_cn[3]);
    n += z4 * n2;
    D d = D(tan_cd[0]) + z2 * D(tan_cd[1]), d2 = D(tan_cd[2]) + z2 * D(tan_cd[3]);
    d += z4 * d2;
    n *= z;
    D s0 = D((i & 1) ? 1.0 : 0.0), s1 = D((1 - (i & 1)) ? 1.0 : 0.0);
    D r1 = (n * s1 - d * s0) / (n * s0 + d * s1);
    uint64_t tail = (dbits(r1) + 7) & (~(uint64_t)0 >> 35);
    if (tail <= 14) {
        uint32_t ax = u & (~0u >> 1), sgn = u >> 31;
        for (int j = 0; j < 8; j++)
            if (lf_bits(tan_st[j][0]) == ax) return sgn ? lf_sub(-tan_st[j][1], tan_st[j][2]) : lf_add(tan_st[j][1], tan_st[j][2]);
    }
    return d2f(r1);
}

// ---- atan2f ----------------------------------------------------------------------------------------------------------------
LF_HDI D muldd(D xh, D xl, D ch, D cl, D* l) {
    D ahlh = ch * xl, alhh = cl * xh, ahhh = ch * xh, ahhl = fma(ch, xh, -ahhh);
    ahhl += alhh + ahlh;
    ch = ahhh + ahhl;
    *l = (ahhh - ch) + ahhl;
    return ch;
}
LF_HDI D mulddd(D xh, D xl, D ch, D* l) {
    D ahlh = ch * xl, ahhh = ch * xh, ahhl = fma(ch, xh, -ahhh);
    ahhl += ahlh;
    ch = ahhh + ahhl;
    *l = (ahhh - ch) + ahhl;
    return ch;
}
// c: n pairs (high, low)
LF_HD inline D polydd(D xh, D xl, int n, const double (*c)[2], D* l) {
    int i = n - 1;
    D ch = D(c[i][0]), cl = D(c[i][1]);
    while (--i >= 0) {
        ch = muldd(xh, xl, ch, cl, &cl);
        D th = ch + D(c[i][0]), tl = (D(c[i][0]) - th) + ch;
        ch = th;
        cl += tl + D(c[i][1]);
    }
    *l = cl;
    return ch;
}

LF_TBL double at_cn[] = {0x1p+0, 0x1.40e0698f94c35p+1, 0x1.248c5da347f0dp+1, 0x1.d873386572976p-1, 0x1.46fa40b20f1dp-3,
    0x1.33f5e041eed0fp-7, 0x1.546bbf28667c5p-14};
LF_TBL double at_cd[] = {0x1p+0, 0x1.6b8b143a3f6dap+1, 0x1.8421201d18ed5p+1, 0x1.8221d086914ebp+0, 0x1.670657e3a07bap-2,
    0x1.0f4951fd1e72dp-5, 0x1.b3874b8798286p-11};
LF_TBL double at_c[32][2] = {{0x1p+0, -0x1.8c1dac5492248p-87}, {-0x1.5555555555555p-2, -0x1.55553bf3a2abep-56},
    {0x1.999999999999ap-3, -0x1.99deed1ec9071p-57}, {-0x1.2492492492492p-3, -0x1.fd99c8d18269ap-58},
    {0x1.c71c71c71c717p-4, -0x1.651eee4c4d9dp-61}, {-0x1.745d1745d1649p-4, -0x1.632683d6c44a6p-58},
    {0x1.3b13b13b11c63p-4, 0x1.bf69c1f8af41dp-58}, {-0x1.11111110e6338p-4, 0x1.3c3e431e8bb68p-61},
    {0x1.e1e1e1dc45c4ap-5, -0x1.be2db05c77bbfp-59}, {-0x1.af286b8164b4fp-5, 0x1.a4673491f0942p-61},
    {0x1.86185e9ad4846p-5, 0x1.e12e32d79fceep-59}, {-0x1.642c6d5161faep-5, 0x1.3ce76c1ca03fp-59},
    {0x1.47ad6f277e5bfp-5, -0x1.abd8d85bdb714p-60}, {-0x1.2f64a2ee8896dp-5, 0x1.ef87d4b615323p-61},
    {0x1.1a6a2b31741b5p-5, 0x1.a5d9d973547eep-62}, {-0x1.07fbdad65e0a6p-5, -0x1.65ac07f5d35f4p-61},
    {0x1.ee9932a9a5f8bp-6, 0x1.f8b9623f6f55ap-61}, {-0x1.ce8b5b9584dc6p-6, 0x1.fe5af96e8ea2dp-61},
    {0x1.ac9cb288087b7p-6, -0x1.450cdfceaf5cap-60}, {-0x1.84b025351f3e6p-6, 0x1.579561b0d73dap-61},
    {0x1.52f5b8ecdd52bp-6, 0x1.036bd2c6fba47p-60}, {-0x1.163a8c44909dcp-6, 0x1.18f735ffb9f16p-60},
    {0x1.a400dce3eea6fp-7, -0x1.c90569c0c1b5cp-61}, {-0x1.1caa78ae6db3ap-7, -0x1.4c60f8161ea09p-61},
    {0x1.52672453c0731p-8, 0x1.834efb598c338p-62}, {-0x1.5850c5be137cfp-9, -0x1.445fc150ca7f5p-63},
    {0x1.23eb98d22e1cap-10, -0x1.388fbaf1d783p-64}, {-0x1.8f4e974a40741p-12, 0x1.271198a97da34p-66},
    {0x1.a5cf2e9cf76e5p-14, -0x1.887eb4a63b665p-68}, {-0x1.420c270719e32p-16, 0x1.efd595b27888bp-71},
    {0x1.3ba2d69b51677p-19, -0x1.4fb06829cdfc7p-73}, {-0x1.29b7e6f676385p-23, -0x1.a783b6de718fbp-77}};


// y/x tiny: Taylor z - z^3/3 with z = y/x
LF_HD inline float atan2f_tiny(float y, float x) {
    D dy = f2d(y), dx = f2d(x);
    D z = dy / dx;
    D e = fma(-z, dx, dy);
    D c = D(-0x1.5555555555555p-2);  // -1/3 rounded to nearest
    D zz = z * z;
    D cz = c * z;
    e = e / dx + cz * zz;
    uint64_t t = dbits(z);
    if ((t & 0xfffffffull) == 0) {  // boundary case: avoid the double rounding to binary32
        if ((z * e) > D(0.0)) t += 1; else t -= 1;
    }
    return d2f(dfrom(t));
}

LF_HD inline float atan2f_cr(float y, float x) {
    const double pi = 0x1.921fb54442d18p+1, pi2 = 0x1.921fb54442d18p+0;
    uint32_t ux = lf_bits(x), uy = lf_bits(y), ax = ux & (~0u >> 1), ay = uy & (~0u >> 1);
    D sg_y = D((uy >> 31) ? -1.0 : 1.0);
    if (ay >= (0xffu << 23) || ax >= (0xffu << 23)) {  // x or y is nan or inf
        if (ay > (0xffu << 23)) return nan_f();
        if (ax > (0xffu << 23)) return nan_f();
        uint32_t yinf = ay == (0xffu << 23), xinf = ax == (0xffu << 23);
        if (yinf & xinf) {
            if (ux >> 31) return d2f(D(0x1.2d97c7f3321d2p+1) * sg_y);  // +-3pi/4
            return d2f(D(0x1.921fb54442d18p-1) * sg_y);                  // +-pi/4
        }
        if (xinf) {
            if (ux >> 31) return d2f(D(pi) * sg_y);
            return d2f(D(0.0) * sg_y);
        }
        if (yinf) return d2f(D(pi2) * sg_y);
    }
    if (ay == 0) {
        if (!ax) {
            uint32_t i = (uy >> 31) * 4 + (ux >> 31) * 2;
            if (ux >> 31) return d2f(D(at_off[i]) + D(at_offl[i]));
            return d2f(D(at_off[i]));
        }
        if (!(ux >> 31)) return d2f(D(0.0) * sg_y);
    }
    uint32_t gt = ay > ax, i = (uy >> 31) * 4 + (ux >> 31) * 2 + gt;
    D mg = D(gt ? 1.0 : 0.0), mg1 = D(gt ? 0.0 : 1.0);  // m[gt], m[1-gt]
    D sgt = D(gt ? -1.0 : 1.0);

    D zx = f2d(x), zy = f2d(y);
    D z = (mg * zx + mg1 * zy) / (mg * zy + mg1 * zx);  // x/y if |y| > |x|, y/x otherwise
    D r;
    int d = (int)ax - (int)ay;
    if (d < (27 << 23) && d > (-(27 << 23))) {
        D z2 = z * z, z4 = z2 * z2, z8 = z4 * z4;
        D cn0 = D(at_cn[0]) + z2 * D(at_cn[1]);
        D cn2 = D(at_cn[2]) + z2 * D(at_cn[3]);
        D cn4 = D(at_cn[4]) + z2 * D(at_cn[5]);
        D cn6 = D(at_cn[6]);
        cn0 += z4 * cn2;
        cn4 += z4 * cn6;
        cn0 += z8 * cn4;
        D cd0 = D(at_cd[0]) + z2 * D(at_cd[1]);
        D cd2 = D(at_cd[2]) + z2 * D(at_cd[3]);
        D cd4 = D(at_cd[4]) + z2 * D(at_cd[5]);
        D cd6 = D(at_cd[6]);
        cd0 += z4 * cd2;
        cd4 += z4 * cd6;
        cd0 += z8 * cd4;
        r = cn0 / cd0;
    } else {
        r = D(1.0);
    }
    z *= sgt;
    r = z * r + D(at_off[i]);
    if (((dbits(r) + 8) & 0xfffffffull) <= 16) {
        if (ay < ax && ((ax - ay) >> 23 >= 25)) return atan2f_tiny(y, x);  // tiny y/x
        D zh, zl;
        if (!gt) {
            zh = zy / zx;
            zl = fma(zh, -zx, zy) / zx;
        } else {
            zh = zx / zy;
            zl = fma(zh, -zy, zx) / zy;
        }
        D z2l, z2h = muldd(zh, zl, zh, zl, &z2l);
        D pl, ph = polydd(z2h, z2l, 32, at_c, &pl);
        (void)z2l;
        zh *= sgt;
        zl *= sgt;
        ph = muldd(zh, zl, ph, pl, &pl);
        D sh = ph + D(at_off[i]), sl = ((D(at_off[i]) - sh) + ph) + pl + D(at_offl[i]);
        float rf = d2f(sh);
        D th = f2d(rf), dh = sh - th, tm = dh + sl;
        if ((th + th * D(0x1p-60)) == (th - th * D(0x1p-60))) {
            uint64_t tth = dbits(th) & ((uint64_t)0x7ff << 52);
            tth -= (uint64_t)24 << 52;
            if (dabs(tm) > dfrom(tth)) tm *= D(1.25); else tm *= D(0.75);
        }
        r = th + tm;
    }
    return d2f(r);
}

// ---- powf ------------------------------------------------------------------------------------------------------------------
LF_HDI int ctz32(uint32_t v) { int n = 0; while (!(v & 1)) { v >>= 1; ++n; } return n; }  // v != 0
LF_HDI int clz32(uint32_t v) { int n = 0; while (!(v & 0x80000000u)) { v <<= 1; ++n; } return n; }  // v != 0

LF_HDI int isint(float y0) {
    uint32_t w = lf_bits(y0);
    int ey = (int)((w >> 23) & 0xff) - 127, s = ey + 9;
    if (ey >= 0) {
        if (s >= 32) return 1;
        return !(w << s);
    }
    if (!(w << 1)) return 1;
    return 0;
}
LF_HDI int isodd(float y0) {
    uint32_t w = lf_bits(y0);
    int ey = (int)((w >> 23) & 0xff) - 127, s = ey + 9, odd = 0;
    if (ey >= 0) {
        if (s < 32 && !(w << s)) odd = (w >> (32 - s)) & 1;
        if (s == 32) odd = w & 1;
    }
    return odd;
}
LF_HDI int is_signaling(float x) { return ((lf_bits(x) ^ 0x00400000u) & 0x7fffffffu) > 0x7fc00000u; }

// nonzero if x^y is exact (and representable as a binary32); CORE-MATH's is_exact (the argument is only used in the
// accurate path, where an exact result must not be pushed off a rounding boundary)
LF_HD inline int is_exact(float x, float y) {
    uint32_t v = lf_bits(x), w = lf_bits(y);
    if ((v << 1) != 0x7f000000 && (w << (32 - 16)) != 0) return 0;
    if ((v << 1) == 0x7f000000) return 1;  // |x| = 1
    if (y >= 0 && isint(y)) {
        uint32_t m = v & 0x7fffff;
        int32_t e = (int32_t)((v << 1) >> 24) - 0x96;
        if (e >= -149) m |= 0x800000; else e++;
        int t = ctz32(m);
        m = m >> t;
        e += t;
        if (y == 0 || y == 1) return 1;
        if (m == 1) return -149 <= lf_mul(y, lf_i2f(e)) && lf_mul(y, lf_i2f(e)) < 128;
        if (y < 0 || 15 < y) return 0;
        int y_int = (int)y;
        if (m > pw_xmax[y_int]) return 0;
        uint64_t my = m * m;  // (m^y itself is not needed: only its bit length below)
        for (int i = 2; i < y_int; i++) my = my * m;
        (void)my;
        t = 32 - clz32(m);
        int32_t ez = e * y_int + t;
        if (ez <= -149 || 128 < ez) return 0;
        return e * y_int >= -149;
    }
    uint32_t n = w & 0x7fffff;
    int32_t f = (int32_t)((w << 1) >> 24) - 0x96;
    if (f >= -149) n |= 0x800000; else f++;
    int t = ctz32(n);
    n = n >> t;
    f += t;
    uint32_t m = v & 0x7fffff;
    int32_t e = (int32_t)((v << 1) >> 24) - 0x96;
    if (e >= -149) m |= 0x800000; else e++;
    t = ctz32(m);
    m = m >> t;
    e += t;
    if (y < 0) {
        int32_t ez;
        if (m != 1) return 0;
        if (f >= 0) ez = (int32_t)((uint32_t)((e >= 0) ? -(e << f) : (-e << f)) * n);
        else {
            t = ctz32((uint32_t)e);
            if (-f > t) return 0;
            ez = (int32_t)((uint32_t)(-e >> (-f)) * n);
        }
        return -149 <= ez && ez < 128;
    }
    while (f++) {
        if (e & 1) return 0;
        e = e / 2;
        float dm = lf_u2f(m);
        float rs = lf_sqrt(dm);
        float s = (float)(int)rs;  // roundf of a small positive value
        if (lf_sub(rs, s) >= 0.5f) s = lf_add(s, 1.0f);
        if (lf_mul(s, s) != dm) return 0;
        m = (uint32_t)s;
    }
    if (m > 1) {
        if (15 < n) return 0;
        if (m > pw_xmax[n]) return 0;
    }
    uint32_t my = m, n0 = n;
    while (n0-- > 1) my = my * m;
    t = 32 - clz32(my);
    return -149 <= e * (int)n && e * (int)n + t <= 128;
}

LF_TBL double pw_ix[] = {
    0x1p+0, 0x1.f07c1f07cp-1, 0x1.e1e1e1e1ep-1, 0x1.d41d41d42p-1,
    0x1.c71c71c72p-1, 0x1.bacf914c2p-1, 0x1.af286bca2p-1, 0x1.a41a41a42p-1,
    0x1.99999999ap-1, 0x1.8f9c18f9cp-1, 0x1.861861862p-1, 0x1.7d05f417dp-1,
    0x1.745d1745dp-1, 0x1.6c16c16c1p-1, 0x1.642c8590bp-1, 0x1.5c9882b93p-1,
    0x1.555555555p-1, 0x1.4e5e0a72fp-1, 0x1.47ae147aep-1, 0x1.414141414p-1,
    0x1.3b13b13b1p-1, 0x1.3521cfb2bp-1, 0x1.2f684bda1p-1, 0x1.29e4129e4p-1,
    0x1.249249249p-1, 0x1.1f7047dc1p-1, 0x1.1a7b9611ap-1, 0x1.15b1e5f75p-1,
    0x1.111111111p-1, 0x1.0c9714fbdp-1, 0x1.084210842p-1, 0x1.041041041p-1, 0x1p-1
    };
LF_TBL double pw_lix[][2] = {
    {0x0p+0, 0x0p+0}, {-0x1.6cp-5, 0x1.4b229b87f3f89p-15},
    {-0x1.66p-4, -0x1.fb7d654235799p-15}, {-0x1.08p-3, -0x1.8b119b2c9c87bp-12},
    {-0x1.5cp-3, -0x1.a39fa6533294dp-19}, {-0x1.acp-3, -0x1.ebc5b663dd4b8p-12},
    {-0x1.fcp-3, 0x1.f4a37fe0fa46fp-14}, {-0x1.24p-2, -0x1.01eac33103e6bp-12},
    {-0x1.4ap-2, 0x1.61ed0d15725ep-12}, {-0x1.6ep-2, -0x1.10e6ceb499ba9p-13},
    {-0x1.92p-2, 0x1.115db8ada837dp-12}, {-0x1.b4p-2, -0x1.fafdce266d7aep-12},
    {-0x1.d6p-2, -0x1.d4f80cd19906fp-12}, {-0x1.f8p-2, 0x1.5ea5ccd0a7396p-12},
    {0x1.e8p-2, -0x1.0500d67fe62ebp-13}, {0x1.c8p-2, 0x1.9dc2d41aa4626p-14},
    {0x1.a8p-2, 0x1.ff2e2ff321344p-11}, {0x1.8ap-2, 0x1.130157f4c3a3ep-11},
    {0x1.6cp-2, 0x1.61ed0cad929ccp-11}, {0x1.5p-2, -0x1.2089a632d7949p-11},
    {0x1.32p-2, 0x1.7fdc6dfb2d21ap-11}, {0x1.16p-2, 0x1.380a6c36088f3p-11},
    {0x1.f6p-3, -0x1.3ab7dc7ba81acp-18}, {0x1.cp-3, -0x1.cc2c0061ef1a2p-14},
    {0x1.8ap-3, 0x1.130157c97bbep-12}, {0x1.56p-3, 0x1.ee14ff34c4128p-14},
    {0x1.22p-3, 0x1.b5b854c4fde69p-12}, {0x1.ep-4, 0x1.635d1df7cb0b5p-13},
    {0x1.7ep-4, -0x1.3f6d2636c101ep-13}, {0x1.1cp-4, -0x1.33567f1b193a4p-14},
    {0x1.78p-5, -0x1.8d66c5313a71dp-14}, {0x1.74p-6, 0x1.f7430ee200ep-17}, {0x0p+0, 0x0p+0}
    };
LF_TBL double pw_c[] = {0x1.71547652b82fep+0, -0x1.71547652b82fep-1, 0x1.ec709dc3a2d0bp-2, -0x1.71547652bc4a9p-2,
    0x1.2776c441b72ep-2, -0x1.ec709bdf453ecp-3, 0x1.a6406efd4b877p-3, -0x1.717d824a520f7p-3};
LF_TBL double pw_ce[] = {0x1.62e42fefa398bp-5, 0x1.ebfbdff84555ap-11, 0x1.c6b08d4ad86d3p-17,
    0x1.3b2ad1b1716a2p-23, 0x1.5d7472718ce9dp-30, 0x1.4a1d7f457ac56p-37};
LF_TBL double pw_tb[] = {0x1p+0, 0x1.0b5586cf9890fp+0, 0x1.172b83c7d517bp+0, 0x1.2387a6e756238p+0,
    0x1.306fe0a31b715p+0, 0x1.3dea64c123422p+0, 0x1.4bfdad5362a27p+0, 0x1.5ab07dd485429p+0,
    0x1.6a09e667f3bcdp+0, 0x1.7a11473eb0187p+0, 0x1.8ace5422aa0dbp+0, 0x1.9c49182a3f09p+0,
    0x1.ae89f995ad3adp+0, 0x1.c199bdd85529cp+0, 0x1.d5818dcfba487p+0, 0x1.ea4afa2a490dap+0};
LF_TBL double pw_ch[][2] = {{0x1.71547652b82fep+1, 0x1.777d0ffda2b89p-55}, {0x1.ec709dc3a03fdp-1, 0x1.d27f04ff73b3ap-55},
    {0x1.2776c50ef9bfep-1, 0x1.e4b514251d0ecp-55}, {0x1.a61762a7aded9p-2, 0x1.de632dc7f6998p-57},
    {0x1.484b13d7c02aep-2, 0x1.a320ec342ddb3p-56}, {0x1.0c9a84993fd48p-2, -0x1.e6425ce9a74a4p-57},
    {0x1.c68f568d8beafp-3, -0x1.03a175487feabp-57}, {0x1.89f3b14657dfbp-3, 0x1.f04a3acf0bcf7p-57},
    {0x1.5b9ad2f2d12ap-3, -0x1.68fdff6815a6fp-58}, {0x1.3702165b88acbp-3, 0x1.45b052ace6c8ep-60},
    {0x1.1998f60f2f005p-3, -0x1.79a94f62fb524p-57}, {0x1.f9bc428e30809p-4, -0x1.51f063387e47p-59},
    {0x1.1ac0ab871296ap-3, 0x1.2ba6a2e1a625bp-57}
    };
LF_TBL double pw_ce2[][2] = {{0x1p+0, 0x1.f7d70599926c4p-98}, {0x1.62e42fefa39efp-1, 0x1.abc9e3b39856bp-56},
    {0x1.ebfbdff82c58fp-3, -0x1.5e43a540c283dp-57}, {0x1.c6b08d704a0cp-5, -0x1.d3316277451e6p-59},
    {0x1.3b2ab6fba4e77p-7, 0x1.4e66003ba7f85p-62}, {0x1.5d87fe78a6731p-10, 0x1.07183d46a9697p-66},
    {0x1.430912f86c787p-13, 0x1.bc81afca4c93p-67}, {0x1.ffcbfc588b0c7p-17, -0x1.e63f6f0116f4cp-71},
    {0x1.62c0223a5c826p-20, -0x1.30542d98ea4a5p-74}, {0x1.b5253d395e7c6p-24, -0x1.9285a132ce05ep-80},
    {0x1.e4cf5158b7b01p-28, -0x1.9ac1facae1b88p-83}, {0x1.e8cac7351a7a8p-32, -0x1.4fb82adebd76bp-91},
    {0x1.c3bd65182746dp-36, 0x1.84ad0689d30ep-91}, {0x1.8161931d765c3p-40, -0x1.254c6535279cep-95},
    {0x1.314943a26c9e2p-44, -0x1.f4f2fdc14fb82p-98}, {0x1.c36e53b459602p-49, 0x1.f0d06a5a63c41p-103},
    {0x1.397637b3876a4p-53, -0x1.5632c551ae458p-107}, {0x1.98fbfefdddb51p-58, -0x1.fd134923d52b4p-115}};


LF_HD inline float powf_accurate2(float x0, float y0, int is_exact_) {
    D x = f2d(x0), y = f2d(y0);
    uint64_t t = dbits(x);
    int e = (int)((t >> 52) & 0x7ff) - 0x3ff, xsgn = (int)(t >> 63);
    t &= ~(uint64_t)0 >> 12;
    int k = t > 0x6a09e667f3bcdull;
    e += k;
    t |= (uint64_t)0x3ff << 52;
    x = dfrom(t);
    D ok = D(k ? 2.0 : 1.0);
    D xm = x - ok, xp = x + ok, zh = xm / xp, zl = fma(zh, -xp, xm) / xp;
    D z2l, z2h = muldd(zh, zl, zh, zl, &z2l);
    z2h = polydd(z2h, z2l, 13, pw_ch, &z2l);
    zh = muldd(zh, zl, z2h, z2l, &zl);
    zh = mulddd(zh, zl, y, &zl);
    D ey = i2d(e) * y, eh = ey + zh, el = ((ey - eh) + zh) + zl, ee = roundeven_finite(eh);
    eh -= ee;
    eh = polydd(eh, el, 18, pw_ce2, &el);
    uint64_t r = ((uint64_t)0x3ff + (uint64_t)d2ll(ee)) << 52;
    uint64_t lh = dbits(eh);
    D ll = el;
    if ((!is_exact_ && (lh & 0xfffffff) == 0) || (is_exact_ && ((lh + 1) & 0xfffffff) <= 2)) {
        if (dabs(ll) > D(0x1p-91)) {
            if (el < D(0.0)) lh--; else lh++;
            eh = dfrom(lh);
        }
    }
    eh *= dfrom(r);
    if (xsgn && isodd(y0)) eh = -eh;
    return d2f(eh);
}

LF_HD inline float powf_cr(float x0, float y0) {
    D x = f2d(x0), y = f2d(y0);
    uint64_t tx = dbits(x), ty = dbits(y);
    int xsgn = 0;
    if ((tx << 1) == ((uint64_t)0x3ff << 53)) {  // |x| = 1
        if (tx >> 63) {                          // x = -1
            if ((ty << 1) > ((uint64_t)0x7ff << 53)) return nan_f();  // y = nan
            if (isint(y0)) return isodd(y0) ? x0 : -x0;
            return nan_f();
        }
        return is_signaling(y0) ? nan_f() : x0;  // 1^y = 1
    }
    if ((ty << 1) == 0) return is_signaling(x0) ? nan_f() : 1.0f;  // x^0 = 1
    if (ty == ((uint64_t)0x3ff << 52)) return is_signaling(x0) ? nan_f() : x0;  // x^1 = x
    if ((ty << 1) >= ((uint64_t)0x7ff << 53)) {  // y = Inf/NaN
        if ((tx << 1) > ((uint64_t)0x7ff << 53)) return nan_f();  // x = NaN
        if ((ty << 1) == ((uint64_t)0x7ff << 53)) {
            if (((tx << 1) < ((uint64_t)0x3ff << 53)) ^ (ty >> 63)) return 0.0f;
            return lf_from_bits(0x7f800000u);
        }
        return nan_f();
    }
    if (tx >= ((uint64_t)0x7ff << 52)) {  // x is Inf, NaN or less than 0
        if ((tx << 1) == ((uint64_t)0x7ff << 53)) {  // x is +Inf or -Inf
            if (!isodd(y0)) x0 = fabs_f(x0);
            if (ty >> 63) return lf_div(1.0f, x0);
            return x0;
        }
        if ((tx << 1) > ((uint64_t)0x7ff << 53)) return nan_f();  // x is NaN
        if (tx > ((uint64_t)0x7ff << 52)) {  // x <= 0
            xsgn = 1;
            if (!isint(y0) && x.v != 0.0) return nan_f();
        }
    }
    if (!(tx << 1)) {      // x = +0 or -0
        if (ty >> 63) {    // y < 0
            if (isodd(y0)) return lf_div(1.0f, copysign_f(0.0f, x0));
            return lf_div(1.0f, 0.0f);
        }
        if (isodd(y0)) return lf_mul(copysign_f(1.0f, x0), 0.0f);
        return 0.0f;
    }
    uint64_t m = tx & ~(uint64_t)0 >> 12;
    int e = (int)((tx >> 52) & 0x7ff) - 0x3ff;
    int j = (int)((m + ((uint64_t)1 << (52 - 6))) >> (52 - 5)), k = j > 13;
    e += k;
    D xd = dfrom(m | (uint64_t)0x3ff << 52);
    D z = fma(xd, D(pw_ix[j]), D(-1.0));
    D z2 = z * z, z4 = z2 * z2;
    D c6 = D(pw_c[6]) + z * D(pw_c[7]);
    D c4 = D(pw_c[4]) + z * D(pw_c[5]);
    D c2 = D(pw_c[2]) + z * D(pw_c[3]);
    D c0 = D(pw_c[0]) + z * D(pw_c[1]);
    c0 += z2 * c2;
    c4 += z2 * c6;
    c0 += z4 * c4;
    D l = z * c0 - D(pw_lix[j][1]);
    y *= D(16.0);
    D zt = (i2d(e) - D(pw_lix[j][0])) * y;
    z = l * y + zt;
    if (z > D(2048.0)) {  // overflow
        if (isodd(y0)) return lf_mul(copysign_f(0x1p127f, x0), 0x1p127f);
        return lf_mul(0x1p127f, 0x1p127f);
    }
    if (z < D(-2400.0)) {  // underflow
        if (isodd(y0)) return lf_mul(copysign_f(0x1p-126f, x0), 0x1p-126f);
        return lf_mul(0x1p-126f, 0x1p-126f);
    }
    if (dabs(z) < D(0x1p-26)) return d2f(D(1.0) + z);
    D ia = dfloor(z), h = fma(l, y, zt - ia);
    int64_t il = d2ll(ia), jl = il & 0xf, el = il - jl;
    el >>= 4;
    D s = D(pw_tb[jl]);
    s = s * dfrom(((uint64_t)el + (uint64_t)0x3ff) << 52);
    D h2 = h * h;
    c0 = D(pw_ce[0]) + h * D(pw_ce[1]);
    c2 = D(pw_ce[2]) + h * D(pw_ce[3]);
    c4 = D(pw_ce[4]) + h * D(pw_ce[5]);
    c0 += h2 * (c2 + h2 * c4);
    D w = s * h;
    D rr = s + w * c0;
    const uint64_t offs = 468;
    if (((dbits(rr) + offs) & 0xfffffff) <= 2 * offs) return powf_accurate2(x0, y0, is_exact(x0, y0));
    if (xsgn && isodd(y0)) rr = -rr;
    return d2f(rr);
}


// ---- binary64 hypot (CORE-MATH cr_hypot, correctly rounded; round to nearest) and atan2 (double-double, not correctly rounded) ---
// Both serve the auto-flip roll sign (car_pre_tick.cuh), whose inputs are binary32 values widened to double.
LF_HDI int clz64(uint64_t v) { int n = 0; while (!(v >> 63)) { v <<= 1; ++n; } return n; }  // v != 0

// both operands subnormal: a, b are their encodings (x >= y > 0, below 2^52)
LF_HD inline D hypot_denorm(uint64_t a, uint64_t b) {
    D af = ll2d((int64_t)a), bf = ll2d((int64_t)b);
    a <<= 1;
    b <<= 1;
    uint64_t rm = (uint64_t)d2ll(dsqrt(af * af + bf * bf));
    int64_t tm = (int64_t)(rm << 1);
    int64_t Dd = (int64_t)(a * a + b * b - (uint64_t)tm * (uint64_t)tm);
    while (Dd > 2 * tm) { Dd -= 2 * tm + 1; tm++; }
    while (Dd < 0) { Dd += 2 * tm - 1; tm--; }
    int rb = (int)(tm & 1);  // round bit of rm
    int sb = Dd != 0;        // sticky bit (the midpoint case cannot occur in the subnormal range)
    rm = (uint64_t)tm >> 1;
    if (sb) rm += (uint64_t)rb;
    return dfrom(rm);
}

// the exact integer refinement for the hard cases; x >= y > 0 scaled so that x has exponent 0x3ff
LF_HD inline D hypot_hard(D x, D y) {
    uint64_t xi = dbits(x), yi = dbits(y);
    uint64_t bm = (xi & (~0ull >> 12)) | 1ull << 52;
    uint64_t lm = (yi & (~0ull >> 12)) | 1ull << 52;
    int be = (int)(xi >> 52), le = (int)(yi >> 52);
    uint64_t ri = dbits(dsqrt(x * x + y * y));
    const int bs = 2;
    uint64_t rm = ri & (~0ull >> 12);
    int re = (int)(ri >> 52) - 0x3ff;
    rm |= 1ull << 52;
    for (int i = 0; i < 3; i++) {
        if (rm == 1ull << 52) { rm = ~0ull >> 11; re--; } else rm--;
    }
    bm <<= bs;
    uint64_t m2 = bm * bm;
    int de = be - le, ls = bs - de;
    if (ls >= 0) {
        lm <<= ls;
        m2 += lm * lm;
    } else {
        // lm * lm as 128 bits (lm < 2^53); m2 += lm2 >> n with a sticky bit for the n bits shifted out, 0 < n <= 54
        uint64_t hi, lo;
        {
            uint64_t a0 = lm & 0xFFFFFFFFull, a1 = lm >> 32;
            uint64_t p0 = a0 * a0, p1 = a0 * a1, p2 = a1 * a1;
            uint64_t mid = (p0 >> 32) + (p1 & 0xFFFFFFFFull) + (p1 & 0xFFFFFFFFull);
            lo = (p0 & 0xFFFFFFFFull) | (mid << 32);
            hi = p2 + (p1 >> 32) + (p1 >> 32) + (mid >> 32);
        }
        int n = -2 * ls;
        m2 += (hi << (64 - n)) | (lo >> n);
        m2 |= (lo & ((1ull << n) - 1)) != 0 ? 1ull : 0ull;
    }
    int k = bs + re;
    int64_t Dd;
    do {
        rm += 1 + (rm >= (1ull << 53) ? 1u : 0u);
        uint64_t tm = rm << k, rm2 = tm * tm;
        Dd = (int64_t)(m2 - rm2);
    } while (Dd > 0);
    if (Dd != 0) {  // not exact (round to nearest)
        uint64_t tm = (rm << k) - (uint64_t)(1 << (k - (rm <= (1ull << 53) ? 1 : 0)));
        Dd = (int64_t)(m2 - tm * tm);
        if (Dd != 0) rm += (uint64_t)(Dd >> 63);
        else rm -= rm & (uint64_t)(1 << (1 - (rm <= (1ull << 53) ? 1 : 0)));
    }
    if (rm >= (1ull << 53)) { rm >>= 1; re++; }
    int64_t e = (int64_t)be - 1 + re;
    return dfrom(((uint64_t)e << 52) + rm);
}

// sqrt(x^2 + y^2) correctly rounded (CORE-MATH cr_hypot), no flags; NaN is the canonical one, an infinity wins over a NaN
LF_HD inline double hypot64_cr(double x0, double y0) {
    D x = D(x0), y = D(y0);
    uint64_t xi = dbits(x), yi = dbits(y);
    const uint64_t emsk = 0x7ffull << 52;
    uint64_t ex = xi & emsk, ey = yi & emsk;
    x = dabs(x);
    y = dabs(y);
    if (ex == emsk || ey == emsk) {
        bool xinf = (xi << 1) == (emsk << 1), yinf = (yi << 1) == (emsk << 1);
        if (xinf || yinf) return dfrom(emsk).v;
        return dfrom(0x7ff8000000000000ull).v;
    }
    D u = x > y ? x : y, v = x > y ? y : x;
    D xd = u, yd = v;
    ey = dbits(yd);
    if (!(ey >> 52)) {  // y is subnormal
        if (!ey) return xd.v;
        ex = dbits(xd);
        if (!(ex >> 52)) return hypot_denorm(ex, ey).v;
        int nz = clz64(ey);
        ey <<= nz - 11;
        ey &= ~0ull >> 12;
        ey -= (uint64_t)(nz - 12) << 52;
        yd = dfrom(ey);
    }
    uint64_t de = dbits(xd) - dbits(yd);
    if (de > (27ull << 52)) return fma(D(0x1p-27), v, u).v;
    int64_t off = (int64_t)((0x3ffull << 52) - (dbits(xd) & emsk));
    xd = dfrom(dbits(xd) + (uint64_t)off);
    yd = dfrom(dbits(yd) + (uint64_t)off);
    x = xd;
    y = yd;
    D x2 = x * x, dx2 = fma(x, x, -x2);
    D y2 = y * y, dy2 = fma(y, y, -y2);
    D r2 = x2 + y2, ir2 = D(0.5) / r2, dr2 = ((x2 - r2) + y2) + (dx2 + dy2);
    D th = dsqrt(r2), rsqrt = th * ir2;
    D dz = dr2 - fma(th, th, -r2), tl = rsqrt * dz;
    {  // fasttwosum(th, tl)
        D s = th + tl, z = s - th;
        tl = tl - z;
        th = s;
    }
    uint64_t thu = dbits(th), tlu = dbits(dabs(tl));
    ex = thu & emsk;
    ey = tlu;
    uint64_t aidr = ey + (0x3feull << 52) - ex;
    uint64_t mid = (aidr - 0x3c90000000000000ull + 16) >> 5;
    uint64_t midm = (aidr - 0x3c80000000000000ull + 16) >> 5;
    if (mid == 0 || midm == 0 || aidr < 0x39b0000000000000ull || aidr > 0x3c9fffffffffff80ull) thu = dbits(hypot_hard(x, y));
    thu -= (uint64_t)off;
    if (thu >= (0x7ffull << 52)) return dfrom(emsk).v;  // >= 2^1024
    return dfrom(thu).v;
}

// atan2(y, x) in double-double: the 32-term polynomial of atan2f's accurate path (about 2^-80 relative at |y/x| = 1, better
// below) on the exact double-double quotient, one rounding at the end. Deterministic on host and device; not CORE-MATH's
// correctly rounded atan2 (192-bit arithmetic, ~1500 lines): measured rate of results one ulp off.
LF_HD inline double atan2_64_cr(double y0, double x0) {
    D y = D(y0), x = D(x0);
    uint64_t uy = dbits(y), ux = dbits(x), ay = uy & 0x7FFFFFFFFFFFFFFFull, ax = ux & 0x7FFFFFFFFFFFFFFFull;
    const uint64_t INF = 0x7ffull << 52;
    D sg_y = D((uy >> 63) ? -1.0 : 1.0);
    const D PI_H = D(0x1.921fb54442d18p+1), PI_L = D(0x1.1a62633145c07p-53);
    const D PI2_H = D(0x1.921fb54442d18p+0), PI2_L = D(0x1.1a62633145c07p-54);
    if (ay >= INF || ax >= INF) {
        if (ay > INF || ax > INF) return dfrom(0x7ff8000000000000ull).v;
        bool yinf = ay == INF, xinf = ax == INF;
        if (yinf && xinf) {
            if (ux >> 63) return (sg_y * D(0x1.2d97c7f3321d2p+1) + sg_y * D(0x1p-54)).v;  // +-3pi/4
            return (sg_y * D(0x1.921fb54442d18p-1) + sg_y * D(0x1p-55)).v;                  // +-pi/4
        }
        if (xinf) {
            if (ux >> 63) return (sg_y * PI_H + sg_y * PI_L).v;
            return (D(0.0) * sg_y).v;
        }
        return (sg_y * PI2_H + sg_y * PI2_L).v;
    }
    if (ay == 0) {
        if (ax == 0) {
            uint32_t i = (uint32_t)(uy >> 63) * 4 + (uint32_t)(ux >> 63) * 2;
            if (ux >> 63) return (D(at_off[i]) + D(at_offl[i])).v;
            return at_off[i];
        }
        if (!(ux >> 63)) return (D(0.0) * sg_y).v;
    }
    uint32_t gt = ay > ax, i = (uint32_t)(uy >> 63) * 4 + (uint32_t)(ux >> 63) * 2 + gt;
    D sgt = D(gt ? -1.0 : 1.0);
    // y/x tiny with x > 0 (the offset is a zero): atan(z) = z(1 - z^2/3 ...) is z to more than 600 bits, and the low parts of
    // the double-double products would underflow; the correctly rounded quotient (which may be subnormal) is the result
    if (!gt && !(ux >> 63) && dabs(y / x) < D(0x1p-300)) return (y / x).v;
    {  // atan2 is scale-invariant: bring a huge or tiny pair to the unit range (exact power-of-two scalings), where the
       // Veltkamp splits of the double-double products neither overflow nor lose the low parts to the subnormals
        int e = (int)((gt ? ay : ax) >> 52) - 1023;
        if (e > 400 || e < -400) {
            int h = -(e / 2);
            D p = dfrom((uint64_t)(1023 + h) << 52), q = dfrom((uint64_t)(1023 + (-e - h)) << 52);
            y = y * p * q;
            x = x * p * q;
        }
    }
    D zh, zl;
    if (!gt) {
        zh = y / x;
        zl = fma(zh, -x, y) / x;
    } else {
        zh = x / y;
        zl = fma(zh, -y, x) / y;
    }
    D z2l, z2h = muldd(zh, zl, zh, zl, &z2l);
    D pl, ph = polydd(z2h, z2l, 32, at_c, &pl);
    zh *= sgt;
    zl *= sgt;
    ph = muldd(zh, zl, ph, pl, &pl);
    D sh = ph + D(at_off[i]), sl = ((D(at_off[i]) - sh) + ph) + pl + D(at_offl[i]);
    return (sh + sl).v;
}

}  // namespace lfcr
