//! The vector math: every glam operation V3 uses, ported in `cuda/common/glam.cuh`, against
//! glam 0.34.1 itself on the CPU (the shim, MSVC or gcc built with contraction off) and on the GPU (the strict-float
//! kernels), bit for bit, on random inputs across binades, signs, zeros, denormals, overflow and the specials.
//!
//! A mismatch is "exact" unless both words are NaN: a NaN's sign and payload follow the hardware (x86 makes the
//! negative default NaN, the GPU the positive one, and operand order picks the propagated payload), so NaN against NaN
//! is counted apart as `nan_only` and never as a failure.

use glam::{Affine3A, BVec3A, BVec4A, Mat3A, Quat, Vec3A, Vec4};
use rocketsim_cuda::{Op, SIMTEST_FATBIN, ops, run_host, shim_may_fuse};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

const CHUNK: usize = 1 << 20;
const CASES: usize = 10 * CHUNK; // 10.5M per op

// ---- inputs -------------------------------------------------------------------------------------------------------

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

const NICE: [f32; 16] = [1.0, -1.0, 0.5, 2.0, 3.0, -0.25, 1e-3, 100.0, 1e-20, 1e20, 0.1, 2147483648.0, 4294967296.0, 18446744073709551616.0, 2147483520.0, 16777217.0];
const SPECIAL: [u32; 8] = [0x7F80_0000, 0xFF80_0000, 0x7FC0_0000, 0xFFC0_0001, 0x7F7F_FFFF, 0x0080_0000, 0x3400_0000, 0x0000_0001];

/// One lane. `special` admits inf and NaN.
fn lane(r: &mut Rng, prev: f32, special: bool) -> f32 {
    let x = r.next();
    let sel = (x % 100) as u32;
    let mant = ((x >> 8) as u32) & 0x007F_FFFF;
    let sign = (((x >> 40) & 1) as u32) << 31;
    let pick = |lo: u32, hi: u32| lo + ((x >> 20) as u32) % (hi - lo + 1);
    let bits = match sel {
        0..38 => sign | (pick(127 - 30, 127 + 30) << 23) | mant,
        38..58 => sign | (pick(1, 254) << 23) | mant,
        58..66 => sign | (pick(248, 254) << 23) | mant,
        66..74 => sign | (pick(0, 3) << 23) | mant, // denormals and the smallest normals
        74..80 => sign,                              // +-0
        80..86 => return NICE[(x >> 24) as usize % NICE.len()] * if sign != 0 { -1.0 } else { 1.0 },
        86..94 => {
            if special {
                SPECIAL[(x >> 24) as usize % SPECIAL.len()]
            } else {
                sign | (pick(100, 150) << 23) | mant
            }
        }
        _ => return if sign != 0 { -prev } else { prev }, // ties and exact cancellations
    };
    f32::from_bits(bits)
}

fn gen_case(r: &mut Rng, out: &mut [f32]) {
    let c = r.next();
    let special = c % 100 < 6;
    if (c >> 8) % 100 < 45 {
        // clustered binades: the lanes of one vector are of similar size, so lengths and dots are not dominated
        let base = 30 + ((c >> 16) % 200) as u32;
        for v in out.iter_mut() {
            let x = r.next();
            let e = base + (x % 7) as u32 - 3;
            *v = f32::from_bits((((x >> 8) & 1) as u32) << 31 | (e << 23) | ((x >> 16) as u32 & 0x007F_FFFF));
        }
    } else {
        let mut prev = 1.0;
        for v in out.iter_mut() {
            prev = lane(r, prev, special);
            *v = prev;
        }
    }
}

fn gen_chunk(op: &Op, chunk: usize) -> Vec<f32> {
    let mut r = Rng(((op.index as u64) << 32) ^ (chunk as u64).wrapping_mul(0xA24B_AED4_963E_E407) ^ 0x1234_5678);
    let mut input = vec![0.0f32; CHUNK * op.in_n];
    for case in input.chunks_exact_mut(op.in_n) {
        gen_case(&mut r, case);
        // integer-to-float ops read their lanes as integers: give them every width, so every rounding position is hit
        if matches!(op.name.as_str(), "i2f" | "u2f" | "ull2f") {
            for v in case.iter_mut() {
                let x = r.next();
                *v = f32::from_bits((x as u32) >> (x >> 59) as u32);
            }
        }
    }
    input
}

// ---- the glam side ------------------------------------------------------------------------------------------------

fn v3(i: &[f32]) -> Vec3A {
    Vec3A::new(i[0], i[1], i[2])
}
fn v4(i: &[f32]) -> Vec4 {
    Vec4::new(i[0], i[1], i[2], i[3])
}
fn qu(i: &[f32]) -> Quat {
    Quat::from_xyzw(i[0], i[1], i[2], i[3])
}
fn m3(i: &[f32]) -> Mat3A {
    Mat3A::from_cols(v3(i), v3(&i[3..]), v3(&i[6..]))
}
fn af(i: &[f32]) -> Affine3A {
    Affine3A { matrix3: m3(i), translation: v3(&i[9..]) }
}
fn s3(o: &mut [u32], v: Vec3A) {
    o[..3].copy_from_slice(&[v.x.to_bits(), v.y.to_bits(), v.z.to_bits()]);
}
fn s4(o: &mut [u32], v: Vec4) {
    o[..4].copy_from_slice(&[v.x.to_bits(), v.y.to_bits(), v.z.to_bits(), v.w.to_bits()]);
}
fn sq(o: &mut [u32], q: Quat) {
    o[..4].copy_from_slice(&[q.x.to_bits(), q.y.to_bits(), q.z.to_bits(), q.w.to_bits()]);
}
fn sm(o: &mut [u32], m: Mat3A) {
    s3(o, m.x_axis);
    s3(&mut o[3..], m.y_axis);
    s3(&mut o[6..], m.z_axis);
}
fn sa(o: &mut [u32], a: Affine3A) {
    sm(o, a.matrix3);
    s3(&mut o[9..], a.translation);
}
fn b3(o: &mut [u32], m: BVec3A) {
    let a: [bool; 3] = m.into();
    for k in 0..3 {
        o[k] = if a[k] { !0 } else { 0 };
    }
}
fn b4(o: &mut [u32], m: BVec4A) {
    let a: [bool; 4] = m.into();
    for k in 0..4 {
        o[k] = if a[k] { !0 } else { 0 };
    }
}
fn sf(o: &mut [u32], v: f32) {
    o[0] = v.to_bits();
}
fn tie0(a: f32, b: f32) -> bool {
    a == 0.0 && b == 0.0 && a.to_bits() != b.to_bits()
}
/// V3's `AffineExt` (bullet/linear_math/affine_ext.rs), verbatim.
fn aff_transpose(a: &Affine3A) -> Affine3A {
    let matrix3 = a.matrix3.transpose();
    Affine3A { matrix3, translation: matrix3 * -a.translation }
}

/// glam 0.34.1 on the CPU, one op over a chunk. Panics on an op it does not know, so the table and this stay in step.
fn glam_run(op: &Op, input: &[f32], out: &mut [u32]) {
    let (n, m) = (op.in_n, op.out_n);
    macro_rules! each {
        (|$i:ident, $o:ident| $body:expr) => {
            for (c, $o) in out.chunks_exact_mut(m).enumerate() {
                let $i = &input[c * n..(c + 1) * n];
                $body;
            }
        };
    }
    match op.name.as_str() {
        "rs_min" => each!(|i, o| if tie0(i[0], i[1]) { o[0] = 0xDEADBEEF } else { sf(o, i[0].min(i[1])) }),
        "rs_max" => each!(|i, o| if tie0(i[0], i[1]) { o[0] = 0xDEADBEEF } else { sf(o, i[0].max(i[1])) }),
        "rs_clamp" => each!(|i, o| if !(i[1] <= i[2]) { o[0] = 0xDEADBEEF } else { sf(o, i[0].clamp(i[1], i[2])) }),
        "rs_signum" => each!(|i, o| sf(o, i[0].signum())),
        "f2i" => each!(|i, o| o[0] = i[0] as i32 as u32),
        "f2u" => each!(|i, o| o[0] = i[0] as u32),
        "f2ull" => each!(|i, o| {
            let v = i[0] as u64;
            o[0] = v as u32;
            o[1] = (v >> 32) as u32;
        }),
        "i2f" => each!(|i, o| sf(o, (i[0].to_bits() as i32) as f32)),
        "u2f" => each!(|i, o| sf(o, i[0].to_bits() as f32)),
        "ull2f" => each!(|i, o| sf(o, (((i[1].to_bits() as u64) << 32) | i[0].to_bits() as u64) as f32)),
        "v3_add" => each!(|i, o| s3(o, v3(i) + v3(&i[3..]))),
        "v3_sub" => each!(|i, o| s3(o, v3(i) - v3(&i[3..]))),
        "v3_mul" => each!(|i, o| s3(o, v3(i) * v3(&i[3..]))),
        "v3_div" => each!(|i, o| s3(o, v3(i) / v3(&i[3..]))),
        "v3_mul_s" => each!(|i, o| s3(o, v3(i) * i[3])),
        "s_mul_v3" => each!(|i, o| s3(o, i[3] * v3(i))),
        "v3_div_s" => each!(|i, o| s3(o, v3(i) / i[3])),
        "s_div_v3" => each!(|i, o| s3(o, i[3] / v3(i))),
        "v3_neg" => each!(|i, o| s3(o, -v3(i))),
        "v3_dot" => each!(|i, o| sf(o, v3(i).dot(v3(&i[3..])))),
        "v3_cross" => each!(|i, o| s3(o, v3(i).cross(v3(&i[3..])))),
        "v3_length" => each!(|i, o| sf(o, v3(i).length())),
        "v3_length_squared" => each!(|i, o| sf(o, v3(i).length_squared())),
        "v3_length_recip" => each!(|i, o| sf(o, v3(i).length_recip())),
        "v3_normalize" => each!(|i, o| s3(o, v3(i).normalize())),
        "v3_normalize_or_zero" => each!(|i, o| s3(o, v3(i).normalize_or_zero())),
        "v3_normalize_or" => each!(|i, o| s3(o, v3(i).normalize_or(v3(&i[3..])))),
        "v3_try_normalize" => each!(|i, o| match v3(i).try_normalize() {
            Some(v) => {
                o[3] = 1;
                s3(o, v);
            }
            None => {
                o[3] = 0;
                s3(o, Vec3A::ZERO);
            }
        }),
        "v3_distance_squared" => each!(|i, o| sf(o, v3(i).distance_squared(v3(&i[3..])))),
        "v3_lerp" => each!(|i, o| s3(o, v3(i).lerp(v3(&i[3..]), i[6]))),
        "v3_abs" => each!(|i, o| s3(o, v3(i).abs())),
        "v3_signum" => each!(|i, o| s3(o, v3(i).signum())),
        "v3_min" => each!(|i, o| s3(o, v3(i).min(v3(&i[3..])))),
        "v3_max" => each!(|i, o| s3(o, v3(i).max(v3(&i[3..])))),
        "v3_clamp" => each!(|i, o| s3(o, v3(i).clamp(v3(&i[3..]), v3(&i[6..])))),
        "v3_min_element" => each!(|i, o| sf(o, v3(i).min_element())),
        "v3_max_element" => each!(|i, o| sf(o, v3(i).max_element())),
        "v3_min_position" => each!(|i, o| o[0] = v3(i).min_position() as u32),
        "v3_max_position" => each!(|i, o| o[0] = v3(i).max_position() as u32),
        "v3_cmpeq" => each!(|i, o| b3(o, v3(i).cmpeq(v3(&i[3..])))),
        "v3_cmpge" => each!(|i, o| b3(o, v3(i).cmpge(v3(&i[3..])))),
        "v3_cmpgt" => each!(|i, o| b3(o, v3(i).cmpgt(v3(&i[3..])))),
        "v3_cmple" => each!(|i, o| b3(o, v3(i).cmple(v3(&i[3..])))),
        "v3_cmplt" => each!(|i, o| b3(o, v3(i).cmplt(v3(&i[3..])))),
        "v3_cmp_any_all" => each!(|i, o| {
            let k = v3(i).cmple(v3(&i[3..]));
            o[0] = k.any() as u32;
            o[1] = k.all() as u32;
        }),
        "v3_select" => each!(|i, o| s3(o, Vec3A::select(v3(i).cmplt(v3(&i[3..])), v3(&i[6..]), v3(&i[9..])))),
        "v3_is_finite_mask" => each!(|i, o| b3(o, v3(i).is_finite_mask())),
        "v3_as_ivec3" => each!(|i, o| {
            let v = v3(i).as_ivec3();
            o[..3].copy_from_slice(&[v.x as u32, v.y as u32, v.z as u32]);
        }),
        "v4_add" => each!(|i, o| s4(o, v4(i) + v4(&i[4..]))),
        "v4_sub" => each!(|i, o| s4(o, v4(i) - v4(&i[4..]))),
        "v4_mul" => each!(|i, o| s4(o, v4(i) * v4(&i[4..]))),
        "v4_div" => each!(|i, o| s4(o, v4(i) / v4(&i[4..]))),
        "v4_one_div" => each!(|i, o| s4(o, Vec4::ONE / v4(i))),
        "v4_neg_div" => each!(|i, o| s4(o, -v4(i) / v4(&i[4..]))),
        "v4_abs" => each!(|i, o| s4(o, v4(i).abs())),
        "v4_min" => each!(|i, o| s4(o, v4(i).min(v4(&i[4..])))),
        "v4_max" => each!(|i, o| s4(o, v4(i).max(v4(&i[4..])))),
        "v4_dot" => each!(|i, o| sf(o, v4(i).dot(v4(&i[4..])))),
        "v4_length" => each!(|i, o| sf(o, v4(i).length())),
        "v4_normalize" => each!(|i, o| s4(o, v4(i).normalize())),
        "v4_max_element" => each!(|i, o| sf(o, v4(i).max_element())),
        "v4_max_position" => each!(|i, o| o[0] = v4(i).max_position() as u32),
        "v4_cmpge" => each!(|i, o| b4(o, v4(i).cmpge(v4(&i[4..])))),
        "v4_cmpgt" => each!(|i, o| b4(o, v4(i).cmpgt(v4(&i[4..])))),
        "v4_cmple" => each!(|i, o| b4(o, v4(i).cmple(v4(&i[4..])))),
        "v4_cmplt" => each!(|i, o| b4(o, v4(i).cmplt(v4(&i[4..])))),
        "v4_mask_ops" => each!(|i, o| b4(o, (v4(i).cmpge(v4(&i[4..])) & v4(i).cmple(v4(&i[8..]))) | !v4(&i[4..]).cmpgt(v4(&i[8..])))),
        "v4_select" => each!(|i, o| s4(o, Vec4::select(v4(i).cmplt(v4(&i[4..])), v4(&i[8..]), v4(&i[12..])))),
        "v4_is_finite_mask" => each!(|i, o| b4(o, v4(i).is_finite_mask())),
        "m3_mul_v3" => each!(|i, o| s3(o, m3(i) * v3(&i[9..]))),
        "m3_mul_transpose_v3" => each!(|i, o| s3(o, m3(i).mul_transpose_vec3a(v3(&i[9..])))),
        "m3_mul_m3" => each!(|i, o| sm(o, m3(i) * m3(&i[9..]))),
        "m3_transpose" => each!(|i, o| sm(o, m3(i).transpose())),
        "m3_from_quat" => each!(|i, o| sm(o, Mat3A::from_quat(qu(i)))),
        "aff_mul" => each!(|i, o| sa(o, af(i) * af(&i[12..]))),
        "aff_transform_point3a" => each!(|i, o| s3(o, af(i).transform_point3a(v3(&i[12..])))),
        "aff_transform_vector3a" => each!(|i, o| s3(o, af(i).transform_vector3a(v3(&i[12..])))),
        "aff_transpose" => each!(|i, o| sa(o, aff_transpose(&af(i)))),
        "aff_inv_x_form" => each!(|i, o| s3(o, af(i).matrix3.mul_transpose_vec3a(v3(&i[12..]) - af(i).translation))),
        "q_mul" => each!(|i, o| sq(o, qu(i) * qu(&i[4..]))),
        "q_normalize" => each!(|i, o| sq(o, qu(i).normalize())),
        "q_mul_vec3a" => each!(|i, o| s3(o, qu(i).mul_vec3a(v3(&i[4..])))),
        "q_from_mat3a" => each!(|i, o| sq(o, Quat::from_mat3a(&m3(i)))),
        "q_integrate_tail" => each!(|i, o| {
            let q = (qu(i) * qu(&i[4..])).normalize();
            sq(o, q);
            sm(&mut o[4..], Mat3A::from_quat(q));
        }),
        other => panic!("no glam implementation for op {other}"),
    }
}

// ---- comparison ---------------------------------------------------------------------------------------------------

/// Whether word `k` of the op's output is a float (a NaN there is hardware-defined) or an integer/mask (compared strictly).
fn is_float_word(op: &Op, k: usize) -> bool {
    let n = op.name.as_str();
    if matches!(n, "f2i" | "f2u" | "f2ull" | "v3_as_ivec3" | "v3_min_position" | "v3_max_position" | "v4_max_position" | "v3_cmp_any_all")
        || n.contains("cmp")
        || n.contains("mask")
    {
        return false;
    }
    !(n == "v3_try_normalize" && k == 3)
}

#[derive(Default, Clone, Debug)]
struct Tally {
    cases: usize,
    exact: usize,
    nan_only: usize,
    first: Option<(Vec<f32>, Vec<u32>, Vec<u32>)>,
}

fn compare(op: &Op, input: &[f32], a: &[u32], b: &[u32], t: &mut Tally) {
    t.cases += a.len() / op.out_n;
    for (c, (x, y)) in a.chunks_exact(op.out_n).zip(b.chunks_exact(op.out_n)).enumerate() {
        if x == y {
            continue;
        }
        let strict = (0..op.out_n).any(|k| x[k] != y[k] && !(is_float_word(op, k) && f32::from_bits(x[k]).is_nan() && f32::from_bits(y[k]).is_nan()));
        if strict {
            t.exact += 1;
            if t.first.is_none() {
                t.first = Some((input[c * op.in_n..(c + 1) * op.in_n].to_vec(), x.to_vec(), y.to_vec()));
            }
        } else {
            t.nan_only += 1;
        }
    }
}

fn report(title: &str, rows: &[(Op, Tally)]) -> usize {
    println!("{title}: {} ops", rows.len());
    let (mut cases, mut exact, mut nan) = (0, 0, 0);
    for (op, t) in rows {
        println!("  {:<24} cases {:>9}  mismatches {:>6}  nan-payload-only {:>7}", op.name, t.cases, t.exact, t.nan_only);
        if let Some((i, x, y)) = &t.first {
            println!("      first: in {:x?}\n             ref {:x?}\n             got {:x?}", i.iter().map(|f| f.to_bits()).collect::<Vec<_>>(), x, y);
        }
        cases += t.cases;
        exact += t.exact;
        nan += t.nan_only;
    }
    println!("{title}: total cases {cases}, mismatches {exact}, nan-payload-only {nan}");
    exact
}

// ---- tests --------------------------------------------------------------------------------------------------------

#[test]
fn the_reference_has_no_fma() {
    assert!(!cfg!(target_feature = "fma"), "the glam reference must be built without the fma target feature");
    assert!(!shim_may_fuse(), "the host shim was compiled where the compiler may fuse multiply-add");
}

#[test]
fn host_shim_equals_glam_bit_for_bit() {
    assert!(!cfg!(target_feature = "fma") && !shim_may_fuse());
    let all = ops();
    let next = AtomicUsize::new(0);
    let rows = Mutex::new(Vec::new());
    let workers = std::thread::available_parallelism().map_or(2, |n| n.get().min(4));
    std::thread::scope(|s| {
        for _ in 0..workers {
            s.spawn(|| {
                loop {
                    let k = next.fetch_add(1, Ordering::Relaxed);
                    let Some(op) = all.get(k) else { break };
                    let mut t = Tally::default();
                    let (mut a, mut b) = (vec![0u32; CHUNK * op.out_n], vec![0u32; CHUNK * op.out_n]);
                    for chunk in 0..CASES / CHUNK {
                        let input = gen_chunk(op, chunk);
                        glam_run(op, &input, &mut a);
                        run_host(op, &input, &mut b);
                        compare(op, &input, &a, &b, &mut t);
                    }
                    rows.lock().unwrap().push((op.clone(), t));
                }
            });
        }
    });
    let mut rows = rows.into_inner().unwrap();
    rows.sort_by_key(|(op, _)| op.index);
    assert_eq!(rows.len(), all.len());
    let bad = report("host shim vs glam 0.34.1", &rows);
    assert_eq!(bad, 0, "the host build differs from glam");
}

#[test]
fn device_equals_host_bit_for_bit() -> Result<(), Box<dyn std::error::Error>> {
    use rocketsim_cuda::cuda::{LaunchConfig, PushKernelArg, context, load};
    let ctx = context(0)?;
    let stream = ctx.new_stream()?;
    let module = load(&ctx, SIMTEST_FATBIN)?;
    let kernel = module.load_function("simtest_run")?;
    let mut rows = Vec::new();
    for op in ops() {
        let mut t = Tally::default();
        let mut host = vec![0u32; CHUNK * op.out_n];
        for chunk in 0..CASES / CHUNK {
            let input = gen_chunk(&op, chunk);
            run_host(&op, &input, &mut host);
            let in_d = stream.clone_htod(&input)?;
            let mut out_d = stream.alloc_zeros::<u32>(CHUNK * op.out_n)?;
            let (opi, n, in_n, out_n) = (op.index as i32, CHUNK as i32, op.in_n as i32, op.out_n as i32);
            unsafe { stream.launch_builder(&kernel).arg(&opi).arg(&n).arg(&in_n).arg(&out_n).arg(&in_d).arg(&mut out_d).launch(LaunchConfig::for_num_elems(CHUNK as u32))? };
            let dev = stream.clone_dtoh(&out_d)?;
            compare(&op, &input, &host, &dev, &mut t);
        }
        rows.push((op, t));
    }
    let bad = report(&format!("device (sm_{}) vs host", rocketsim_cuda::cuda::ARCHS), &rows);
    assert_eq!(bad, 0, "the device differs from the host");
    Ok(())
}
