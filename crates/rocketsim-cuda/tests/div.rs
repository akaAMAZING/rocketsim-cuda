//! The device division `lf_div` (`cuda/common/hd.cuh`): IEEE correctly rounded, bit-identical to CUDA's `__fdiv_rn`, without that
//! library routine's out-of-line slow path (which saves and restores convergence barriers through a GPR, the code pattern the build
//! gate bans from every kernel that runs the phases).
//!   * `div_stream`: `ROCKETSIM_DIV_N` (default 2^28) pairs per mode on the device, `lf_div` against `__fdiv_rn`, bit for bit (a NaN
//!     with another payload counts): mode 0 all bits random, 1 exponents at the fast path's window edges and edge mantissas,
//!     2 inside the window, 3 zero / subnormal / normal / infinity / NaN operands, 4 the all-ones divisor mantissa at every exponent.
//!   * explicit operands: the special-value grid and a random sample, `lf_div` against the host's IEEE division (two NaNs agree).
//! The instruction sequence of the fast path is the library's, and its exactness there is this test's to show.
//!   cargo test --release -p rocketsim-cuda --test div -- --nocapture

use rocketsim_cuda::cuda::{LaunchConfig, PushKernelArg, context, load};
use rocketsim_cuda::SIMTEST_FATBIN;

fn mix(z: u64) -> u64 {
    let z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    let z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

#[test]
fn lf_div_equals_fdiv_rn_and_ieee_division() {
    let ctx = context(0).expect("CUDA device");
    let stream = ctx.new_stream().unwrap();
    let module = load(&ctx, SIMTEST_FATBIN).unwrap();
    let n: u64 = std::env::var("ROCKETSIM_DIV_N").ok().and_then(|v| v.parse().ok()).unwrap_or(1 << 28);

    let stream_k = module.load_function("div_stream").unwrap();
    let mut failed = false;
    for mode in 0..5i32 {
        let mut bad = stream.alloc_zeros::<u64>(1).unwrap();
        let mut first = stream.alloc_zeros::<u32>(2).unwrap();
        let seed = 0x1234_5678u64 + 7919 * mode as u64;
        let cfg = LaunchConfig { grid_dim: (1024, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        // SAFETY: `bad` and `first` hold 1 and 2 words.
        unsafe { stream.launch_builder(&stream_k).arg(&mode).arg(&seed).arg(&n).arg(&mut bad).arg(&mut first).launch(cfg).unwrap() };
        let (b, f) = (stream.clone_dtoh(&bad).unwrap()[0], stream.clone_dtoh(&first).unwrap());
        println!("mode {mode}: {n} pairs, {b} differ from __fdiv_rn{}", if b > 0 { format!(" (first {:08x} / {:08x})", f[0], f[1]) } else { String::new() });
        failed |= b > 0;
    }
    assert!(!failed, "lf_div differs from __fdiv_rn");

    // explicit operands against the host's IEEE division: the special grid, then a random sample
    let specials = [0u32, 1, 0x007f_ffff, 0x0080_0000, 0x3f80_0000, 0x3fff_ffff, 0x4000_0000, 0x7f7f_ffff, 0x7f80_0000, 0x7fc0_0000, 0x7f80_0001, 0x0100_0000, 0x7e80_0000, 0x0000_0123, 0x3380_0000];
    let (mut a, mut b) = (Vec::new(), Vec::new());
    for &x in &specials {
        for &y in &specials {
            for sign in 0..4u32 {
                a.push(x | (sign & 1) << 31);
                b.push(y | (sign >> 1) << 31);
            }
        }
    }
    for i in 0..(1u64 << 20) {
        let r = mix(i);
        a.push(r as u32);
        b.push(mix(r) as u32);
    }
    // the all-ones divisor mantissa at every exponent, both signs, against the host's IEEE division (independent of the card's
    // reciprocal seed, which __fdiv_rn shares)
    for eb in 1..=254u32 {
        for j in 0..64u64 {
            let r = mix(0xd1b5_4a32 ^ (u64::from(eb) << 8) ^ j);
            a.push(r as u32 & 0x7fff_ffff | (j as u32 & 1) << 31);
            b.push(eb << 23 | 0x7f_ffff | (j as u32 >> 1 & 1) << 31);
        }
    }
    let count = a.len() as u32;
    let (da, db) = (stream.clone_htod(&a).unwrap(), stream.clone_htod(&b).unwrap());
    let mut out = stream.alloc_zeros::<u32>(a.len()).unwrap();
    let pairs = module.load_function("div_pairs").unwrap();
    // SAFETY: the buffers hold `count` words.
    unsafe { stream.launch_builder(&pairs).arg(&da).arg(&db).arg(&count).arg(&mut out).launch(LaunchConfig::for_num_elems(count)).unwrap() };
    let got = stream.clone_dtoh(&out).unwrap();
    let mut wrong = 0;
    for i in 0..a.len() {
        let want = f32::from_bits(a[i]) / f32::from_bits(b[i]);
        let have = f32::from_bits(got[i]);
        if !(want.to_bits() == have.to_bits() || (want.is_nan() && have.is_nan())) {
            if wrong < 5 {
                println!("{:08x} / {:08x}: device {:08x}, host {:08x}", a[i], b[i], got[i], want.to_bits());
            }
            wrong += 1;
        }
    }
    println!("explicit operands: {} pairs, {wrong} differ from the host's division", a.len());
    assert_eq!(wrong, 0, "lf_div differs from the host's IEEE division");
}
