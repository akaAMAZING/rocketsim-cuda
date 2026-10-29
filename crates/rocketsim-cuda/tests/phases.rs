//! The phase test driver. For every start state (played by V3 itself, `v3dump` scenarios) and every phase k: import V3's state
//! at boundary k, run phase k, compare with V3's state at boundary k + 1, field by field, and print the per-class table. The
//! runners:
//!   * the oracle (V3 itself): zero differences prove the harness; it is the standard a ported phase is held to;
//!   * `rscuda_phase_platform`, the host C ABI built with the platform C library behind `lf::sin` ... (LF_LIBM_PLATFORM): V3
//!     calls the platform library, so this build is what a ported phase is held to bit for bit (the porting verification);
//!   * `rscuda_phase`, the host C ABI built with the correctly rounded libm (the default, cuda/common/libm_cr.cuh), and
//!     `phase_k`, the device kernel (the same libm): the two must equal each other bit for bit in every phase and every state,
//!     whatever V3 says. Their differences from V3 are reported, and every differing word is attributed to the libm call that
//!     causes it (a third host build switches one function at a time to the platform library); a difference that is not the
//!     libm's fails the run for the required phases.
//!   ROCKETSIM_MESHES=<dir> cargo test --release -p rocketsim-cuda --test phases -- --nocapture
//! `ROCKETSIM_CUDA_SAMPLES` states per class (default 100). `ROCKETSIM_PHASE_REQUIRE=3,4` (phase indices: 0 P0, 1 P1, 2 P2, 3 P3a,
//! 4 P3b, 5 P4, 6 P5, 7 P6, 8 P7, 9 P8) turns those phases from reported into asserted: on the platform host every class
//! discrete-identical and `ROCKETSIM_PHASE_MAX_ULP` (default 4) at most on every float word of the states that ran; on the device
//! every differing word attributed to a libm call. A phase that is not implemented (status -1) fills the `failed` column; the
//! `ran` column counts the states each table's verdict is taken over. By default every phase is required.

mod common;

use common::{DevicePhase, Oracle, samples_per_class, all_configs, class_samples, run_phase_suite, run_phase_tests};
use rocketsim_cuda::compare::Stats;
use rocketsim::PHASE_NAMES;

fn required() -> Vec<usize> {
    std::env::var("ROCKETSIM_PHASE_REQUIRE")
        .map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|_| (0..rocketsim::PHASES).collect())
}

#[test]
fn phase_driver_oracle_is_clean_and_the_builds_agree() {
    let per_class: usize = samples_per_class(100);
    let mut configs = all_configs();
    let samples = class_samples(per_class, &mut configs);
    println!("{} start states with V3's state at all {} boundaries", samples.len(), rocketsim::PHASES + 1);

    // the oracle: V3 run from the rebuilt dump must reproduce V3's next boundary exactly
    let oracle = run_phase_tests(&samples, &mut configs, &mut Oracle::new(), &[]);
    println!("{}", oracle.render());
    let t = oracle.total();
    assert_eq!(t.failed, 0);
    assert_eq!(t.stats.diff_words(), (0, 0), "the oracle differs from V3: the harness is wrong\n{}", t.stats.field_table(20));
    assert_eq!(t.stats.discrete_equal, t.stats.states);
    assert_eq!(t.stats.bit_equal, t.stats.states);

    let suite = run_phase_suite(&samples, &mut configs, &mut DevicePhase::new(), &[]);
    println!("{}", suite.platform.render());
    println!("{}", suite.cr_vs_device.render());
    println!("{}", suite.device.render());
    println!("{}", suite.render_attribution());
    for t in [&suite.platform, &suite.cr_vs_device, &suite.device] {
        println!("{}", t.render_by_phase(12));
    }

    // the single-source check: the correctly rounded host and the device do the same, state for state, in every phase
    for (key, c) in &suite.cr_vs_device.cells {
        assert_eq!(c.stats.diff_words(), (0, 0), "{key:?}: host (correctly rounded libm) and device differ\n{}", c.stats.field_table(20));
        assert_eq!(c.stats.bit_equal, c.stats.states, "{key:?}: host (correctly rounded libm) and device differ");
    }
    // all three report the same statuses (also asserted state by state inside the suite)
    for (key, p) in &suite.platform.cells {
        let (c, d) = (&suite.cr.cells[key], &suite.device.cells[key]);
        assert_eq!((p.ran, p.failed), (c.ran, c.failed), "{key:?}: the host builds report different statuses");
        assert_eq!((c.ran, c.failed), (d.ran, d.failed), "{key:?}: host and device report different statuses");
    }

    // the required phases: the platform host holds to V3, and the device's differences are the libm's
    let req = required();
    let max_ulp: u64 = std::env::var("ROCKETSIM_PHASE_MAX_ULP").ok().and_then(|v| v.parse().ok()).unwrap_or(4);
    for &k in &req {
        // asserted on the states the phase completed; states that reached an unported shape pair (status -1) are in the
        // table's `failed` column, and at least one state must have run
        let mut st = Stats::default();
        let mut ran = 0;
        for ((_, kk), c) in &suite.platform.cells {
            if *kk == k {
                st.merge(&c.ok);
                ran += c.ran;
            }
        }
        assert!(ran > 0, "{}: phase not implemented", PHASE_NAMES[k]);
        let (d, _) = st.diff_words();
        let worst = st.worst().map_or(0, |w| w.1);
        assert!(d == 0 && worst <= max_ulp, "host (platform libm) {} is not within {max_ulp} ulp of V3:\n{}", PHASE_NAMES[k], st.field_table(20));
        let a = &suite.attribution[&k];
        assert_eq!(a.other, 0, "{}: the device differs from V3 in words the libm does not explain:\n{}", PHASE_NAMES[k], a.examples.join("\n"));
    }
}
