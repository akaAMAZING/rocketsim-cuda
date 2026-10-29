//! Start states (played by V3 itself) through the whole plumbing: V3 dump -> `ArenaState` (every class) -> AoSoA-32 buffer ->
//! copy-in and copy-out through a thread-local `ArenaState` on the host (C++ shim) and on the device (kernel `sim_identity`) ->
//! back. Everything must be byte-identical, padding lanes included.
//!   ROCKETSIM_MESHES=<dir> cargo test --release -p rocketsim-cuda --test aosoa -- --nocapture
//! `ROCKETSIM_CUDA_SAMPLES` states per class (default 400).

mod common;

use common::{CLASSES, all_configs, read_rows, samples_per_class};
use rocketsim_cuda::{aosoa, device::SimDevice, dump::parse_arena, ffi, layout::{ARENA_WORDS, ArenaState}};
use rocketsim::dump::ArenaDump;

fn start_states(per_class: usize) -> (Vec<Box<ArenaState>>, Vec<String>) {
    let configs = all_configs();
    let mut states = Vec::new();
    let mut classes = Vec::new();
    let mut failures = Vec::new();
    for class in CLASSES {
        let mut ok = 0;
        for row in read_rows(class, per_class) {
            let hash = ArenaDump::from_bytes(row.dump.clone()).unwrap().config_hash();
            let Some(cfg) = configs.get(&hash) else {
                failures.push(format!("{class} row {}: config hash {hash:016x} is none of the 98 car configurations", row.index));
                continue;
            };
            match parse_arena(&row.dump, &cfg.sd) {
                Ok((s, _)) => {
                    states.push(s);
                    classes.push(class.to_string());
                    ok += 1;
                }
                Err(e) => failures.push(format!("{class} row {}: {e}", row.index)),
            }
        }
        println!("{class}: {ok} states imported");
    }
    assert!(failures.is_empty(), "{} start states do not import:\n{}", failures.len(), failures[..failures.len().min(10)].join("\n"));
    (states, classes)
}

#[test]
fn start_states_round_trip_through_aosoa_on_host_and_device() {
    let per_class: usize = samples_per_class(400);
    let (mut states, _) = start_states(per_class);
    // a count that is not a multiple of 32 exercises the padded last block
    states.truncate(states.len() - states.len() % 32 + 7);
    let n = states.len();
    println!("{n} start states, ArenaState {} bytes, {} AoSoA blocks", ARENA_WORDS * 4, n.div_ceil(32));

    // layout: Rust's index function against C++'s, and pack / unpack
    for a in [0usize, 1, 31, 32, 33, n - 1] {
        for w in [0usize, 1, 31, 32, ARENA_WORDS - 1] {
            assert_eq!(aosoa::index(a, w), ffi::cpp_aosoa_index(a as u32, w), "index({a}, {w})");
        }
    }
    assert_eq!(aosoa::words(n), ffi::cpp_aosoa_words(n as u32));
    let packed = aosoa::pack(&states);
    let back = aosoa::unpack(&packed, n);
    assert!(states.iter().zip(&back).all(|(a, b)| a.as_bytes() == b.as_bytes()), "pack then unpack changed a state");

    // host: copy-in and copy-out through a local ArenaState (the C++ single-source functions)
    let host = ffi::cpp_aosoa_identity(&packed, n as u32);
    assert!(host == packed, "host identity round trip differs");

    // device: the same through the sim_identity kernel
    let dev = SimDevice::new(0).expect("a CUDA device");
    let input = dev.upload_states(&states).unwrap();
    let mut out = dev.alloc_states(n).unwrap();
    dev.identity(&input, &mut out).unwrap();
    let words = dev.download_words(&out).unwrap();
    let diff = words.iter().zip(&packed).filter(|(a, b)| a != b).count();
    assert_eq!(diff, 0, "device identity round trip differs in {diff} of {} words", packed.len());
    let down = dev.download_states(&out).unwrap();
    assert!(states.iter().zip(&down).all(|(a, b)| a.as_bytes() == b.as_bytes()));
    // the tick-boundary copy moves TICK_WORDS only and the start states (tick boundaries) survive it
    assert_eq!(ffi::cpp_tick_words(), rocketsim_cuda::layout::TICK_WORDS);
    let host_tick = ffi::cpp_aosoa_identity_tick(&packed, n as u32);
    assert!(host_tick == packed, "host tick-boundary round trip differs (the mid-tick part of a tick-boundary state must be zero)");
    let mut out_tick = dev.alloc_states(n).unwrap();
    dev.identity_tick(&input, &mut out_tick).unwrap();
    assert!(dev.download_words(&out_tick).unwrap() == packed, "device tick-boundary round trip differs");
    println!("tick-boundary copies ({} of {} words) byte-identical too", rocketsim_cuda::layout::TICK_WORDS, ARENA_WORDS);
    // StaticData: uploaded once, comes back identical
    let cfg = all_configs().into_values().next().unwrap();
    let ds = dev.upload_static(&cfg.sd).unwrap();
    let back = dev.stream.clone_dtoh(&ds.buf).unwrap();
    assert!(back == cfg.sd.as_bytes(), "StaticData upload differs");
    println!("StaticData uploaded ({} bytes) and read back identical", back.len());
    println!("AoSoA identity round trip byte-identical for {n} states ({} words): host and device", packed.len());
}
