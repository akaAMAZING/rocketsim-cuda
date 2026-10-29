//! The Rust mirrors of `cuda/sim/arena_state.cuh` against the C++ compiler's own layout: the size and alignment of every
//! struct and the offset and size of every field (the C++ host shim, csrc/layout.cpp, built from the same header the
//! device kernels include).

use rocketsim_cuda::{ffi, layout::*};

fn rust_structs() -> Vec<(&'static str, usize, &'static [(&'static str, usize, usize)])> {
    use core::mem::size_of;
    macro_rules! s {
        ($($t:ident),* $(,)?) => { vec![$((stringify!($t), size_of::<$t>(), $t::FIELDS)),*] };
    }
    s![
        Vec3, Quat, Mat3, Affine3, Aabb, U64, Edge, HandleEdges, Controls, HitInfo, RaycastInfo, Wheel, Car, PadTouching, BallDyn, Body,
        ManifoldPoint, Manifold, RecordPoint, ContactRecord, DynProxy, DynCell, Pair, ArenaState, MeshInfo, Tri, WideNode, Leaf, TriInfo,
        Plane, BodyConst, WheelConst, CarConst, PadConst, ProxyConst, CellRange, StaticData,
    ]
}

#[test]
fn every_field_offset_and_size_matches_the_cpp_layout() {
    let rows = ffi::layout_rows();
    let structs = ffi::layout_structs();
    let rust = rust_structs();
    let mut checked = 0;
    for (name, size, fields) in &rust {
        let (_, csize, calign) = structs.iter().find(|(n, _, _)| n == name).unwrap_or_else(|| panic!("{name}: no C++ struct"));
        assert_eq!(size, csize, "{name}: Rust size {size}, C++ size {csize}");
        assert_eq!(*csize % 4, 0, "{name}: size is not a whole number of words");
        assert!(*calign == 4 || *calign == 2, "{name}: alignment {calign}");
        let cpp: Vec<_> = rows.iter().filter(|r| r.0 == *name).collect();
        assert_eq!(cpp.len(), fields.len(), "{name}: {} C++ fields, {} Rust fields", cpp.len(), fields.len());
        let mut end = 0;
        for (&(f, off, sz), c) in fields.iter().zip(&cpp) {
            assert_eq!(f, c.1, "{name}: field order differs ({f} vs {})", c.1);
            assert_eq!((off, sz), (c.2, c.3), "{name}.{f}: Rust (offset {off}, size {sz}), C++ (offset {}, size {})", c.2, c.3);
            assert_eq!(off, end, "{name}.{f}: padding before the field");
            end = off + sz;
            checked += 1;
        }
        assert_eq!(end, *size, "{name}: the fields do not fill the struct (padding or a missing field)");
    }
    // every C++ struct of the header has a mirror
    for (n, _, _) in &structs {
        if !n.starts_with("Opt<") {
            assert!(rust.iter().any(|r| r.0 == n), "C++ struct {n} has no Rust mirror");
        }
    }
    // Opt<T> = {u32 some; T v}: size 4 + size_of T, no padding
    use core::mem::size_of;
    for (n, rs) in [
        ("Opt<u32>", size_of::<Opt<u32>>()),
        ("Opt<Vec3>", size_of::<Opt<Vec3>>()),
        ("Opt<U64>", size_of::<Opt<U64>>()),
        ("Opt<HitInfo>", size_of::<Opt<HitInfo>>()),
        ("Opt<RaycastInfo>", size_of::<Opt<RaycastInfo>>()),
    ] {
        let c = structs.iter().find(|s| s.0 == n).unwrap();
        assert_eq!(rs, c.1, "{n}");
    }
    assert_eq!(ffi::cpp_arena_words(), ARENA_WORDS);
    assert_eq!(ffi::cpp_static_words(), STATIC_WORDS);
    println!("{} fields of {} structs match; ArenaState {} bytes ({} words), StaticData {} bytes", checked, rust.len(), ARENA_WORDS * 4, ARENA_WORDS, STATIC_WORDS * 4);
}

/// Every word of ArenaState is described (the comparator relies on it), and the word kinds are what the header says.
#[test]
fn every_word_is_described() {
    let leaves = arena_leaves();
    assert_eq!(leaves.len(), ARENA_WORDS);
    let floats = leaves.iter().filter(|l| l.1 == Kind::Float).count();
    println!("ArenaState: {} words, {} float, {} discrete", ARENA_WORDS, floats, ARENA_WORDS - floats);
    let mut s = ArenaState::zeroed();
    // a word of a float field changes kind-wise as documented: spot checks
    let names: std::collections::BTreeSet<_> = leaves.iter().map(|l| l.0.as_str()).collect();
    for n in ["cars.vel", "cars.wheels.lat_friction", "bodies.world_trans", "manifolds.points.pos_world_on_a", "edges", "overflow"] {
        assert!(names.contains(n), "no group {n}");
    }
    s.n_pairs = 1;
    assert_eq!(s.words().iter().filter(|&&w| w != 0).count(), 1);
}
