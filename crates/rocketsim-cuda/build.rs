use std::path::Path;

fn main() {
    // Device side of the glam parity test and the simulator kernels (strict float policy), on its own thread: the host shim
    // below compiles meanwhile and its few seconds hide behind nvcc.
    let device = std::thread::spawn(|| rocketsim_cuda_build::compile(&[("simtest", true), ("sim", true)]));

    // Host side: the same headers behind a C ABI, with contraction off.
    let cuda = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../cuda");
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .file("csrc/shim.cpp")
        .file("csrc/layout.cpp")
        .file("csrc/phase.cpp")
        .file("csrc/phase_platform.cpp")
        .file("csrc/phase_mixed.cpp")
        .file("csrc/libm.cpp")
        .include(&cuda)
        .opt_level(2)
        .warnings(true);
    if build.get_compiler().is_like_msvc() {
        build.flag("/std:c++17").flag("/fp:precise");
    } else {
        build.flag("-std=c++17").flag("-ffp-contract=off");
    }
    build.compile("rscuda_shim");
    println!("cargo:rerun-if-changed=csrc");
    println!("cargo:rerun-if-changed={}", cuda.display());
    device.join().expect("nvcc failed (its message is above)");
}
