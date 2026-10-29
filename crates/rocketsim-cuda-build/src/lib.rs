//! Build-script helper: compiles each kernel group `cuda/<group>/<group>.cu` (one translation unit per group) into
//! `$OUT_DIR/<group>.fatbin` for the calling crate to embed. Architectures are SASS only, no PTX: a card the binary was
//! not built for fails to load instead of running a JIT-compiled kernel nobody tested.
//!
//! Environment:
//!   * `CUDA_HOME`: the toolkit root (the directory holding `bin/nvcc`); without it `nvcc` is looked up on the `PATH`.
//!     The toolkit must be CUDA 13.2 or newer.
//!   * `ROCKETSIM_CUDA_ARCHS`: the compute capabilities to build for, e.g. `86` or `86,120`. Without it the build targets
//!     the GPUs `nvidia-smi` reports on the build machine.

pub mod barrier;

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

/// The oldest CUDA toolkit a build accepts.
const MIN_TOOLKIT: (u32, u32) = (13, 2);

/// Compiles the groups, each flagged with whether it follows the physics float policy: no contraction into FMA, IEEE
/// division and square root, denormals kept. A group without it may contract. Also sets `ROCKETSIM_CUDA_ARCHS` for the
/// calling crate (e.g. `86,120`).
pub fn compile(groups: &[(&str, bool)]) {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let src = manifest.ancestors().map(|d| d.join("cuda")).find(|d| d.is_dir()).expect("no cuda/ above the crate");
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    println!("cargo:rerun-if-changed={}", src.display());
    for var in ["CUDA_HOME", "ROCKETSIM_CUDA_ARCHS"] {
        println!("cargo:rerun-if-env-changed={var}");
    }

    let archs = archs();
    for &(group, strict) in groups {
        fatbin(&src, group, &out.join(format!("{group}.fatbin")), &archs, strict);
    }
    println!("cargo:rustc-env=ROCKETSIM_CUDA_ARCHS={}", archs.iter().map(u32::to_string).collect::<Vec<_>>().join(","));
}

/// Compiles `<cuda>/<group>/<group>.cu` into the fat binary `output` for `archs`. A `strict` (physics float policy) group is
/// then held to the convergence-barrier gate (`barrier`): the build fails when any kernel of any architecture saves or
/// restores a convergence barrier through a general register.
pub fn fatbin(cuda: &Path, group: &str, output: &Path, archs: &[u32], strict: bool) {
    let nvcc = nvcc();
    check_toolkit(&nvcc);
    let input = cuda.join(group).join(format!("{group}.cu"));
    let mut cmd = Command::new(&nvcc);
    cmd.args(["-fatbin", "-O3", "-std=c++17", "-Xptxas=-warn-spills"]).arg("-I").arg(cuda);
    for a in archs {
        cmd.arg(format!("-gencode=arch=compute_{a},code=sm_{a}"));
    }
    // One thread per architecture: the archs compile side by side into a byte-identical fat binary.
    cmd.arg(format!("--threads={}", archs.len()));
    if strict {
        cmd.args(["--fmad=false", "-prec-div=true", "-prec-sqrt=true", "-ftz=false"]);
    }
    host_compiler(&mut cmd);
    cmd.arg("-o").arg(output).arg(&input);
    let status = cmd.status().unwrap_or_else(|e| panic!("cannot run {}: {e}", nvcc.display()));
    assert!(status.success(), "nvcc failed on {}", input.display());
    if strict {
        if let Err(report) = barrier::check(output, archs) {
            panic!("{report}");
        }
    }
}

/// `$CUDA_HOME/bin/nvcc`, else `nvcc` on the path.
fn nvcc() -> PathBuf {
    env::var_os("CUDA_HOME").map_or_else(|| PathBuf::from("nvcc"), |home| Path::new(&home).join("bin").join("nvcc"))
}

/// Parses `release 13.2,` out of `nvcc --version`.
fn toolkit_version(text: &str) -> Option<(u32, u32)> {
    let rest = text.split("release ").nth(1)?;
    let (major, minor) = rest.split(',').next()?.trim().split_once('.')?;
    Some((major.parse().ok()?, minor.parse().ok()?))
}

fn check_toolkit(nvcc: &Path) {
    let version = Command::new(nvcc)
        .arg("--version")
        .output()
        .unwrap_or_else(|e| panic!("no nvcc at {} ({e}); set CUDA_HOME to the CUDA toolkit root (.cargo/config.example.toml)", nvcc.display()));
    let text = String::from_utf8_lossy(&version.stdout);
    let found = toolkit_version(&text).unwrap_or_else(|| panic!("cannot read the CUDA version from `nvcc --version`:\n{text}"));
    assert!(found >= MIN_TOOLKIT, "CUDA {}.{} or newer is required, nvcc is {}.{}:\n{text}", MIN_TOOLKIT.0, MIN_TOOLKIT.1, found.0, found.1);
}

/// `ROCKETSIM_CUDA_ARCHS` (e.g. `86,120`), else the compute capability of every GPU `nvidia-smi` reports on this machine.
pub fn archs() -> Vec<u32> {
    if let Ok(list) = env::var("ROCKETSIM_CUDA_ARCHS") {
        return list.split(',').map(|a| a.trim().parse().expect("ROCKETSIM_CUDA_ARCHS: e.g. 86 or 86,120")).collect();
    }
    let mut archs = Vec::new();
    if let Ok(smi) = Command::new("nvidia-smi").args(["--query-gpu=compute_cap", "--format=csv,noheader"]).output() {
        for line in String::from_utf8_lossy(&smi.stdout).lines() {
            if let Some((major, minor)) = line.trim().split_once('.') {
                if let (Ok(major), Ok(minor)) = (major.parse::<u32>(), minor.parse::<u32>()) {
                    let cc = major * 10 + minor;
                    if !archs.contains(&cc) {
                        archs.push(cc);
                    }
                }
            }
        }
    }
    assert!(!archs.is_empty(), "no GPU found by nvidia-smi: set ROCKETSIM_CUDA_ARCHS to the compute capabilities to build for, e.g. 86 or 86,120");
    archs.sort_unstable();
    archs
}

/// nvcc needs MSVC's `cl.exe` and its environment on Windows; on Linux it finds the system compiler itself.
#[cfg(windows)]
fn host_compiler(cmd: &mut Command) {
    let cl = cc::windows_registry::find_tool("x86_64-pc-windows-msvc", "cl.exe").expect("MSVC cl.exe not found");
    cmd.envs(cl.get_envs());
    cmd.arg("-ccbin").arg(cl.path().parent().unwrap());
}

#[cfg(not(windows))]
fn host_compiler(_: &mut Command) {}
