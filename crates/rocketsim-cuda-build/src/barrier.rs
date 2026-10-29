//! The convergence-barrier gate: fails the build of a physics-policy kernel
//! group when any kernel's SASS saves or restores a convergence barrier through a general register.
//!
//! The fault. A warp reconverges at a barrier register (`BSSY Bk` opens a region, `BSYNC Bk` waits for it). ptxas has B0..B11 for a
//! whole call chain: a region that is open across a call, and the callee's own regions, must all fit. When they do not, ptxas parks a
//! barrier in a GPR (`BMOV.32.CLEAR Rx, Bk` after the `BSSY`, a *save*) and puts it back in a barrier register before the wait
//! (`BMOV.32 Bj, Rx`, a *restore*). On the RTX 3080 (sm_86) a wait on a restored barrier did not hold the warp: lanes ran on apart,
//! into code that ptxas had compiled as converged, and the arenas of their warp-mates were corrupted (a minimal reproduction in plain
//! integer code fails the same way at nesting depth 13 and deeper; sm_120 has the same spill code). The
//! hardware rule is not established, so there is no per-site judgement here: a save or a restore anywhere in the code of a kernel
//! (its own and every function it calls: cuobjdump lists a kernel with its callees) fails the build, with no allowlist. The cure
//! is in the source: keep the divergent nesting along every call chain within B0..B11, e.g. by calling deep code from a flat loop.
//!
//! `BMOV.32.CLEAR RZ, Bk` (a barrier cleared and discarded) is neither a save nor a restore and is only counted.
//!
//! The scan reads `cuobjdump -sass` (NVIDIA's own disassembler, in the toolkit's `cuda_cuobjdump` component) once per architecture,
//! a few seconds beside the compile. It is not a Rust decoder of the cubin on purpose: the fat binary's per-architecture images
//! are compressed and the instruction encodings differ between architectures, so a decoder would be a second, unverified
//! disassembler under the gate.

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// What one BMOV does to a barrier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bmov {
    /// `BMOV.32.CLEAR Rx, Bk`: the barrier parked in a GPR.
    Save,
    /// `BMOV.32 Bk, Rx`: a parked barrier put back.
    Restore,
    /// `BMOV.32.CLEAR RZ, Bk`: cleared and discarded.
    Clear,
}

/// The BMOV class of one `cuobjdump -sass` line, if it is a BMOV (a predicate prefix and the trailing encoding comment are ignored).
pub fn classify(line: &str) -> Option<Bmov> {
    let l = line.trim_start();
    let l = l.strip_prefix("/*")?;
    let (addr, rest) = l.split_once("*/")?;
    if addr.is_empty() || !addr.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    let mut rest = rest.trim_start();
    if rest.starts_with('@') {
        rest = rest.split_once(char::is_whitespace)?.1.trim_start();
    }
    let insn = rest.split(';').next()?.trim();
    let (op, operands) = insn.split_once(char::is_whitespace)?;
    if op != "BMOV" && !op.starts_with("BMOV.") {
        return None;
    }
    let dst = operands.split(',').next()?.trim();
    Some(if dst == "RZ" {
        Bmov::Clear
    } else if dst.starts_with('B') {
        Bmov::Restore
    } else {
        Bmov::Save
    })
}

/// The address of an instruction line (the hex between the first `/*` and `*/`), for the report.
fn address(line: &str) -> &str {
    line.trim_start().strip_prefix("/*").and_then(|l| l.split_once("*/")).map_or("?", |(a, _)| a)
}

/// One kernel's findings.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Kernel {
    pub name: String,
    pub saves: u32,
    pub restores: u32,
    pub clears: u32,
    /// The first few addresses of a save or restore.
    pub at: Vec<String>,
}

impl Kernel {
    pub fn bad(&self) -> bool {
        self.saves + self.restores > 0
    }
}

/// Scans a `cuobjdump -sass` listing: per kernel (`Function : name`) the BMOVs it holds.
pub fn scan_listing(listing: impl BufRead) -> std::io::Result<Vec<Kernel>> {
    let mut kernels: Vec<Kernel> = Vec::new();
    let mut listing = listing;
    let mut line = String::new();
    // A listing is 100 MB for the simulator: one buffer, and a substring test before any parsing.
    loop {
        line.clear();
        if listing.read_line(&mut line)? == 0 {
            break;
        }
        if !line.contains("BMOV") && !line.contains("Function :") {
            continue;
        }
        if let Some(name) = line.split_once("Function :").map(|(_, n)| n.trim()) {
            kernels.push(Kernel { name: name.to_string(), ..Kernel::default() });
            continue;
        }
        let Some(kind) = classify(&line) else { continue };
        let Some(k) = kernels.last_mut() else { continue };
        match kind {
            Bmov::Clear => k.clears += 1,
            Bmov::Save | Bmov::Restore => {
                if kind == Bmov::Save {
                    k.saves += 1;
                } else {
                    k.restores += 1;
                }
                if k.at.len() < 4 {
                    k.at.push(address(&line).to_string());
                }
            }
        }
    }
    Ok(kernels)
}

/// `$CUDA_HOME/bin/cuobjdump`, else `cuobjdump` on the path (the same rule as nvcc's, lib.rs).
fn cuobjdump() -> PathBuf {
    std::env::var_os("CUDA_HOME").map_or_else(|| PathBuf::from("cuobjdump"), |home| Path::new(&home).join("bin").join("cuobjdump"))
}

/// Scans the SASS of architecture `arch` (e.g. 86) in `file` (a fat binary, a cubin or an executable).
pub fn scan(file: &Path, arch: u32) -> Result<Vec<Kernel>, String> {
    let tool = cuobjdump();
    let mut child = Command::new(&tool)
        .args(["-sass", "-arch", &format!("sm_{arch}")])
        .arg(file)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| format!("cannot run {} ({e}); the toolkit needs its cuda_cuobjdump component, CUDA_HOME is the root", tool.display()))?;
    let kernels = scan_listing(BufReader::with_capacity(1 << 20, child.stdout.take().unwrap())).map_err(|e| e.to_string())?;
    let status = child.wait().map_err(|e| e.to_string())?;
    if !status.success() || kernels.is_empty() {
        return Err(format!("{}: cuobjdump found no sm_{arch} kernel (status {status}); a gate that reads nothing passes nothing", file.display()));
    }
    Ok(kernels)
}

/// The gate: every architecture in `archs`, every kernel of `file`. Err holds the report of what failed.
pub fn check(file: &Path, archs: &[u32]) -> Result<(), String> {
    let mut report = String::new();
    for &arch in archs {
        let kernels = scan(file, arch)?;
        // clears alone pass, but they came before every save seen so far: name the site so it gets read
        for k in kernels.iter().filter(|k| !k.bad() && k.clears > 0) {
            println!("cargo:warning=sm_{arch} {}: {} BMOV.32.CLEAR RZ (barrier clears, no save or restore); read the site", k.name, k.clears);
        }
        for k in kernels.iter().filter(|k| k.bad()) {
            report.push_str(&format!(
                "  sm_{arch} {}: {} barrier saves, {} restores (first at {})\n",
                k.name,
                k.saves,
                k.restores,
                k.at.join(", ")
            ));
        }
    }
    if report.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "{}: a convergence barrier is saved or restored through a GPR (BMOV) in the call tree of a kernel; the divergent nesting along a call chain does not fit B0..B11 (no allowlist)\n{report}",
            file.display()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CLEAN: &str = "\
\t\tFunction : sim8
\t.headerflags\t@\"EF_CUDA_SM86\"
        /*0000*/                   MOV R1, c[0x0][0x28] ;                    /* 0x00000a0000017a02 */
        /*0b40*/                   BSSY B0, 0x3500 ;                         /* 0x0000003000007945 */
        /*0b50*/                   BMOV.32.CLEAR RZ, B11 ;                   /* 0x00000000ff007355 */
        /*0b60*/                   BSYNC B0 ;                                /* 0x0000000000007941 */
";
    const SAVE: &str = "\
\t\tFunction : k8
        /*9c80*/                   BMOV.32.CLEAR R6, B0 ;                    /* 0x0000000000067355 */
";
    const RESTORE: &str = "\
\t\tFunction : k8
        /*9cc0*/                   BMOV.32 B0, R6 ;                          /* 0x0000000600007356 */
";

    fn scan_str(s: &str) -> Vec<Kernel> {
        scan_listing(s.as_bytes()).unwrap()
    }

    #[test]
    fn clean_listing_passes_and_counts_the_clear() {
        let k = scan_str(CLEAN);
        assert_eq!(k.len(), 1);
        assert!(!k[0].bad());
        assert_eq!(k[0].clears, 1);
    }

    #[test]
    fn a_save_fails() {
        let k = scan_str(SAVE);
        assert!(k[0].bad());
        assert_eq!((k[0].saves, k[0].restores, k[0].at.clone()), (1, 0, vec!["9c80".to_string()]));
    }

    #[test]
    fn a_restore_fails() {
        let k = scan_str(RESTORE);
        assert_eq!((k[0].saves, k[0].restores), (0, 1));
        assert!(k[0].bad());
    }

    #[test]
    fn a_predicated_bmov_and_sm120_spelling_count() {
        let t = "\t\tFunction : k\n        /*1000*/              @P0 BMOV.32.CLEAR R9, B3 ;                 /* 0x1 */\n        /*1010*/                   BMOV.32 B3, R9 ;   /* 0x2 */\n";
        let k = scan_str(t);
        assert_eq!((k[0].saves, k[0].restores), (1, 1));
    }

    #[test]
    fn other_lines_are_not_bmov() {
        assert_eq!(classify("        /*0000*/                   MOV R1, c[0x0][0x28] ;   /* 0x1 */"), None);
        assert_eq!(classify("\t.section\t.text.sim8,\"ax\",@progbits"), None);
        assert_eq!(classify("        /*0040*/                   BSSY B0, 0x3500 ;   /* 0x1 */"), None);
    }

    #[test]
    fn instructions_before_any_function_are_ignored() {
        assert!(scan_str("        /*9c80*/  BMOV.32.CLEAR R6, B0 ;  /* 0x1 */\n").is_empty());
    }
}
