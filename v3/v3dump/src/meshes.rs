//! Loads the user's collision meshes into V3 once per process. The meshes are the user's files: they are not part of this
//! repository. Folder: `$ROCKETSIM_MESHES`, the directory that holds `soccar/*.cmf`.

use std::{
    fs,
    path::{Path, PathBuf},
    sync::OnceLock,
};

use rocketsim::GameMode;

pub fn meshes_dir() -> PathBuf {
    std::env::var_os("ROCKETSIM_MESHES").map_or_else(|| Path::new("collision_meshes").to_path_buf(), PathBuf::from)
}

static INIT: OnceLock<Result<(), String>> = OnceLock::new();

/// Loads the 16 Soccar meshes, sorted by file name (`mesh_0`, `mesh_1`, `mesh_10` ... `mesh_9`); V3 checks each file's hash itself. Safe to call from every test and thread.
pub fn init_meshes() {
    INIT.get_or_init(|| {
        let dir = meshes_dir().join("soccar");
        let mut files: Vec<PathBuf> = fs::read_dir(&dir)
            .map_err(|e| format!("{}: {e} (set ROCKETSIM_MESHES)", dir.display()))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.is_file() && p.extension().is_some_and(|x| x == "cmf"))
            .collect();
        files.sort();
        let bytes: Vec<Vec<u8>> = files
            .iter()
            .map(|p| fs::read(p).map_err(|e| format!("{}: {e}", p.display())))
            .collect::<Result<_, _>>()?;
        rocketsim::init_from_mem([(GameMode::Soccar, bytes)], true).map_err(|e| e.to_string())
    })
    .as_ref()
    .unwrap_or_else(|e| panic!("mesh init failed: {e}"));
}
