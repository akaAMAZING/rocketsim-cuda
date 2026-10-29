# rocketsim-cuda

RocketSim on the GPU. A bit-exact CUDA C++ port of [RocketSim](https://github.com/ZealanL/RocketSim) V3 that steps tens of
thousands of Rocket League arenas in parallel on one NVIDIA card, with a small Rust host crate to load arenas, step them and
read them back.

It is built for reinforcement learning: the arenas live in GPU memory next to your networks, so a training loop never
copies game state through the CPU, and every arena steps exactly as RocketSim would have stepped it.

## Where it comes from

- **RocketSim** by ZealanL is the standard Rocket League physics simulator for bot training: Bullet Physics plus the
  game's own car, ball and arena rules.
- **RocketSim V3** (branch `v3-rust`, by ZealanL and VirxEC) is its Rust rewrite.
- **rocketsim-cuda** ports V3 at commit `4572287` (v0.3.3, 26 October 2026) to CUDA, function by function. Every ported
  function cites the V3 source lines it mirrors (`rigid_body.rs:432-456` and so on), keeps V3's operation order, branch
  order and container semantics, and does all float math through one layer that reproduces glam's SSE2 order with no FMA
  contraction.

## What it adds over CPU RocketSim

- **Massive parallelism.** One GPU thread per arena; 4,096 to 65,536 arenas per launch on one card; eight ticks per kernel
  launch (one 15 Hz decision at 120 Hz physics).
- **Exactness you can check.** The same C++ source compiles for the GPU and for the CPU:
  - the CPU build with the platform's own math library equals V3 **bit for bit**;
  - the GPU build equals the CPU build with correctly rounded math **bit for bit**;
  - so the GPU's only difference from V3 is the last-bit rounding of `sin` and `cos`, which the tests attribute word by
    word.
- **Determinism.** A result does not depend on the block size, the order of the arenas, or the launch: the same start state
  and controls give the same words on every run and every card.
- **Coalesced state.** Arenas are stored AoSoA-32 (one warp reads 32 arenas' copy of a word in one transaction); a tick
  boundary moves only the persistent part of the state.
- **Exact import and export.** A patch to V3 (`v3/rocketsim-dump.patch`) exports V3's complete arena state, including the
  hidden history V3 keeps between ticks (contact manifolds, broadphase pair order, wheel caches, pad membership, the
  ball's sleep timer). Any V3 arena can be moved to the GPU, stepped, and moved back.

## Scope

- Soccar, 1v1 (two cars per arena). All seven car hitbox presets, in any pairing.
- Mutators: the values the arena is built with (set in V3 before the export). Changing mutators at run time is not
  supported.
- Not supported: 2v2 and 3v3, Hoops, Dropshot, Heatseeker, Snowday, ball-only arenas, `remove_car`. A tick that would
  need a collision pair the port does not implement is flagged in the step result, never silently wrong.
- About 11.1 KB of state per arena (730 MB for 65,536 arenas).

## Speed

Arena-ticks per second (one arena advanced one tick at 120 Hz), RTX 3080 against a Ryzen 9 5900X, both on the same idle
machine. GPU: the 8-tick decision kernel, median of 20 launches, on a mixed set of states sampled from play (kickoffs,
pinches, goal-mouth scrambles, demos, car-ball contact, wall and ceiling play, free flight, ground driving). CPU: stock
RocketSim V3 `4572287`, release build, on its own scenario set.

| | arena-ticks/s |
|---|---|
| GPU, 65,536 arenas | **4.89 M** (best launch 5.0 M) |
| GPU, 16,384 arenas | 4.38 M |
| GPU, 4,096 arenas | 2.26 M |
| CPU, 1 thread | 0.56 M |
| CPU, 24 threads (all cores) | 4.73 M |

Read it plainly: on an RTX 3080 the GPU steps about as fast as a whole 12-core desktop CPU, or about nine CPU cores. The
point is where the state lives, not raw tick speed: the arenas sit in GPU memory beside the policy, so a GPU training loop
never copies observations and controls through the host, and the CPU stays free. Cost per tick depends on the state:
contact (pinches, car-car, the ball against the goal frame) is the most expensive, and quiet states such as a kickoff
approach run several times faster (the example below reports about 31 M arena-ticks/s on its 16,384 kickoffs). Larger
cards have more SMs; we have not measured them.

## Accuracy

Measured on an RTX 3080 against V3 `4572287` (all in the test suite):

- **Every phase of the tick.** V3's tick is cut at ten fixed points. From V3's own states, each phase is run on the CPU
  (platform math), on the CPU (correctly rounded math) and on the GPU, and compared with V3 word by word: 100% bit-exact
  on the platform build, and CPU = GPU on every word.
- **Whole decisions.** 5,000 states sampled from V3 play across kickoffs, open play, head-on drives, corner pinches and
  boost-pad contention, eight ticks each with fresh controls: the platform build equals V3 on 100% of states; the GPU
  equals the correctly rounded build on 100%, and equals V3 on 99.96% of states, every remaining difference attributed to
  `sin`/`cos` rounding. Every event (touches, bumps, demos, pickups, goals) agrees.
- **State completeness.** 1,410,170 tick and phase boundaries from 102 V3 scenarios are converted to the GPU state and back
  and finished by V3 bit-identically, so the GPU state holds everything V3 needs.
- **The math layer.** 880 million vector-math cases against glam 0.34.1 and every 32-bit input of `sinf`, `cosf`,
  `tanf` (plus 10^9 cases of each other function) on the GPU against the CPU: 0 mismatches.

## Requirements

- An NVIDIA GPU and driver. Tested on an RTX 3080 (sm_86); any architecture the CUDA toolkit supports builds.
- CUDA toolkit 13.2 or newer (only `nvcc` at build time; at run time just the driver).
- Rust (the toolchain is pinned in `rust-toolchain.toml`), `git` and `bash` (Git Bash on Windows).
- Windows (MSVC) or Linux.
- RocketSim's collision meshes. They come from your own copy of the game and are not distributed; dump them with
  [RLArenaCollisionDumper](https://github.com/ZealanL/RLArenaCollisionDumper) as for RocketSim itself.

## Quick start

```bash
git clone https://github.com/akaAMAZING/rocketsim-cuda
cd rocketsim-cuda
bash v3/fetch_v3.sh                     # RocketSim V3 at the pinned commit, hash-checked, with the dump patch applied
cp .cargo/config.example.toml .cargo/config.toml   # then set CUDA_HOME (and optionally ROCKETSIM_CUDA_ARCHS)
export ROCKETSIM_MESHES=/path/to/collision_meshes  # the folder that holds soccar/*.cmf
cargo run --release --example kickoff -- 16384 300
```

The first build compiles the kernels for the GPUs `nvidia-smi` reports (set `ROCKETSIM_CUDA_ARCHS`, e.g. `86,120`, to build
for others) and takes several minutes.

## Usage

V3 builds arenas and is the source of every state; the GPU steps them. In Rust:

```rust
use rocketsim::{Arena, ArenaConfig, CarBodyConfig, GameMode, Team};
use rocketsim_cuda::{device::{SIM8_BLOCK, SimDevice}, dump::{parse_arena, parse_static}, layout::{Controls, NUM_CARS}};

let mut v3 = Arena::new_with_config(ArenaConfig::new(GameMode::Soccar));
v3.add_car(Team::Blue, CarBodyConfig::OCTANE);
v3.add_car(Team::Orange, CarBodyConfig::OCTANE);
v3.reset_to_random_kickoff(None);

let sd = parse_static(v3.export_static().as_bytes())?;              // collision data and constants, once per car pairing
let (state, _) = parse_arena(v3.export_state().as_bytes(), &sd)?;   // the full arena state
let dev = SimDevice::new(0)?;
let dsd = dev.upload_static(&sd)?;
let mut states = dev.upload_states(&vec![state; 16_384])?;           // stored AoSoA-32 on the card

let ctl = Controls { throttle: 1.0, steer: 0.0, pitch: 0.0, yaw: 0.0, roll: 0.0, jump: 0, boost: 1, handbrake: 0 };
let controls = dev.upload_controls(&vec![[ctl; NUM_CARS]; 16_384])?;
let mut result = dev.stream.alloc_zeros::<u32>(16_384)?;
dev.sim8(&mut states, &dsd, &controls, &mut result, SIM8_BLOCK, None)?; // 8 ticks of every arena
let back = dev.download_states(&states)?;                           // or read the buffers in your own kernels
```

- `sim8` advances every arena eight ticks with that arena's controls. `result[a]` holds the tick a goal was scored on (0 for
  none; the arena then stops at that tick), plus a flag bit if a tick met an unsupported collision pair or a capacity limit.
- To write your own kernels against the state (observations, rewards, resets on the GPU), include `cuda/sim/arena_step.cuh`
  and call `step_ticks` per arena; `ArenaState` and `StaticData` are plain structs in `cuda/sim/arena_state.cuh`, mirrored
  field for field in `crates/rocketsim-cuda/src/layout.rs`.
- One `StaticData` serves every arena with the same car bodies and team order.
- A state written back by `write_arena` can be imported into V3 again, so the CPU and GPU can hand arenas back and forth.

## Tests

```bash
export ROCKETSIM_MESHES=/path/to/collision_meshes
cargo test --release --workspace                   # about 10 minutes on an RTX 3080 (the libm test sweeps all 2^32 inputs)
cargo test --release -p rocketsim-cuda --test speed -- --ignored --nocapture  # throughput; only meaningful on an idle machine
```

The tests draw their states from V3 itself (`v3/v3dump`: kickoffs, open play, head-on drives, corner pinches, pad
contention). `ROCKETSIM_CUDA_SAMPLES` sets the states per class.

Comments in `cuda/` cite the V3 source line each function mirrors. The citations refer to V3 `ad8b38c` (v0.3.1), where the
port began; the code that v0.3.3 changed (ball sleeping, boost-pad membership) cites the pinned `4572287`.

## License

MIT (`LICENSE`). RocketSim (MIT), Bullet Physics (zlib), glam (MIT or Apache-2.0) and CORE-MATH (MIT) are credited in
`THIRD_PARTY_NOTICES.md`. All credit for the physics belongs to RocketSim's authors; this project only moves it to the GPU.
