//! Steps many copies of a 1v1 kickoff on the GPU. V3 builds the arena and resets it to a kickoff; its state is exported,
//! imported as an `ArenaState`, replicated to N arenas and advanced by the decision kernel (8 ticks per launch) with both cars
//! driving and boosting straight ahead (the blue car starts on the centre line, so it drives into the ball).
//!   ROCKETSIM_MESHES=<collision_meshes dir> cargo run --release --example kickoff -- [arenas] [decisions]
//! Defaults: 16384 arenas, 300 decisions (300 x 8 ticks = 20 s of game time).

use std::time::Instant;

use rocketsim::{Arena, ArenaConfig, CarBodyConfig, GameMode, Team};
use rocketsim_cuda::{
    device::{SIM8_BLOCK, SimDevice, TICKS_PER_DECISION},
    dump::{parse_arena, parse_static},
    layout::{Controls, NUM_CARS},
};

fn main() {
    let mut args = std::env::args().skip(1);
    let arenas: usize = args.next().map_or(16_384, |a| a.parse().expect("arenas: a number"));
    let decisions: usize = args.next().map_or(300, |a| a.parse().expect("decisions: a number"));

    // V3 builds the arena (Octane against Octane) and places both cars at a kickoff.
    v3dump::meshes::init_meshes();
    let mut v3 = Arena::new_with_config(ArenaConfig::new(GameMode::Soccar));
    v3.add_car(Team::Blue, CarBodyConfig::OCTANE);
    v3.add_car(Team::Orange, CarBodyConfig::OCTANE);
    // Kickoff seeds pick the spawn spots; take one that puts the blue car on the centre line, driving straight at the ball.
    let seed = (0..1000).find(|&s| {
        v3.reset_to_random_kickoff(Some(s));
        v3.get_car_state(0).phys.pos.x == 0.0
    });
    v3.reset_to_random_kickoff(Some(seed.expect("a kickoff seed with the blue car on the centre line")));

    // The static dump (collision data and constants) is uploaded once; the state dump is replicated to every arena.
    let sd = parse_static(v3.export_static().as_bytes()).expect("static dump");
    let (state, _) = parse_arena(v3.export_state().as_bytes(), &sd).expect("arena dump");
    let dev = SimDevice::new(0).expect("a CUDA device");
    let dsd = dev.upload_static(&sd).unwrap();
    let mut states = dev.upload_states(&vec![state; arenas]).unwrap();

    let drive = Controls { throttle: 1.0, steer: 0.0, pitch: 0.0, yaw: 0.0, roll: 0.0, jump: 0, boost: 1, handbrake: 0 };
    let controls = dev.upload_controls(&vec![[drive; NUM_CARS]; arenas]).unwrap();
    let mut result = dev.stream.alloc_zeros::<u32>(arenas).unwrap();

    let mut launch = |states: &mut _| dev.sim8(states, &dsd, &controls, &mut result, SIM8_BLOCK, None).unwrap();
    launch(&mut states); // warm-up: loads the module's code onto the card
    dev.stream.synchronize().unwrap();
    let t = Instant::now();
    for _ in 0..decisions {
        launch(&mut states);
    }
    dev.stream.synchronize().unwrap();
    let secs = t.elapsed().as_secs_f64();

    let out = dev.download_states(&states).unwrap();
    let ball = out[0].ball.pos;
    println!("{arenas} arenas, {decisions} decisions of {TICKS_PER_DECISION} ticks, {:.1} game seconds", (decisions + 1) as f64 * TICKS_PER_DECISION as f64 / 120.0);
    println!("arena 0 ball position (uu): ({:.1}, {:.1}, {:.1})", ball.x, ball.y, ball.z);
    println!("{:.2} M arena-ticks/s ({:.2} s)", arenas as f64 * (decisions * TICKS_PER_DECISION) as f64 / secs / 1e6, secs);
}
