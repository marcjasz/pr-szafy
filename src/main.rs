#![deny(warnings)]
extern crate ctrlc;
extern crate mpi;

use mpi::{traits::*, Threading};
use rand::Rng;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};
use std::thread;
mod agent;
mod comm;
mod util;

pub enum MessageTag {
    Resources,
    EnterRequest,
    LeaveRequest,
    Finish,
}

impl MessageTag {
    pub fn from_i32(int: i32) -> Self {
        match int {
            0 => MessageTag::Resources,
            1 => MessageTag::EnterRequest,
            2 => MessageTag::LeaveRequest,
            3 => MessageTag::Finish,
            _ => panic!("invalid message tag"),
        }
    }
}

#[derive(Clone)]
pub struct CommonState {
    rooms_count: u8,
    _lifts_count: u8,
    world_size: usize,
}

impl CommonState {
    fn new(rooms_count: u8, _lifts_count: u8, world_size: usize) -> Self {
        Self {
            rooms_count,
            _lifts_count,
            world_size,
        }
    }
}

// mpirun -n 4 target/debug/pr-szafy

fn check_threading_support(threading: mpi::Threading) {
    println!("Supported level of threading: {:?}", threading);
    assert_eq!(threading, mpi::environment::threading_support());
}

fn init_mpi() -> mpi::environment::Universe {
    let (universe, threading) = mpi::initialize_with_threading(Threading::Multiple).unwrap();
    check_threading_support(threading);
    universe
}

fn handle_ctrlc(
    is_alive: &AtomicBool,
    clock: &RwLock<comm::Clock>,
    world: &mpi::topology::SystemCommunicator,
) {
    is_alive.store(false, Ordering::SeqCst);
    comm::broadcast_with_time(clock, world, &vec![], MessageTag::Finish as i32);
}

fn main() {
    let universe = init_mpi();
    let world = Arc::new(universe.world());
    let clock = Arc::new(RwLock::new(comm::Clock::new()));
    let is_alive = Arc::new(AtomicBool::new(true));
    let rank = world.rank();

    let is_alive_ctrlc = is_alive.clone();
    let clock_ctrlc = clock.clone();
    let world_ctrlc = world.clone();
    ctrlc::set_handler(move || handle_ctrlc(&is_alive_ctrlc, &clock_ctrlc, &world_ctrlc))
        .expect("Error while setting Ctrl-C handler");

    let common_state = CommonState::new(5, 3, world.size() as usize);
    let agent = Arc::new(RwLock::new(agent::Agent::new(
        rank,
        rand::thread_rng().gen_range(1, common_state.rooms_count),
        common_state,
    )));

    let world_agent = world.clone();
    let clock_agent = clock.clone();
    let agent_main = agent.clone();
    let agent_handle =
        thread::spawn(move || agent::main_loop(&agent_main, &is_alive, &world_agent, &clock_agent));

    let world_comm = world.clone();
    let clock_comm = clock.clone();
    let agent_comm = agent.clone();
    let comm_handle =
        thread::spawn(move || comm::receiver_loop(&agent_comm, &world_comm, &clock_comm));

    comm_handle.join().unwrap();
    agent_handle.join().unwrap();
}
