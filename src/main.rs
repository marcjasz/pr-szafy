#![deny(warnings)]
extern crate ctrlc;
extern crate mpi;

use mpi::{traits::*, Threading};
use rand::Rng;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
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
    lifts_count: u8,
    world_size: usize,
}

impl CommonState {
    fn new(rooms_count: u8, lifts_count: u8, world_size: usize) -> Self {
        Self {
            rooms_count,
            lifts_count,
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

fn handle_ctrlc(is_alive: &AtomicBool, world: &mpi::topology::SystemCommunicator) {
    is_alive.store(false, Ordering::SeqCst);
    let dummy_msg = vec![0];
    for i in 0..world.size() {
        world
            .process_at_rank(i)
            .send_with_tag(&dummy_msg[..], MessageTag::Finish as i32);
    }
}

fn main() {
    let universe = init_mpi();
    let world = Arc::new(universe.world());
    let is_alive = Arc::new(AtomicBool::new(true));
    let rank = world.rank();

    let is_alive_ctrlc = is_alive.clone();
    let world_ctrlc = world.clone();
    ctrlc::set_handler(move || handle_ctrlc(&is_alive_ctrlc, &world_ctrlc))
        .expect("Error while setting Ctrl-C handler");

    let common_state = CommonState::new(7, 3, world.size() as usize);
    let agent = Arc::new(agent::Agent::new(
        rank,
        rand::thread_rng().gen_range(1, common_state.rooms_count),
        common_state,
        universe.world(),
    ));

    let agent_main = agent.clone();
    let agent_handle = thread::Builder::new()
        .name(format!("agent-{}", rank))
        .spawn(move || agent::main_loop(&agent_main, &is_alive))
        .unwrap();

    let agent_comm = agent.clone();
    let comm_handle = thread::Builder::new()
        .name(format!("comm-{}", rank))
        .spawn(move || comm::receiver_loop(&agent_comm))
        .unwrap();

    comm_handle.join().unwrap();
    agent_handle.join().unwrap();
}
