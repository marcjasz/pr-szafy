#![deny(warnings)]
extern crate ctrlc;
extern crate mpi;

use mpi::{traits::*, Threading};
use std::env;
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
    LeaveResources,
}

impl MessageTag {
    pub fn from_i32(int: i32) -> Self {
        match int {
            0 => MessageTag::Resources,
            1 => MessageTag::EnterRequest,
            2 => MessageTag::LeaveRequest,
            3 => MessageTag::Finish,
            4 => MessageTag::LeaveResources,
            _ => panic!("invalid message tag"),
        }
    }
}

#[derive(Clone)]
pub struct Config {
    rooms_count: u8,
    lifts_count: u8,
    world_size: usize,
}

impl Config {
    fn new(rooms_count: u8, lifts_count: u8, world_size: usize) -> Self {
        Self {
            rooms_count,
            lifts_count,
            world_size,
        }
    }
}

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
    let args: Vec<String> = env::args().collect();
    let rooms_count: u8;
    let lifts_count: u8;
    match args.len() {
        3 => {
            rooms_count = args[1].parse().unwrap();
            lifts_count = args[2].parse().unwrap();
        }
        _ => {
            panic!("two program arguments required (unsigned integers)");
        }
    }

    let universe = init_mpi();
    let world = Arc::new(universe.world());
    let is_alive = Arc::new(AtomicBool::new(true));
    let rank = world.rank();

    // capture Ctrl-C and kill children gracefully
    let is_alive_ctrlc = is_alive.clone();
    let world_ctrlc = world.clone();
    ctrlc::set_handler(move || handle_ctrlc(&is_alive_ctrlc, &world_ctrlc))
        .expect("Error while setting Ctrl-C handler");

    // set up problem instance
    let config = Config::new(rooms_count, lifts_count, world.size() as usize);
    let agent = Arc::new(agent::Agent::new(rank, config, universe.world()));

    // state machine thread
    let agent_main = agent.clone();
    let agent_handle = thread::Builder::new()
        .name(format!("agent-{}", rank))
        .spawn(move || agent::main_loop(&agent_main, &is_alive))
        .unwrap();

    // message receiver thread
    let agent_comm = agent.clone();
    let comm_handle = thread::Builder::new()
        .name(format!("comm-{}", rank))
        .spawn(move || comm::receiver_loop(&agent_comm))
        .unwrap();

    comm_handle.join().unwrap();
    agent_handle.join().unwrap();
}
