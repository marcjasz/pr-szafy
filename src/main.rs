#![deny(warnings)]
extern crate mpi;
extern crate ctrlc;

use mpi::{
    Threading,
    traits::*
};
use std::sync::{Arc, RwLock};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use mpi::point_to_point as p2p;
mod comm;
mod agent;
mod util;

pub enum MessageTag {
    Resources,
    EnterRequest,
    LeaveRequest,
    Finish,
}

pub struct CommonState<'world_lifetime> {
    world: &'world_lifetime mpi::topology::SystemCommunicator,
    clock: &'world_lifetime RwLock<comm::Clock>,
    rooms_count: u8,
    _lifts_count: u8,
}

impl<'world_lifetime> CommonState<'world_lifetime> {
    fn new(
        world: &'world_lifetime mpi::topology::SystemCommunicator, 
        clock: &'world_lifetime RwLock<comm::Clock>, 
        rooms_count: u8, 
        _lifts_count: u8
    ) -> Self {
        Self {
            world,
            clock,
            rooms_count,
            _lifts_count,
        }
    }
}

// mpirun -n 4 target/debug/pr-szafy

fn check_threading_support(threading: mpi::Threading){
    println!("Supported level of threading: {:?}", threading);
    assert_eq!(threading, mpi::environment::threading_support());
}

fn init_mpi() -> mpi::environment::Universe {
    let (universe, threading) = mpi::initialize_with_threading(Threading::Multiple).unwrap();
    check_threading_support(threading);
    universe
}

fn main() {
    let universe_main = Arc::new(init_mpi());
    let world = universe_main.world();
    let universe_comm = Arc::clone(&universe_main);
    let clock_main = Arc::new(RwLock::new(comm::Clock::new()));
    let clock_comm = clock_main.clone();
    let common_state = CommonState::new(&world, &clock_main, 5, 3);
    let is_alive = Arc::new(AtomicBool::new(true));
    let is_alive_ctrlc = is_alive.clone();
    let clock_ctrlc = clock_main.clone();
    let world_ctrlc = universe_main.clone().world();

    ctrlc::set_handler(move || {
        is_alive_ctrlc.store(false, Ordering::SeqCst);
        comm::broadcast_with_tag(&clock_ctrlc, &world_ctrlc, &vec![], MessageTag::Finish as i32);
    }).expect("Error while setting Ctrl-C handler");

    let comm_handle = thread::spawn(move || {
        let world = universe_comm.world();
        let rank = world.rank();
        let logger = util::Logger::new(&clock_comm, rank);
        loop {
            let (message, status): (Vec<u16>, p2p::Status) = comm::receive(&clock_comm, &world);

            logger.log(format!(
                "Got message {:?}. Status is: {:?}",
                message, status)
            );

            if status.tag() == MessageTag::Finish as i32 { break; }
        }
        logger.log("Exiting".to_string());
    });
    
    agent::main_loop(&common_state, &is_alive);
    comm_handle.join().unwrap();
}
