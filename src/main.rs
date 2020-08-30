#![deny(warnings)]
extern crate mpi;

use mpi::{
    Threading,
    traits::*
};
use std::sync::Arc;
use std::sync::RwLock;
use std::thread;
use mpi::point_to_point as p2p;
mod comm;
mod agent;
mod util;

pub enum MessageTag {
    Resources = 1,
    EnterRequest = 2,
    LeaveRequest = 3,
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
    let universe_comm = Arc::clone(&universe_main);
    let clock_main = Arc::new(RwLock::new(comm::Clock::new()));
    let clock_comm = clock_main.clone();
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
        }
    });
    agent::main_loop(&clock_main, &universe_main.world());
    comm_handle.join().unwrap();
}
