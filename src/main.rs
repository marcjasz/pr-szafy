#![deny(warnings)]
extern crate mpi;

use mpi::Threading;
use std::sync::Arc;
use std::thread;
mod comm;
mod agent;
mod util;

// mpirun -n 4 target/debug/rust-pg

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
    let universe = init_mpi();
    let universe_main = Arc::new(universe);
    let universe_comm = Arc::clone(&universe_main);
    let comm_handle = thread::spawn(move || {
        comm::comm_thread(universe_comm.world());
    });
    agent::main_loop(universe_main.world());
    comm_handle.join().unwrap();
}
