#![deny(warnings)]
extern crate mpi;

use mpi::point_to_point as p2p;
use mpi::topology::Rank;
use mpi::traits::*;
use mpi::Threading;
use ansi_term::Colour::Fixed;
use std::sync::Arc;
use std::thread;

// mpirun -n 4 target/debug/rust-pg
fn comm_thread(world: mpi::topology::SystemCommunicator) {
    let size = world.size();
    let rank = world.rank();

    let next_rank = if rank + 1 < size { rank + 1 } else { 0 };
    let next_process = world.process_at_rank(next_rank);
    let previous_rank = if rank > 0 { rank - 1 } else { size - 1 };
    let previous_process = world.process_at_rank(previous_rank);

    let (msg, status): (Rank, _) = p2p::send_receive(&rank, &previous_process, &next_process);
    log(rank, format!(
        "Process {} got message {}. Status is: {:?}",
        rank, msg, status)
    );
}

fn check_threading_support(threading: mpi::Threading){
    println!("Supported level of threading: {:?}", threading);
    assert_eq!(threading, mpi::environment::threading_support());
}

fn log(rank: i32, msg: String) -> () {
    let color_num = ((rank * 48) + (rank / 256)) % 240 + 15;
    let color = Fixed(color_num as u8);
    println!("{}. {}", rank, color.paint(msg));
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
        comm_thread(universe_comm.world());
    });
    comm_handle.join().unwrap();
}
