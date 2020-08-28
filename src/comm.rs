use mpi::traits::*;
use mpi::point_to_point as p2p;
use crate::util;

pub fn comm_thread(world: mpi::topology::SystemCommunicator) {
    let rank = world.rank();
    loop {
        let (message, status): (Vec<u8>, p2p::Status) = world.any_process().receive_vec::<u8>();

        util::log(rank, format!(
            "Process {} got message {:?}. Status is: {:?}",
            rank, message, status)
        );
    }
}
