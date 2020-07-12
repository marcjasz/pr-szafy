use mpi::topology::Rank;
use mpi::traits::*;
use mpi::point_to_point as p2p;
use crate::util;

pub fn comm_thread(world: mpi::topology::SystemCommunicator) {
    let size = world.size();
    let rank = world.rank();
    let next_rank = if rank + 1 < size { rank + 1 } else { 0 };
    let next_process = world.process_at_rank(next_rank);
    let previous_rank = if rank > 0 { rank - 1 } else { size - 1 };
    let previous_process = world.process_at_rank(previous_rank);

    let (msg, status): (Rank, _) = p2p::send_receive(&rank, &previous_process, &next_process);
    util::log(rank, format!(
        "Process {} got message {}. Status is: {:?}",
        rank, msg, status)
    );
}
