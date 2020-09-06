use mpi::traits::*;
use mpi::point_to_point as p2p;
use std::sync::RwLock;

#[derive(Copy, Clone)]
pub struct Clock {
    pub time: u16
}

impl Clock {
    pub fn new () -> Self {
        Self {
            time: 0
        }
    }

    pub fn inc(&mut self) -> u16 {
        self.time += 1;
        return self.time
    }

    pub fn inc_compare(&mut self, other_time: u16) -> u16 {
        self.time = std::cmp::max(self.time, other_time) + 1;
        return self.time
    }
}

pub fn broadcast_with_tag(
    clock: &RwLock<Clock>, 
    world: &mpi::topology::SystemCommunicator, 
    message: &Vec<u16>, tag: i32
) {
    let mut timestamped_message = message.clone();
    {
        let time = clock.write().unwrap().inc();
        timestamped_message.push(time);
    }
    for i in 0..world.size() {
        world.process_at_rank(i).send_with_tag(&timestamped_message[..], tag);
    }
}

pub fn receive(
    clock: &RwLock<Clock>, 
    world: &mpi::topology::SystemCommunicator
) -> (Vec<u16>, p2p::Status) {
    let (mut message, status) = world.any_process().receive_vec::<u16>();
    {
        clock.write().unwrap().inc_compare(message.pop().unwrap_or(0));
    }
    return (message, status)
}

