use crate::agent;
use crate::util;
use crate::MessageTag;
use mpi::point_to_point as p2p;
use mpi::traits::*;
use std::sync::RwLock;

#[derive(Copy, Clone)]
pub struct Clock {
    pub time: u16,
}

impl Clock {
    pub fn new() -> Self {
        Self { time: 0 }
    }

    pub fn inc(&mut self) -> u16 {
        self.time += 1;
        return self.time;
    }

    pub fn inc_compare(&mut self, other_time: u16) -> u16 {
        self.time = std::cmp::max(self.time, other_time) + 1;
        return self.time;
    }
}

pub fn send_with_time(
    clock: &RwLock<Clock>,
    world: &mpi::topology::SystemCommunicator,
    message: &Vec<u16>,
    receiver_rank: i32,
    tag: i32,
) {
    let mut timestamped_message = message.clone();
    let time = clock.write().unwrap().inc();
    timestamped_message.push(time);
    world
        .process_at_rank(receiver_rank)
        .send_with_tag(&timestamped_message[..], tag);
}

pub fn broadcast_with_time(
    clock: &RwLock<Clock>,
    world: &mpi::topology::SystemCommunicator,
    message: &Vec<u16>,
    tag: i32,
) {
    let mut timestamped_message = message.clone();
    let time = clock.write().unwrap().inc();
    timestamped_message.push(time);
    for i in 0..world.size() {
        if i == world.rank() {
            continue;
        }
        world
            .process_at_rank(i)
            .send_with_tag(&timestamped_message[..], tag);
    }
}

pub fn receive(
    clock: &RwLock<Clock>,
    world: &mpi::topology::SystemCommunicator,
) -> (Vec<u16>, p2p::Status) {
    let (message, status) = world.any_process().receive_vec::<u16>();
    clock
        .write()
        .unwrap()
        .inc_compare(message.last().copied().unwrap_or(0));
    return (message, status);
}

pub fn receiver_loop(
    agent: &RwLock<agent::Agent>,
    world: &mpi::topology::SystemCommunicator,
    clock: &RwLock<Clock>,
) {
    let logger = util::Logger::new(&clock, world.rank());
    loop {
        let (message, status): (Vec<u16>, p2p::Status) = receive(&clock, &world);
        let message_timestamp = message.last().copied().unwrap_or(0);

        logger.log(format!(
            "Got message {:?}. Status is: {:?}",
            message, status
        ));

        match MessageTag::from_i32(status.tag()) {
            MessageTag::EnterRequest => agent.write().unwrap().handle_request(
                clock,
                world,
                &logger,
                status.source_rank() as i32,
                message_timestamp,
            ),
            MessageTag::Finish => break,
            _ => (),
        }
    }
    logger.log("Exiting".to_string());
}
