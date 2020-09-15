use crate::agent;
use crate::util;
use crate::MessageTag;
use mpi::point_to_point as p2p;
use mpi::traits::*;
use std::sync::RwLock;

pub struct Clock {
    time: RwLock<u16>,
}

impl Clock {
    pub fn new() -> Self {
        Self {
            time: RwLock::new(0),
        }
    }

    pub fn inc(&self) -> u16 {
        let mut time = self.time.write().unwrap();
        *time += 1;
        return *time;
    }

    pub fn inc_compare(&self, other_time: u16) -> u16 {
        let mut time = self.time.write().unwrap();
        *time = std::cmp::max(*time, other_time) + 1;
        return *time;
    }

    pub fn time(&self) -> u16 {
        return *self.time.read().unwrap();
    }
}

pub fn send_with_time(
    clock: &Clock,
    world: &mpi::topology::SystemCommunicator,
    message: &Vec<u16>,
    receiver_rank: i32,
    tag: i32,
) {
    let mut timestamped_message = message.clone();
    let time = clock.inc();
    timestamped_message.push(time);
    world
        .process_at_rank(receiver_rank)
        .send_with_tag(&timestamped_message[..], tag);
}

pub fn broadcast_with_time(
    clock: &Clock,
    world: &mpi::topology::SystemCommunicator,
    message: &Vec<u16>,
    tag: i32,
) {
    let mut timestamped_message = message.clone();
    let time = clock.inc();
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
    clock: &Clock,
    world: &mpi::topology::SystemCommunicator,
) -> (Vec<u16>, p2p::Status) {
    let (message, status) = world.any_process().receive_vec::<u16>();
    clock.inc_compare(message.last().copied().unwrap_or(0));
    return (message, status);
}

pub fn receiver_loop(
    agent: &RwLock<agent::Agent>,
    world: &mpi::topology::SystemCommunicator,
    clock: &Clock,
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
