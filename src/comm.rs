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
        *time = std::cmp::max(*time, other_time);
        *time += 1;
        return *time;
    }

    pub fn time(&self) -> u16 {
        return *self.time.read().unwrap();
    }
}

#[derive(Clone)]
pub struct TimestampedCommunicator<'a> {
    pub clock: &'a Clock,
    world: &'a mpi::topology::SystemCommunicator,
}

impl<'a> TimestampedCommunicator<'a> {
    pub fn new(clock: &'a Clock, world: &'a mpi::topology::SystemCommunicator) -> Self {
        Self { clock, world }
    }

    pub fn send_with_time(&self, message: &Vec<u16>, receiver_rank: i32, tag: i32) {
        let mut timestamped_message = message.clone();
        let time = self.clock.time();
        timestamped_message.push(time);
        self.world
            .process_at_rank(receiver_rank)
            .send_with_tag(&timestamped_message[..], tag);
    }

    pub fn broadcast_with_time(&self, message: &Vec<u16>, tag: i32) {
        let mut timestamped_message = message.clone();
        let time = self.clock.time();
        timestamped_message.push(time);
        for i in 0..self.world.size() {
            if i == self.world.rank() {
                continue;
            }
            self.world
                .process_at_rank(i)
                .send_with_tag(&timestamped_message[..], tag);
        }
    }

    pub fn receive(&self) -> (Vec<u16>, p2p::Status) {
        let (message, status) = self.world.any_process().receive_vec::<u16>();
        util::sleep_random();
        self.clock.inc_compare(message.last().copied().unwrap_or(0));
        return (message, status);
    }
}

pub fn receiver_loop(agent: &agent::Agent) {
    loop {
        let (message, status): (Vec<u16>, p2p::Status) = agent.receive_request();
        let message_timestamp = message.last().copied().unwrap_or(0);

        match MessageTag::from_i32(status.tag()) {
            MessageTag::Resources => {
                agent.handle_resources(status.source_rank(), message[0], message[1]);
            }
            MessageTag::EnterRequest => {
                agent.handle_enter_request(status.source_rank(), message_timestamp)
            }
            MessageTag::LeaveRequest => {
                agent.handle_leave_request(status.source_rank(), message_timestamp)
            }
            MessageTag::Finish => {
                agent.finish();
                break;
            }
            MessageTag::LeaveResources => {
                agent.handle_leave_resources(status.source_rank());
            }
        }
    }
}
