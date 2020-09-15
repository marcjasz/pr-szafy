#![allow(dead_code)]
use crate::comm;
use crate::util;
use crate::CommonState;
use crate::MessageTag;
use mpi::traits::*;
use rand::Rng;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::RwLock;

#[derive(Clone)]
pub struct Agent {
    need: u8,
    rank: i32,
    enter_time: u16,
    leave_time: u16,
    rooms: Vec<u8>,
    lifts: Vec<u8>,
    defer_rooms: Vec<u8>,
    defer_lifts: Vec<u8>,
    state: AgentState,
    common_state: CommonState,
}

#[derive(Clone, Debug)]
enum AgentState {
    Rest,
    Try,
    Down,
    Crit,
    Leaving,
    Up,
}

impl Agent {
    pub fn new(rank: i32, need: u8, common_state: CommonState) -> Self {
        Self {
            need,
            rank,
            state: AgentState::Rest,
            enter_time: u16::MAX,
            leave_time: u16::MAX,
            rooms: vec![0; common_state.world_size],
            lifts: vec![0; common_state.world_size],
            defer_rooms: Vec::new(),
            defer_lifts: Vec::new(),
            common_state,
        }
    }

    fn next_state(&mut self) {
        let new_state = match self.state {
            AgentState::Rest => AgentState::Try,
            AgentState::Try => AgentState::Down,
            AgentState::Down => AgentState::Crit,
            AgentState::Crit => AgentState::Leaving,
            AgentState::Leaving => AgentState::Up,
            AgentState::Up => AgentState::Rest,
        };
        self.state = new_state;
    }

    fn run(
        &mut self,
        clock: &comm::Clock,
        world: &mpi::topology::SystemCommunicator,
        logger: &util::Logger,
    ) {
        match self.state {
            AgentState::Try => {
                logger.log("Trying to go down".to_string());
                self.enter_time = clock.time();
                comm::broadcast_with_time(clock, world, &vec![], MessageTag::EnterRequest as i32);
                self.rooms = (0..self.common_state.world_size)
                    .map(|index| {
                        if index == self.rank as usize {
                            self.need
                        } else {
                            self.common_state.rooms_count
                        }
                    })
                    .collect();
                self.lifts = vec![1; self.common_state.world_size];
            }
            AgentState::Down => {
                logger.log("Going down".to_string());
            }
            AgentState::Crit => {
                logger.log("Entering the critical section".to_string());
            }
            AgentState::Leaving => {
                logger.log("Leaving the critical section".to_string());
            }
            AgentState::Up => {
                logger.log("Going up".to_string());
            }
            AgentState::Rest => {
                logger.log("Going to rest".to_string());
            }
        }
    }

    pub fn handle_request(
        &mut self,
        clock: &comm::Clock,
        world: &mpi::topology::SystemCommunicator,
        logger: &util::Logger,
        sender_rank: i32,
        request_time: u16,
    ) {
        let message: Vec<u16>;
        if matches!(self.state, AgentState::Rest) || self.enter_time > request_time {
            message = vec![self.common_state.rooms_count as u16, 1];
        } else if matches!(self.state, AgentState::Crit) {
            message = vec![(self.common_state.rooms_count - self.need) as u16, 1];
            self.defer_rooms.push(sender_rank as u8);
        } else {
            message = vec![(self.common_state.rooms_count - self.need) as u16, 0];
            self.defer_rooms.push(sender_rank as u8);
            self.defer_lifts.push(sender_rank as u8);
        }
        logger.log(format!(
            "Received a resource request, granting {} rooms and {} lift to process #{}",
            message[0],
            if message[1] == 1 { "a" } else { "no" },
            sender_rank
        ));
        comm::send_with_time(
            clock,
            world,
            &message,
            sender_rank,
            MessageTag::Resources as i32,
        );
    }
}

pub fn main_loop<'world_lifetime>(
    agent: &'world_lifetime RwLock<Agent>,
    is_alive: &'world_lifetime AtomicBool,
    world: &'world_lifetime mpi::topology::SystemCommunicator,
    clock: &'world_lifetime comm::Clock,
) {
    let logger = util::Logger::new(clock, world.rank());
    let mut rng = rand::thread_rng();
    while is_alive.load(Ordering::SeqCst) {
        clock.inc();
        agent.write().unwrap().run(clock, world, &logger);
        agent.write().unwrap().next_state();
        let secs = rng.gen_range(1, 8);
        std::thread::sleep(std::time::Duration::from_secs(secs));
    }
}
