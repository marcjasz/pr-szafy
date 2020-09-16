#![allow(dead_code)]
use crate::comm;
use crate::util;
use crate::CommonState;
use crate::MessageTag;
use rand::Rng;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::RwLock;

pub struct Agent {
    need: u8,
    rank: i32,
    enter_time: RwLock<u16>,
    leave_time: RwLock<u16>,
    rooms: RwLock<Vec<u8>>,
    lifts: RwLock<Vec<u8>>,
    defer_rooms: RwLock<Vec<u8>>,
    defer_lifts: RwLock<Vec<u8>>,
    state: RwLock<AgentState>,
    common_state: CommonState,
    world: mpi::topology::SystemCommunicator,
    clock: comm::Clock,
}

#[derive(Clone, Copy, Debug)]
enum AgentState {
    Rest,
    Try,
    Down,
    Crit,
    Leaving,
    Up,
}

impl Agent {
    pub fn new(
        rank: i32,
        need: u8,
        common_state: CommonState,
        world: mpi::topology::SystemCommunicator,
    ) -> Self {
        Self {
            need,
            rank,
            state: RwLock::new(AgentState::Rest),
            enter_time: RwLock::new(u16::MAX),
            leave_time: RwLock::new(u16::MAX),
            rooms: RwLock::new(vec![0; common_state.world_size]),
            lifts: RwLock::new(vec![0; common_state.world_size]),
            defer_rooms: RwLock::new(Vec::new()),
            defer_lifts: RwLock::new(Vec::new()),
            common_state,
            clock: comm::Clock::new(),
            world,
        }
    }

    fn state(&self) -> AgentState {
        *self.state.read().unwrap()
    }

    fn set_state(&self, state: AgentState) {
        *self.state.write().unwrap() = state;
    }

    fn enter_time(&self) -> u16 {
        *self.enter_time.read().unwrap()
    }

    fn set_enter_time(&self) {
        *self.enter_time.write().unwrap() = self.clock.time();
    }

    fn next_state(&self) {
        let new_state = match self.state() {
            AgentState::Rest => AgentState::Try,
            AgentState::Try => AgentState::Down,
            AgentState::Down => AgentState::Crit,
            AgentState::Crit => AgentState::Leaving,
            AgentState::Leaving => AgentState::Up,
            AgentState::Up => AgentState::Rest,
        };
        self.clock.inc();
        self.set_state(new_state);
    }

    fn run(&self) {
        let communicator = comm::TimestampedCommunicator::new(&self.clock, &self.world);
        let logger = util::Logger::new(&self.clock, self.rank);
        match self.state() {
            AgentState::Try => {
                logger.log("Trying to go down".to_string());
                self.set_enter_time();
                communicator.broadcast_with_time(&vec![], MessageTag::EnterRequest as i32);
                *self.rooms.write().unwrap() = (1..self.common_state.world_size)
                    .map(|index| {
                        if index == self.rank as usize {
                            self.need
                        } else {
                            self.common_state.rooms_count
                        }
                    })
                    .collect();
                *self.lifts.write().unwrap() = vec![1; self.common_state.world_size];
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

    pub fn receive_request(&self) -> (std::vec::Vec<u16>, mpi::point_to_point::Status) {
        comm::TimestampedCommunicator::new(&self.clock, &self.world).receive()
    }

    pub fn handle_request(&self, sender_rank: i32, request_time: u16) {
        let communicator = comm::TimestampedCommunicator::new(&self.clock, &self.world);
        let logger = util::Logger::new(&self.clock, self.rank);
        let message: Vec<u16>;
        if matches!(self.state(), AgentState::Rest) || self.enter_time() > request_time {
            message = vec![self.common_state.rooms_count as u16, 1];
        } else if matches!(self.state(), AgentState::Crit) {
            message = vec![(self.common_state.rooms_count - self.need) as u16, 1];
            self.defer_rooms.write().unwrap().push(sender_rank as u8);
        } else {
            message = vec![(self.common_state.rooms_count - self.need) as u16, 0];
            self.defer_rooms.write().unwrap().push(sender_rank as u8);
            self.defer_lifts.write().unwrap().push(sender_rank as u8);
        }
        logger.log(format!(
            "Received a resource request, granting {} rooms and {} lift to process #{}",
            message[0],
            if message[1] == 1 { "a" } else { "no" },
            sender_rank
        ));
        communicator.send_with_time(&message, sender_rank, MessageTag::Resources as i32);
    }

    pub fn finish(&self) {
        util::Logger::new(&self.clock, self.rank).log("Exiting".to_string());
    }
}

pub fn main_loop<'world_lifetime>(
    agent: &'world_lifetime Agent,
    is_alive: &'world_lifetime AtomicBool,
) {
    let mut rng = rand::thread_rng();
    while is_alive.load(Ordering::SeqCst) {
        agent.run();
        agent.next_state();
        let secs = rng.gen_range(1, 8);
        std::thread::sleep(std::time::Duration::from_secs(secs));
    }
}
