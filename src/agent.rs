#![allow(dead_code)]
use crate::comm;
use crate::util;
use crate::CommonState;
use crate::MessageTag;
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

    fn leave_time(&self) -> u16 {
        *self.leave_time.read().unwrap()
    }

    fn set_leave_time(&self) {
        *self.leave_time.write().unwrap() = self.clock.time();
    }

    fn rooms_at_rank(&self, rank: i32) -> u8 {
        self.rooms.read().unwrap()[rank as usize]
    }

    fn set_rooms_at_rank(&self, rank: i32, rooms: u8) {
        self.rooms.write().unwrap()[rank as usize] = rooms;
    }

    fn taken_rooms_count(&self) -> u8 {
        self.rooms.read().unwrap().iter().sum()
    }

    fn lifts_at_rank(&self, rank: i32) -> u8 {
        self.lifts.read().unwrap()[rank as usize]
    }

    fn set_lifts_at_rank(&self, rank: i32, lifts: u8) {
        self.lifts.write().unwrap()[rank as usize] = lifts;
    }

    fn taken_lifts_count(&self) -> u8 {
        self.lifts.read().unwrap().iter().sum()
    }

    fn higher_priority(&self, my_time: u16, sender_time: u16, sender_rank: i32) -> bool {
        (my_time > sender_time) || (my_time == sender_time && self.rank > sender_rank)
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
            AgentState::Rest => {
                logger.log("Sending requests for resources to go down".to_string());
                self.set_enter_time();
                communicator.broadcast_with_time(&vec![], MessageTag::EnterRequest as i32);
                *self.rooms.write().unwrap() = (0..self.common_state.world_size)
                    .map(|index| {
                        if index == self.rank as usize {
                            self.need
                        } else {
                            self.common_state.rooms_count
                        }
                    })
                    .collect();
                *self.lifts.write().unwrap() = vec![1; self.common_state.world_size];
                self.next_state();
            }
            AgentState::Try => {
                logger.log("Trying to go down".to_string());
                if self.taken_rooms_count() <= self.common_state.rooms_count
                    && self.taken_lifts_count() <= self.common_state.lifts_count
                {
                    logger.log("Going down".to_string());
                    self.next_state();
                }
            }
            AgentState::Down => {
                let deferred_lift_msg = vec![0, 1];
                logger.log("Entering the critical section".to_string());
                self.defer_lifts
                    .write()
                    .unwrap()
                    .drain(..)
                    .for_each(|rank| {
                        logger.log(format!("Granting a deferred lift to {}", rank));
                        communicator.send_with_time(
                            &deferred_lift_msg,
                            rank as i32,
                            MessageTag::Resources as i32,
                        );
                    });
                self.next_state();
            }
            AgentState::Crit => {
                *self.lifts.write().unwrap() = vec![1; self.common_state.world_size];
                self.set_leave_time();
                logger.log("Sending requests for resources to go up".to_string());
                communicator.broadcast_with_time(&vec![], MessageTag::LeaveRequest as i32);
                self.next_state();
            }
            AgentState::Leaving => {
                logger.log("Trying to leave the critical section".to_string());
                if self.taken_lifts_count() <= self.common_state.lifts_count {
                    logger.log("Leaving the critical section".to_string());
                    self.next_state();
                }
            }
            AgentState::Up => {
                let deferred_lifts: Vec<_> = self.defer_lifts.write().unwrap().drain(..).collect();
                self.defer_rooms
                    .write()
                    .unwrap()
                    .drain(..)
                    .map(|rank| match deferred_lifts.iter().find(|&&r| r == rank) {
                        Some(_) => (rank, vec![self.need as u16, 1]),
                        None => (rank, vec![self.need as u16, 0]),
                    })
                    .inspect(|(rank, message)| {
                        logger.log(format!(
                            "Granting {} deferred rooms and {} lift to process #{}",
                            message[0],
                            if message[1] == 1 { "a" } else { "no" },
                            rank
                        ))
                    })
                    .for_each(|(rank, message)| {
                        communicator.send_with_time(
                            &message,
                            rank as i32,
                            MessageTag::Resources as i32,
                        );
                    });
                logger.log("Going to rest".to_string());
                self.next_state();
            }
        }
    }

    pub fn receive_request(&self) -> (std::vec::Vec<u16>, mpi::point_to_point::Status) {
        comm::TimestampedCommunicator::new(&self.clock, &self.world).receive()
    }

    pub fn handle_enter_request(&self, sender_rank: i32, request_time: u16) {
        let communicator = comm::TimestampedCommunicator::new(&self.clock, &self.world);
        let logger = util::Logger::new(&self.clock, self.rank);
        let message: Vec<u16>;
        if matches!(self.state(), AgentState::Rest)
            || self.higher_priority(self.enter_time(), request_time, sender_rank)
        {
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

    pub fn handle_leave_request(&self, sender_rank: i32, request_time: u16) {
        if matches!(self.state(), AgentState::Leaving | AgentState::Up)
            && self.higher_priority(self.leave_time(), request_time, sender_rank)
        {
            self.defer_lifts.write().unwrap().push(sender_rank as u8);
        } else {
            let message = vec![0, 1];
            let communicator = comm::TimestampedCommunicator::new(&self.clock, &self.world);
            self.set_lifts_at_rank(sender_rank, self.lifts_at_rank(sender_rank) + 1);
            communicator.send_with_time(&message, sender_rank, MessageTag::Resources as i32);
        }
    }

    pub fn handle_resources(&self, sender_rank: i32, rooms: u16, lifts: u16) {
        self.set_rooms_at_rank(sender_rank, self.rooms_at_rank(sender_rank) - rooms as u8);
        self.set_lifts_at_rank(sender_rank, self.lifts_at_rank(sender_rank) - lifts as u8);
        util::Logger::new(&self.clock, self.rank).log(format!(
            "Received resource information: {} rooms and {} lifts claimed by process #{}",
            self.rooms_at_rank(sender_rank),
            self.lifts_at_rank(sender_rank),
            sender_rank,
        ));
    }

    pub fn finish(&self) {
        util::Logger::new(&self.clock, self.rank).log("Exiting".to_string());
    }
}

pub fn main_loop<'world_lifetime>(
    agent: &'world_lifetime Agent,
    is_alive: &'world_lifetime AtomicBool,
) {
    while is_alive.load(Ordering::SeqCst) {
        agent.run();
        util::sleep_random();
    }
}
