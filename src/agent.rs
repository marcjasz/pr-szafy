#![allow(dead_code)]
use crate::comm;
use crate::util;
use crate::Config;
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
    config: Config,
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
        config: Config,
        world: mpi::topology::SystemCommunicator,
    ) -> Self {
        Self {
            need,
            rank,
            state: RwLock::new(AgentState::Rest),
            enter_time: RwLock::new(u16::MAX),
            leave_time: RwLock::new(u16::MAX),
            rooms: RwLock::new(vec![0; config.world_size]),
            lifts: RwLock::new(vec![0; config.world_size]),
            defer_rooms: RwLock::new(Vec::new()),
            defer_lifts: RwLock::new(Vec::new()),
            config,
            clock: comm::Clock::new(),
            world,
        }
    }

    fn state(&self) -> AgentState {
        *self.state.read().unwrap()
    }

    fn set_state(&self, state: AgentState) {
        *self.state.write().unwrap() = state
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
        self.rooms.write().unwrap()[rank as usize] = rooms
    }

    fn taken_rooms_count(&self) -> u8 {
        self.rooms.read().unwrap().iter().sum()
    }

    fn lifts_at_rank(&self, rank: i32) -> u8 {
        self.lifts.read().unwrap()[rank as usize]
    }

    fn set_lifts_at_rank(&self, rank: i32, lifts: u8) {
        self.lifts.write().unwrap()[rank as usize] = lifts
    }

    fn taken_lifts_count(&self) -> u8 {
        self.lifts.read().unwrap().iter().sum()
    }

    fn sender_priority(&self, my_time: u16, sender_time: u16, sender_rank: i32) -> bool {
        (my_time > sender_time) || (my_time == sender_time && self.rank > sender_rank)
    }

    fn next_state<F>(&self, transition_logic: F)
    where
        F: FnOnce(),
    {
        let mut current_state = self.state.write().unwrap();
        let new_state = match *current_state {
            AgentState::Rest => AgentState::Try,
            AgentState::Try => AgentState::Down,
            AgentState::Down => AgentState::Crit,
            AgentState::Crit => AgentState::Leaving,
            AgentState::Leaving => AgentState::Up,
            AgentState::Up => AgentState::Rest,
        };
        transition_logic();
        self.clock.inc();
        *current_state = new_state;
    }

    fn run(&self) {
        let communicator = comm::TimestampedCommunicator::new(&self.clock, &self.world);
        let logger = util::Logger::new(&self.clock, self.rank);
        match self.state() {
            AgentState::Rest => {
                self.next_state(|| {
                    let mut rooms = self.rooms.write().unwrap();
                    let mut lifts = self.lifts.write().unwrap();
                    logger.log("Sending requests for resources to go down".to_string());
                    self.set_enter_time();
                    communicator.broadcast_with_time(&vec![], MessageTag::EnterRequest as i32);
                    (0..self.config.world_size).for_each(|index| {
                        if index == self.rank as usize {
                            rooms[index] = self.need;
                            lifts[index] = 1;
                        } else {
                            rooms[index] = rooms.get(index).unwrap_or(&0) + self.config.rooms_count;
                            lifts[index] = lifts.get(index).unwrap_or(&0) + 1;
                        }
                    });
                });
            }
            AgentState::Try => {
                logger.log("Trying to go down".to_string());
                if self.taken_rooms_count() <= self.config.rooms_count
                    && self.taken_lifts_count() <= self.config.lifts_count
                {
                    self.next_state(|| logger.log("Going down".to_string()));
                }
            }
            AgentState::Down => {
                self.next_state(|| {
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
                });
            }
            AgentState::Crit => {
                self.next_state(|| {
                    let mut lifts = self.lifts.write().unwrap();
                    *lifts = lifts.iter().map(|v| v + 1).collect();
                    lifts[self.rank as usize] = 1;
                    self.set_leave_time();
                    logger.log("Sending requests for resources to go up".to_string());
                    communicator.broadcast_with_time(&vec![], MessageTag::LeaveRequest as i32);
                });
            }
            AgentState::Leaving => {
                logger.log("Trying to leave the critical section".to_string());
                if self.taken_lifts_count() <= self.config.lifts_count {
                    self.next_state(|| logger.log("Leaving the critical section".to_string()));
                }
            }
            AgentState::Up => {
                self.next_state(|| {
                    let mut messages = vec![(0, 0); self.config.world_size];
                    self.defer_rooms
                        .write()
                        .unwrap()
                        .drain(..)
                        .for_each(|rank| {
                            let (rooms, _lifts) = messages[rank as usize];
                            messages[rank as usize].0 = rooms + self.need as u16;
                        });
                    self.defer_lifts
                        .write()
                        .unwrap()
                        .drain(..)
                        .for_each(|rank| {
                            let (_rooms, lifts) = messages[rank as usize];
                            messages[rank as usize].1 = lifts + 1;
                        });
                    messages
                        .iter()
                        .enumerate()
                        .filter(|(_rank, message)| message.0 != 0 || message.1 != 0)
                        .inspect(|(rank, message)| {
                            logger.log(format!(
                                "Granting {} deferred rooms and {} lifts to process #{}",
                                message.0, message.1, rank
                            ))
                        })
                        .for_each(|(rank, message)| {
                            communicator.send_with_time(
                                &vec![message.0, message.1],
                                rank as i32,
                                MessageTag::Resources as i32,
                            );
                        });
                    logger.log("Going to rest".to_string());
                });
            }
        }
    }

    pub fn receive_request(&self) -> (std::vec::Vec<u16>, mpi::point_to_point::Status) {
        comm::TimestampedCommunicator::new(&self.clock, &self.world).receive()
    }

    pub fn handle_enter_request(&self, sender_rank: i32, request_time: u16) {
        let current_state = self.state.read().unwrap();
        let communicator = comm::TimestampedCommunicator::new(&self.clock, &self.world);
        let logger = util::Logger::new(&self.clock, self.rank);
        let message: Vec<u16>;

        if matches!(*current_state, AgentState::Rest)
            || self.sender_priority(self.enter_time(), request_time, sender_rank)
        {
            message = vec![self.config.rooms_count as u16, 1];
        } else if matches!(*current_state, AgentState::Crit) {
            message = vec![(self.config.rooms_count - self.need) as u16, 1];
            self.defer_rooms.write().unwrap().push(sender_rank as u8);
        } else {
            message = vec![(self.config.rooms_count - self.need) as u16, 0];
            self.defer_rooms.write().unwrap().push(sender_rank as u8);
            self.defer_lifts.write().unwrap().push(sender_rank as u8);
        }
        logger.log(format!(
            "Received a resource request, granting {} rooms and {} lift to process #{}",
            message[0],
            if message[1] == 1 { "a" } else { "no" },
            sender_rank,
        ));
        communicator.send_with_time(&message, sender_rank, MessageTag::Resources as i32);
    }

    pub fn handle_leave_request(&self, sender_rank: i32, request_time: u16) {
        let current_state = self.state.read().unwrap();
        if matches!(*current_state, AgentState::Leaving | AgentState::Up)
            && !self.sender_priority(self.leave_time(), request_time, sender_rank)
        {
            util::Logger::new(&self.clock, self.rank).log(format!(
                "Received a resource request, not granting a lift to process #{}",
                sender_rank,
            ));
            self.defer_lifts.write().unwrap().push(sender_rank as u8);
        } else {
            util::Logger::new(&self.clock, self.rank).log(format!(
                "Received a resource request, granting a lift to process #{}",
                sender_rank,
            ));
            self.set_lifts_at_rank(sender_rank, self.lifts_at_rank(sender_rank) + 1);
            let message = vec![0, 1];
            let communicator = comm::TimestampedCommunicator::new(&self.clock, &self.world);
            communicator.send_with_time(&message, sender_rank, MessageTag::LeaveResources as i32);
        }
    }

    pub fn handle_resources(&self, sender_rank: i32, msg_rooms: u16, msg_lifts: u16) {
        let mut rooms = self.rooms.write().unwrap();
        let mut lifts = self.lifts.write().unwrap();
        rooms[sender_rank as usize] -= msg_rooms as u8;
        lifts[sender_rank as usize] -= msg_lifts as u8;

        util::Logger::new(&self.clock, self.rank).log(format!(
            "Received resource information: {} rooms and {} lifts claimed by process #{}",
            rooms[sender_rank as usize], lifts[sender_rank as usize], sender_rank,
        ));
    }

    pub fn handle_leave_resources(&self, sender_rank: i32) {
        let mut lifts = self.lifts.write().unwrap();
        util::Logger::new(&self.clock, self.rank)
            .log(format!("Received a lift from process #{}", sender_rank));
        lifts[sender_rank as usize] -= 1;
        self.defer_lifts.write().unwrap().push(sender_rank as u8);
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
