#![allow(dead_code)]
use mpi::traits::*;
use crate::comm;
use crate::MessageTag;
use crate::CommonState;
use crate::util;
use rand::Rng;
use std::convert::TryInto;
use std::sync::{RwLock};
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Clone)]
struct Agent<'world_lifetime> {
    need: u8,
    rank: i32,
    enter_time: i16,
    leave_time: i16,
    rooms: Vec<u8>,
    lifts: Vec<u8>,
    defer_rooms: Vec<u8>,
    defer_lifts: Vec<u8>,
    state: AgentState,
    common_state: &'world_lifetime CommonState<'world_lifetime>,
    logger: util::Logger<'world_lifetime>,
}
 
#[derive(Clone, Debug)]
enum AgentState {
    Rest,
    Try,
    Down,
    Crit,
    Leaving,
    Up
}

impl<'world_lifetime> Agent<'world_lifetime> {
    fn new(rank: i32, need: u8, common_state: &'world_lifetime CommonState) -> Self {
        Self {
            need: need,
            rank: rank,
            state: AgentState::Rest,
            enter_time: i16::MIN,
            leave_time: i16::MIN,
            rooms: vec![0; common_state.world.size() as usize],
            lifts: vec![0; common_state.world.size() as usize],
            defer_rooms: Vec::new(),
            defer_lifts: Vec::new(),
            common_state: common_state,
            logger: util::Logger::new(common_state.clock, rank),
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

    fn run(&mut self) {
        match self.state {
            AgentState::Try => {
                self.logger.log("Trying to go down".to_string());
                comm::broadcast_with_tag(self.common_state.clock, self.common_state.world, &vec![], MessageTag::EnterRequest as i32);
                self.rooms = self.rooms
                    .iter()
                    .enumerate()
                    .map(|(index, _value)| 
                        if index == self.rank as usize {
                            self.need
                        } else {
                            self.common_state.rooms_count
                        }
                    )
                    .collect();
                self.lifts = vec![1; self.common_state.world.size() as usize];
            },
            AgentState::Down => {
                self.logger.log("Going down".to_string());
            }
            AgentState::Crit => {
                self.logger.log("Entering the critical section".to_string());
            }
            AgentState::Leaving => {
                self.logger.log("Leaving the critical section".to_string());
            }
            AgentState::Up => {
                self.logger.log("Going up".to_string());
                let msg: Vec<u16> = vec![1, self.rank.try_into().unwrap()];
                comm::broadcast_with_tag(self.common_state.clock, self.common_state.world, &msg, MessageTag::Resources as i32);
            }
            AgentState::Rest => {
                self.logger.log("Going to rest".to_string());
            }
        }

    }
}

pub fn main_loop(
    common_state: &CommonState,
    is_alive: &AtomicBool,
) {
    let rank = common_state.world.rank();
    let mut rng = rand::thread_rng();
    let agent = Agent::new(rank, rng.gen_range(1, common_state.rooms_count), common_state);
    let agent_main = RwLock::new(agent);

    while is_alive.load(Ordering::SeqCst) {
        common_state.clock.write().unwrap().inc();
        agent_main.write().unwrap().run();
        agent_main.write().unwrap().next_state();
        let secs = rng.gen_range(1, 8);
        std::thread::sleep(std::time::Duration::from_secs(secs));
    }
}
