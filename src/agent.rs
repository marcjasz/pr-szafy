#![allow(dead_code)]
use mpi::traits::*;
use crate::comm;
use crate::MessageTag;
use crate::CommonState;
use crate::util;
use rand::Rng;
use std::convert::TryInto;
use std::sync::RwLock;


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
        Agent {
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
        }
    }
    
    fn next_state(&self) -> AgentState {
        match self.state {
            AgentState::Rest => AgentState::Try,
            AgentState::Try => AgentState::Down,
            AgentState::Down => AgentState::Crit,
            AgentState::Crit => AgentState::Leaving,
            AgentState::Leaving => AgentState::Up,
            AgentState::Up => AgentState::Rest,
        }
    }

    fn run(&mut self) {
        match self.state {
            AgentState::Rest => {},
            AgentState::Try => {
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
            _ => {}
        }

    }
}

pub fn main_loop(
    clock: &RwLock<comm::Clock>, 
    common_state: &CommonState,
) {
    let rank = common_state.world.rank();
    let logger = util::Logger::new(clock, rank);
    let mut rng = rand::thread_rng();
    let mut agent = Agent::new(rank, rng.gen_range(1, common_state.rooms_count), common_state);
    let msg: Vec<u16> = vec![1, rank.try_into().unwrap()];
    loop {
        let next_state = agent.next_state();
        let secs = rng.gen_range(1, 8);
        match &next_state {
            AgentState::Try => {
                clock.write().unwrap().inc();
                logger.log("Trying to go down".to_string());
                comm::broadcast_with_tag(clock, common_state.world, &vec![], MessageTag::EnterRequest as i32);
            }
            AgentState::Down => {
                clock.write().unwrap().inc();
                logger.log("Going down".to_string());
            }
            AgentState::Crit => {
                clock.write().unwrap().inc();
                logger.log("Entering the critical section".to_string());
            }
            AgentState::Leaving => {
                clock.write().unwrap().inc();
                logger.log("Leaving the critical section".to_string());
            }
            AgentState::Up => {
                logger.log("Going up".to_string());
                comm::broadcast_with_tag(clock, common_state.world, &msg, MessageTag::Resources as i32);
            }
            AgentState::Rest => {
                clock.write().unwrap().inc();
                logger.log("Going to rest".to_string());
            }
        }
        agent.run();
        agent.state = next_state;
        std::thread::sleep(std::time::Duration::from_secs(secs));
    }
}
