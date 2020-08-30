#![allow(dead_code)]
use mpi::traits::*;
use crate::comm;
use crate::MessageTag;
use crate::util;
use rand::Rng;
use std::convert::TryInto;
use std::sync::RwLock;

struct Agent {
    need: u8,
    rank: i32,
    state: AgentState
}

enum AgentState {
    Rest,
    Try,
    Down,
    Crit,
    Leaving,
    Up
}

fn next_state(state: AgentState) -> AgentState {
    match state {
        AgentState::Rest => AgentState::Try,
        AgentState::Try => AgentState::Down,
        AgentState::Down => AgentState::Crit,
        AgentState::Crit => AgentState::Leaving,
        AgentState::Leaving => AgentState::Up,
        AgentState::Up => AgentState::Rest,
    }
}

impl Agent {
    fn new(rank: i32, need: u8) -> Self {
        Agent {
            need: need,
            rank: rank,
            state: AgentState::Rest 
        }
    }
}

pub fn main_loop(clock: &RwLock<comm::Clock>, &world: &mpi::topology::SystemCommunicator) {
    let rank = world.rank();
    let logger = util::Logger::new(clock, rank);
    let mut rng = rand::thread_rng();
    let mut agent = Agent::new(rank, 8);
    let msg: Vec<u16> = vec![1, rank.try_into().unwrap()];
    loop {
        let next_state = next_state(agent.state);
        let secs = rng.gen_range(1, 8);
        match &next_state {
            AgentState::Try => {
                clock.write().unwrap().inc();
                logger.log("Trying to go down".to_string());
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
                comm::broadcast_with_tag(clock, &world, &msg, MessageTag::Resources as i32);
            }
            AgentState::Rest => {
                clock.write().unwrap().inc();
                logger.log("Going to rest".to_string());
            }
        }
        agent.state = next_state;
        std::thread::sleep(std::time::Duration::from_secs(secs));
    }
}
