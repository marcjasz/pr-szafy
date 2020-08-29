#![allow(dead_code)]
use mpi::traits::*;
use crate::comm;
use crate::MessageTag;
use crate::util;
use rand::Rng;
use std::convert::TryInto;
use std::sync::RwLock;

struct Agent<S> {
    need: u8,
    rank: i32,
    state: S
}

enum AgentWrapper {
    Rest(Agent<Rest>),
    Try(Agent<Try>),
    Down(Agent<Down>),
    Crit(Agent<Crit>),
    Leaving(Agent<Leaving>),
    Up(Agent<Up>)
}

impl AgentWrapper {
    fn next(self) -> Self {
        match self {
            AgentWrapper::Rest(agent) => AgentWrapper::Try(agent.into()),
            AgentWrapper::Try(agent) => AgentWrapper::Down(agent.into()),
            AgentWrapper::Down(agent) => AgentWrapper::Crit(agent.into()),
            AgentWrapper::Crit(agent) => AgentWrapper::Leaving(agent.into()),
            AgentWrapper::Leaving(agent) => AgentWrapper::Up(agent.into()),
            AgentWrapper::Up(agent) => AgentWrapper::Rest(agent.into()),
        }
    }
}

impl Agent<Rest> {
    fn new(rank: i32, need: u8) -> Self {
        Agent {
            need: need,
            rank: rank,
            state: Rest { }
        }
    }
}

struct Rest { }

impl From<Agent<Rest>> for Agent<Try> {
    fn from(prev: Agent<Rest>) -> Agent<Try> {
        Agent {
            need: prev.need,
            rank: prev.rank,
            state: Try { }
        }
    }
}

struct Try { }

impl From<Agent<Try>> for Agent<Down> {
    fn from(prev: Agent<Try>) -> Agent<Down> {
        Agent {
            need: prev.need,
            rank: prev.rank,
            state: Down { }
        }
    }
}

struct Down { }

impl From<Agent<Down>> for Agent<Crit> {
    fn from(prev: Agent<Down>) -> Agent<Crit> {
        Agent {
            need: prev.need,
            rank: prev.rank,
            state: Crit { }
        }
    }
}

struct Crit { }

impl From<Agent<Crit>> for Agent<Leaving> {
    fn from(prev: Agent<Crit>) -> Agent<Leaving> {
        Agent {
            need: prev.need,
            rank: prev.rank,
            state: Leaving { }
        }
    }
}

struct Leaving { }

impl From<Agent<Leaving>> for Agent<Up> {
    fn from(prev: Agent<Leaving>) -> Agent<Up> {
        Agent {
            need: prev.need,
            rank: prev.rank,
            state: Up { }
        }
    }
}

struct Up { }

impl From<Agent<Up>> for Agent<Rest> {
    fn from(prev: Agent<Up>) -> Agent<Rest> {
        Agent {
            need: prev.need,
            rank: prev.rank,
            state: Rest { }
        }
    }
}

pub fn main_loop(clock: &RwLock<comm::Clock>, &world: &mpi::topology::SystemCommunicator) {
    let rank = world.rank();
    let logger = util::Logger::new(clock, rank);
    let mut rng = rand::thread_rng();
    let mut agent = AgentWrapper::Rest(Agent::new(rank, 8));
    let msg: Vec<u16> = vec![1, rank.try_into().unwrap()];
    loop {
        let next_state = agent.next();
        let secs = rng.gen_range(1, 8);
        match &next_state {
            AgentWrapper::Try(_state) => {
                clock.write().unwrap().inc();
                logger.log("Trying to go down".to_string());
            }
            AgentWrapper::Down(_state) => {
                clock.write().unwrap().inc();
                logger.log("Going down".to_string());
            }
            AgentWrapper::Crit(_state) => {
                clock.write().unwrap().inc();
                logger.log("Entering the critical section".to_string());
            }
            AgentWrapper::Leaving(_state) => {
                clock.write().unwrap().inc();
                logger.log("Leaving the critical section".to_string());
            }
            AgentWrapper::Up(_state) => {
                logger.log("Going up".to_string());
                comm::broadcast_with_tag(clock, &world, &msg, MessageTag::Resources as i32);
            }
            AgentWrapper::Rest(_state) => {
                clock.write().unwrap().inc();
                logger.log("Going to rest".to_string());
            }
        }
        agent = next_state;
        std::thread::sleep(std::time::Duration::from_secs(secs));
    }
}
