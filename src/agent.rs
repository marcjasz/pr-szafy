#![allow(dead_code)]
use mpi::traits::*;
use crate::util;
use crate::MessageTag;
use rand::Rng;
use std::convert::TryInto;

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
    fn new(rank: i32,need: u8) -> Self {
        util::log(rank, "Initialized".to_string());
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
        util::log(prev.rank, "Trying to go down".to_string());
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
        util::log(prev.rank, "Going down".to_string());
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
        util::log(prev.rank, "Entering the critical section".to_string());
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
        util::log(prev.rank, "Leaving the critical section".to_string());
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
        util::log(prev.rank, "Going up".to_string());
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
        util::log(prev.rank, "Going to rest".to_string());
        Agent {
            need: prev.need,
            rank: prev.rank,
            state: Rest { }
        }
    }
}

pub fn main_loop(world: mpi::topology::SystemCommunicator) {
    let rank = world.rank();
    let mut rng = rand::thread_rng();
    let mut agent = AgentWrapper::Rest(Agent::new(rank, 8));
    let msg: Vec<u8> = vec![1, rank.try_into().unwrap()];
    loop {
        let next_state = agent.next();
        let secs = rng.gen_range(1, 5);
        match &next_state {
            AgentWrapper::Up(_state) => {
                broadcast_with_tag(world, &msg, MessageTag::Resources as i32);
            }
            _ => {
                ();
            }
        }
        agent = next_state;
        std::thread::sleep(std::time::Duration::from_secs(secs));
    }
}


fn broadcast_with_tag(world: mpi::topology::SystemCommunicator, message: &Vec<u8>, tag: i32) {
    for i in 1..world.size() {
        world.process_at_rank(i).send_with_tag(&message[..], tag);
    }
}
