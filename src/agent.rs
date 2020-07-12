#![allow(dead_code)]
use mpi::traits::*;
use crate::util;
use rand::Rng;

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
    for _i in 1..8 {
        agent = agent.next();
        let secs = rng.gen_range(1, 5);
        std::thread::sleep(std::time::Duration::from_secs(secs));
    }
}
