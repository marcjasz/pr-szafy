use ansi_term::Colour::Fixed;
use std::sync::RwLock;
use crate::comm;

pub struct Logger<'clock_lifetime> {
    clock: &'clock_lifetime RwLock<comm::Clock>,
    rank: i32
}

impl Logger<'_> {
    pub fn new(clock: &RwLock<comm::Clock>, rank: i32) -> Logger {
        Logger {
            clock: clock,
            rank: rank
        }
    }

    pub fn log(&self, msg: String) -> () {
        let color = Fixed(self.rank as u8 + 9);
        let time = { self.clock.read().unwrap().time };
        let signature = format!("[{}:{}]", time, self.rank);
        println!("{} {}", color.bold().paint(signature), color.paint(msg));
    }
}
