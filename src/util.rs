use crate::comm;
use ansi_term::Colour::Fixed;
use rand::Rng;
use std::io::{self, Write};

#[derive(Clone)]
pub struct Logger<'clock_lifetime> {
    clock: &'clock_lifetime comm::Clock,
    rank: i32,
}

impl<'clock_lifetime> Logger<'clock_lifetime> {
    pub fn new(clock: &'clock_lifetime comm::Clock, rank: i32) -> Self {
        Self { clock, rank }
    }

    pub fn log(&self, msg: String) -> () {
        let color = Fixed(self.rank as u8 + 9);
        let time = { self.clock.time() };
        let signature = format!("[{}:{}]", self.rank, time);
        println!("{} {}", color.bold().paint(signature), color.paint(msg));
        io::stdout().flush().unwrap();
    }
}

pub fn sleep_random() {
    let secs = rand::thread_rng().gen_range(3, 8);
    std::thread::sleep(std::time::Duration::from_secs(secs));
}
