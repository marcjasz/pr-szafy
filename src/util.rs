use crate::comm;
use ansi_term::Colour::Fixed;

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
        let signature = format!("[{}:{}]", time, self.rank);
        println!("{} {}", color.bold().paint(signature), color.paint(msg));
    }
}
