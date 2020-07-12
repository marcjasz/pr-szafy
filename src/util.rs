use ansi_term::Colour::Fixed;

pub fn log(rank: i32, msg: String) -> () {
    let color_num = ((rank * 48) + (rank / 256)) % 240 + 15;
    let color = Fixed(color_num as u8);
    println!("{}. {}", rank, color.paint(msg));
}
