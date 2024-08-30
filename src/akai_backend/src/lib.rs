use lazy_static::lazy_static;
use rusqlite::Connection;
use utils::create_tables_if_not_exist;
use std::sync::Mutex;

mod utils;
pub mod user;
lazy_static! {
    pub static ref CONNECTION: Mutex<Connection> = Mutex::new(Connection::open("./db.db3").unwrap());
}

#[ic_cdk::init]
fn init() {
    let conn = CONNECTION.lock().unwrap();
    create_tables_if_not_exist(&conn).unwrap();
    ic_cdk::println!("Initialization Complete");
}

