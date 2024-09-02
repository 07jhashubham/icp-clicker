use lazy_static::lazy_static;
use ic_sqlite::CONN;
use utils::create_tables_if_not_exist;
use std::sync::Mutex;

mod utils;
pub mod user;


#[ic_cdk::init]
fn init() {
    create_tables_if_not_exist().unwrap();
    ic_cdk::println!("Initialization Complete");
}

