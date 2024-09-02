use ic_cdk::{query, update};
use ic_sqlite::CONN;
use lazy_static::lazy_static;
use utils::create_tables_if_not_exist;

pub mod user;
mod utils;

#[ic_cdk::init]
fn init() {
    create_tables_if_not_exist().unwrap();
    ic_cdk::println!("Initialization Complete");
}

ic_cdk::export_candid!();
