use std::time::Duration;

use futures::executor::block_on;
use lazy_static::lazy_static;
use utils::{backup, create_tables_if_not_exist};

pub mod aliens;
pub mod badges;
pub mod task;
pub mod user;
mod utils;
lazy_static! {
    pub static ref COMMIT_BACKUPS: bool = false;
}
#[ic_cdk::init]
fn init() {
    create_tables_if_not_exist().unwrap();
    if *COMMIT_BACKUPS {
        ic_cdk_timers::set_timer(Duration::from_secs(10 * 60), || block_on(backup()));
    }
    ic_cdk::println!("Initialization Complete!");
}
ic_cdk::export_candid!();
