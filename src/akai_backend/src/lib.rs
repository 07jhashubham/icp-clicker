use std::{env, time::Duration};

use futures::executor::block_on;
use ic_cdk_timers::set_timer_interval;
use lazy_static::lazy_static;
use utils::{backup, create_tables_if_not_exist};

pub mod aliens;
pub mod badges;
pub mod task;
pub mod user;
mod utils;
lazy_static! {
    pub static ref COMMIT_BACKUPS: bool = {
        match env::var("COMMIT_BACKUPS").as_deref() {
            Ok("true") => true,
            Ok("false") => false,
            _ => false,
        }
    };

    pub static ref BACKUP_DURATION: u64 = {
        if *COMMIT_BACKUPS {
            env::var("BACKUP_DURATION")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0)
        } else {
            0
        }
    };
}

#[ic_cdk::init]
fn init() {
    create_tables_if_not_exist().unwrap();

    if *COMMIT_BACKUPS && *BACKUP_DURATION > 0 {
        set_timer_interval(Duration::from_secs(*BACKUP_DURATION), || block_on(backup()));
    }

    ic_cdk::println!("Initialization Complete!");
}
ic_cdk::export_candid!();
