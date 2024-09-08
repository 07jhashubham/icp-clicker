use std::{env, time::Duration};

use backup::sync::backup;
use db::{task::settle_tasks, utils::create_tables_if_not_exist};
use dotenv::dotenv;
use ic_cdk::spawn;
use ic_cdk_timers::set_timer_interval;
use lazy_static::lazy_static;
mod db;
mod backup;
mod scale_ops;
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
    pub static ref MAX_NUMBER_OF_LABELLINGS_PER_TASK: u8 = {
        env::var("MAX_NUMBER_OF_LABELLINGS_PER_TASK")
            .unwrap_or_else(|_| "0".to_string())
            .parse()
            .unwrap_or(0)
    };
}


#[ic_cdk::init]
fn init() {
    dotenv().ok();
    create_tables_if_not_exist().unwrap();

    if *COMMIT_BACKUPS && *BACKUP_DURATION > 0 {
        set_timer_interval(Duration::from_secs(*BACKUP_DURATION), || spawn(backup()));
    }

    // run polled settlement every 10 secs
    set_timer_interval(Duration::from_secs(10), || {
        let future = settle_tasks();
        futures::executor::block_on(future).unwrap();
    });

    // run polled auto_scaling every 20 secs WORK IN PROGRESS
    // set_timer_interval(Duration::from_secs(20), || {
    //     let future = poll_scale();
    //     futures::executor::block_on(future);
    // });

    ic_cdk::println!("Initialization Complete!");
}

ic_cdk::export_candid!();
