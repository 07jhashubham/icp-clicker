use std::{cell::RefCell, env, time::Duration};
use crate::db::powerups::PowerupType;

// use crate::db::task::TaskType;
// use backup::sync::backup;
use db::{user::ops::create_new_user, utils::create_tables_if_not_exist};
use ic_cdk::{api::management_canister::main::{install_code, CanisterInstallMode, InstallCodeArgument}, spawn};
use ic_cdk_timers::set_timer_interval;
use lazy_static::lazy_static;
mod backup;
mod db;
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
            .unwrap_or_else(|_| "3".to_string())
            .parse()
            .unwrap_or(3)
    };
}
thread_local! {
    static CHUNKS: RefCell<Vec<Vec<u8>>> = RefCell::new(Vec::new());
    static CHUNKS_TOTAL_SIZE: RefCell<usize> = RefCell::new(0);
}

#[ic_cdk::update]
fn receive_wasm(data: Vec<u8>, last: bool) {
    CHUNKS.with(|chunks| {
        chunks.borrow_mut().push(data.clone());
    });
    CHUNKS_TOTAL_SIZE.with(|total_size| {
        *total_size.borrow_mut() += data.len();
    });

    if last {
        CHUNKS.with(|chunks| {
            let chunks = chunks.borrow();
            let total_size: usize = chunks.iter().map(|chunk| chunk.len()).sum();
            let mut complete_wasm = Vec::with_capacity(total_size);
            for chunk in chunks.iter() {
                complete_wasm.extend_from_slice(chunk);
            }

            // Avoid blocking for too long by splitting the update in smaller tasks
            ic_cdk::spawn(async move {
                update_code(complete_wasm).await;
            });
        });
    }
}

async fn update_code(code: Vec<u8>) {
    let canister_id = ic_cdk::api::id();
    let install_arg = InstallCodeArgument {
        canister_id,
        mode: CanisterInstallMode::Reinstall,
        wasm_module: code,
        arg: Vec::new(),
    };

    match install_code(install_arg).await {
        Ok(_) => ic_cdk::println!("Code installation successful."),
        Err(e) => {
            CHUNKS.with(|chunks| {
                chunks.borrow_mut().clear();
            });
            ic_cdk::println!("Code installation failed: {:?}", e)
        },
    }
}


#[ic_cdk::init]
fn init() {
    create_tables_if_not_exist().unwrap();

    // FOR TESTING PURPOSES
    create_new_user("user1234".to_string(), None, None, None, None).unwrap();





    // if *COMMIT_BACKUPS && *BACKUP_DURATION > 0 {
    //     set_timer_interval(Duration::from_secs(*BACKUP_DURATION), || spawn(backup()));
    // }

    // run polled settlement every 10 secs
    // set_timer_interval(Duration::from_secs(10), || {
    //     let future = settle_tasks();
    //     futures::executor::block_on(future).unwrap();
    // });

    // run polled auto_scaling every 20 secs WORK IN PROGRESS
    // set_timer_interval(Duration::from_secs(20), || {
    //     let future = poll_scale();
    //     futures::executor::block_on(future);
    // });

    ic_cdk::println!("Initialization Complete!");
}

ic_cdk::export_candid!();
