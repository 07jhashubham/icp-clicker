use std::{cell::RefCell, env, time::Duration};

use backup::sync::backup;
use dotenv::dotenv;
use ic_cdk::spawn;
use ic_cdk_timers::set_timer_interval;
use ic_stable_structures::{
    memory_manager::{MemoryId, MemoryManager, VirtualMemory},
    DefaultMemoryImpl, StableBTreeMap,
};
use lazy_static::lazy_static;
use task::settle_tasks;
use utils::create_tables_if_not_exist;
mod aliens;
mod backup;
mod badges;
mod task;
mod user;
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
    pub static ref MAX_NUMBER_OF_LABELLINGS_PER_TASK: u8 = {
        env::var("MAX_NUMBER_OF_LABELLINGS_PER_TASK")
            .unwrap_or_else(|_| "0".to_string())
            .parse()
            .unwrap_or(0)
    };
}
type Memory = VirtualMemory<DefaultMemoryImpl>;

thread_local! {
    // The memory manager is used for simulating multiple memories. Given a `MemoryId` it can
    // return a memory that can be used by stable structures.
    pub static MEMORY_MANAGER: RefCell<MemoryManager<DefaultMemoryImpl>> =
        RefCell::new(MemoryManager::init(DefaultMemoryImpl::default()));

    pub static BADGES: RefCell<StableBTreeMap<u32, String, Memory>> = RefCell::new(
        StableBTreeMap::init(
            MEMORY_MANAGER.with(|m| m.borrow().get(MemoryId::new(0))),
        )
    );
}
#[ic_cdk::init]
fn init() {
    dotenv().ok();
    create_tables_if_not_exist().unwrap();

    if *COMMIT_BACKUPS && *BACKUP_DURATION > 0 {
        set_timer_interval(Duration::from_secs(*BACKUP_DURATION), || spawn(backup()));
    }

    set_timer_interval(Duration::from_secs(10), || {
        let future = settle_tasks(); // this creates a future
        futures::executor::block_on(future).unwrap(); // blocks on the future
    });

    
    ic_cdk::println!("Initialization Complete!");
}

ic_cdk::export_candid!();
