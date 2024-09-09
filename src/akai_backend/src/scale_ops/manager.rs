// this contains all the code for the manager with controls what canister has what code...

// WORK IN PROGRESS
use std::cell::RefCell;

use ic_cdk::{init, update};
use ic_stable_memory::{collections::SVec, SBox};

use super::scale::scale;

thread_local! {
    pub static CLUSTER_CANISTERS: RefCell<SVec<SBox<String>>> = RefCell::new(SVec::new());
}
// #[init]
fn deploy() {
    let future = scale();
    futures::executor::block_on(future).unwrap();
}

// #[update]
fn add_child_canister(canister_id: String) {
    CLUSTER_CANISTERS
        .with_borrow_mut(|c| c.push(SBox::new(canister_id).unwrap()))
        .unwrap();
}
