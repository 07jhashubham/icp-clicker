use std::cell::RefCell;

use ic_cdk::{api::management_canister::main::{install_code, update_settings, CanisterId, CanisterInstallMode, InstallCodeArgument, UpdateSettingsArgument}, update};

#[ic_cdk::query]
fn greet(name: String) -> String {
    format!("Hello, {}!", name)
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

ic_cdk::export_candid!();
