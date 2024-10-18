use std::env;

use candid::Principal;
use ic_cdk::{
    api::management_canister::{
        http_request::{CanisterHttpRequestArgument, HttpMethod, HttpResponse},
        main::{
            create_canister, install_code, CanisterIdRecord, CanisterInstallMode,
            CreateCanisterArgument, InstallCodeArgument,
        },
    },
    call, println, spawn,
};
use ic_stable_memory::get_available_size;

// use crate::CHILD_CANISTER;

pub async fn poll_scale() {
    let available_size = get_available_size();

    let threshold: u64 = 20 * 1024 * 1024 * 1024; // create a new canister at 20 gbs stable mem availablity

    if available_size <= threshold {
        println!(
            "Available stable memory: {} bytes, which is below the 20GB threshold.",
            available_size
        );

        scale().await.unwrap();
    } else {
        println!(
            "Sufficient memory available: {} bytes. No scaling needed.",
            available_size
        );
    }
}
pub async fn scale() -> Result<(), String> {
    let (canister,): (CanisterIdRecord,) = create_canister(CreateCanisterArgument::default(), 69)
        .await
        .map_err(|e| format!("{:?}", e))?;

    install_code(InstallCodeArgument {
        mode: CanisterInstallMode::Install,
        canister_id: canister.canister_id,
        wasm_module: download_wasm_code().await?,
        arg: Vec::new(),
    })
    .await
    .map_err(|e| format!("{:?}", e))?;
    register_child_canister(canister.canister_id.to_string()).await?;
    Ok(())
}

async fn download_wasm_code() -> Result<Vec<u8>, String> {
    let request = CanisterHttpRequestArgument {
        url: "https://www.xyz.com/some.wasm".to_string(),
        method: HttpMethod::GET,
        body: None,
        headers: vec![],
        max_response_bytes: None,
        transform: None,
    };

    let (response,): (HttpResponse,) =
        call(Principal::management_canister(), "http_request", (request,))
            .await
            .map_err(|e| format!("{:?}", e))?;

    if response.status == 200usize {
        return Ok(response.body);
    } else {
        Err(format!(
            "Request failed with status code: {}",
            response.status
        ))
    }
}

async fn register_child_canister(canister_id: String) -> Result<(), String> {
    call(
        Principal::from_text(env::var("MAIN_CANISTER").map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?,
        "add_child_canister",
        (canister_id,),
    )
    .await
    .map_err(|e| format!("{:?}", e))?;
    Ok(())
}
