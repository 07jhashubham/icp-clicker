use candid::Principal;
use ic_cdk::{api::management_canister::{http_request::{CanisterHttpRequestArgument, HttpMethod, HttpResponse}, main::{create_canister, install_code, CanisterIdRecord, CanisterInstallMode, CreateCanisterArgument, InstallCodeArgument}}, call};
use ic_stable_memory::SBox;

use crate::CHILD_CANISTER;


async fn scale() -> Result<(), String>{
    let (canister, ): (CanisterIdRecord, ) = create_canister(CreateCanisterArgument::default(), 69).await.map_err(|e| format!("{:?}", e))?;

    install_code(InstallCodeArgument{
        mode: CanisterInstallMode::Install,
        canister_id: canister.canister_id,
        wasm_module: download_wasm_code().await?,
        arg: Vec::new()
    }).await.map_err(|e| format!("{:?}", e))?;

    CHILD_CANISTER.with_borrow_mut(|c| c.push(SBox::new(canister.canister_id.to_string()).unwrap())).map_err(|e| e.to_string())?;
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
            .await.map_err(|e| format!("{:?}", e))?;

    if response.status == 200usize {
        return Ok(response.body);
    } else {
        Err(format!("Request failed with status code: {}", response.status))
    }
}
