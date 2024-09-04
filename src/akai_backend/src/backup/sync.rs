use base64::{engine::general_purpose, Engine};
use candid::{Nat, Principal};
use flate2::{write::GzEncoder, Compression};
use ic_cdk::{
    api::management_canister::http_request::{
        CanisterHttpRequestArgument, HttpHeader, HttpMethod, HttpResponse,
    },
    call,
};
use ic_sqlite::CONN;
use std::{env, io::Write};

use crate::utils::read_page_from_vfs;

fn dump_and_compress_database() -> Vec<u8> {
    let mut conn = CONN.lock().unwrap();
    let mut output = Vec::new();

    let tx = conn.transaction().unwrap();

    let page_count: i64 = tx
        .query_row("PRAGMA page_count;", [], |row| row.get(0))
        .unwrap();
    for page_number in 1..=page_count {
        let page_data = read_page_from_vfs(page_number).unwrap();
        output.extend_from_slice(&page_data);
    }

    tx.commit().unwrap();

    let mut encoder = GzEncoder::new(Vec::new(), Compression::best());
    encoder.write_all(&output).unwrap();
    let compressed_data = encoder.finish().unwrap();

    compressed_data
}
pub async fn backup() {
    let account = env::var("STORAGE_ACCOUNT").expect("missing STORAGE_ACCOUNT");
    let access_key = env::var("STORAGE_ACCESS_KEY").expect("missing STORAGE_ACCESS_KEY");
    let container = env::var("STORAGE_CONTAINER").expect("missing STORAGE_CONTAINER");
    let blob_name = env::var("STORAGE_BLOB_NAME").expect("missing STORAGE_BLOB_NAME");
    let azure_function_url =
        env::var("AZURE_BACKUP_FUNCTION_URL").unwrap_or("your-function-url".to_string());

    let file_content = dump_and_compress_database();
    let base64_content = general_purpose::STANDARD.encode(file_content);

    let payload = serde_json::json!({
        "account": account,
        "accessKey": access_key,
        "container": container,
        "blobName": blob_name,
        "fileContent": base64_content,
    });

    let request = CanisterHttpRequestArgument {
        url: azure_function_url,
        method: HttpMethod::POST,
        body: Some(payload.to_string().into_bytes()),
        headers: vec![HttpHeader {
            name: "Content-Type".to_string(),
            value: "application/json".to_string(),
        }],
        max_response_bytes: None,
        transform: None,
    };

    let (response,): (HttpResponse,) =
        call(Principal::management_canister(), "http_request", (request,))
            .await
            .unwrap();

    if response.status == Nat::from(200 as usize) {
        ic_cdk::println!("File uploaded successfully.");
    } else {
        ic_cdk::println!("Failed to upload file. Status: {}", response.status);
    }
}
