use anyhow::Result;
use base64::{engine::general_purpose, Engine};
use candid::{Nat, Principal};
use flate2::{write::GzEncoder, Compression};

use ic_cdk::{
    api::management_canister::http_request::{
        CanisterHttpRequestArgument, HttpMethod, HttpResponse,
    },
    call,
};
use ic_sqlite_features::CONN;
use lazy_static::lazy_static;
use std::{
    env,
    io::{self, Read, Write},
};

use crate::utils::read_page_from_vfs;

lazy_static! {
    static ref CHUNK_SIZE: usize = env::var("BACKUP_CHUNK_SIZE").unwrap_or((1024 * 1024).to_string()).parse().unwrap_or(1024 * 1024); // 1 MB chunk size
}
#[allow(dead_code)]
pub async fn backup_chunked() {
    let account = env::var("STORAGE_ACCOUNT").expect("missing STORAGE_ACCOUNT");
    let access_key = env::var("STORAGE_ACCESS_KEY").expect("missing STORAGE_ACCESS_KEY");
    let container = env::var("STORAGE_CONTAINER").expect("missing STORAGE_CONTAINER");
    let blob_name = env::var("STORAGE_BLOB_NAME").expect("missing STORAGE_BLOB_NAME");
    let azure_function_url =
        env::var("AZURE_BACKUP_FUNCTION_URL").unwrap_or("your-function-url".to_string());

    let mut reader = stream_and_compress_database().unwrap();
    let mut buffer = vec![0; *CHUNK_SIZE];
    let mut part_number = 0;

    while let Result::Ok(bytes_read) = reader.read(&mut buffer) {
        if bytes_read == 0 {
            break;
        }

        part_number += 1;
        let chunk = &buffer[..bytes_read];
        let base64_content = general_purpose::STANDARD.encode(chunk);

        let function_url = format!(
            "https://{}/api/UploadBlob?account={}&accessKey={}&container={}&blobName={}&partNumber={}",
            azure_function_url, account, access_key, container, blob_name, part_number
        );

        let request = CanisterHttpRequestArgument {
            url: function_url,
            method: HttpMethod::POST,
            body: Some(base64_content.into_bytes()),
            headers: vec![],
            max_response_bytes: None,
            transform: None,
        };

        let (response,): (HttpResponse,) =
            call(Principal::management_canister(), "http_request", (request,))
                .await
                .unwrap();

        if response.status != Nat::from(200 as usize) {
            ic_cdk::println!(
                "Failed to upload chunk {}. Status: {}",
                part_number,
                response.status
            );
            return;
        }
    }

    ic_cdk::println!("File uploaded successfully.");
}

fn stream_and_compress_database() -> Result<impl Read, io::Error> {
    let mut conn = CONN.lock().unwrap();

    // Begin a transaction to ensure consistency
    let tx = conn.transaction().unwrap();

    // Create a pipe (Vec<u8>) to store compressed data
    let mut pipe = Vec::new();
    let mut encoder = GzEncoder::new(&mut pipe, Compression::best());

    let page_count: i64 = tx
        .query_row("PRAGMA page_count;", [], |row| row.get(0))
        .unwrap();

    for page_number in 1..=page_count {
        let page_data = read_page_from_vfs(page_number)?;
        encoder.write_all(&page_data)?;
    }

    // Commit the transaction
    tx.commit().unwrap();

    // Finish the compression process
    encoder.finish()?;

    // Return the compressed data as a cursor for streaming
    Ok(io::Cursor::new(pipe))
}
