use std::{env, io::Read, sync::Mutex};

use crate::{
    aliens::Aliens,
    badges::Badges,
    task::{Task, TaskType},
    user::User,
};
use anyhow::{anyhow, Ok, Result};
use candid::Principal;
use ic_sqlite::CONN;
use lazy_static::lazy_static;
use serde_json::Value;
pub fn create_tables_if_not_exist() -> Result<()> {
    let tables = ["User", "Aliens", "Task", "Badges"];
    let conn = CONN.lock().map_err(|e| anyhow!("{}", e))?;
    let table_exists = tables.iter().all(|table| {
        conn.query_row(
            &format!(
                "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='{}')",
                table
            ),
            [],
            |row| row.get::<_, bool>(0),
        )
        .unwrap_or(false)
    });

    if !table_exists {
        conn.execute_batch(
            "
            BEGIN;
            PRAGMA foreign_keys = ON;

-- Create Badges table
CREATE TABLE Badges (
    id UUID PRIMARY KEY UNIQUE NOT NULL,
    src VARCHAR,
    lvl INT
);

-- Create Aliens table
CREATE TABLE Aliens (
    id UUID PRIMARY KEY NOT NULL,
    lvl INT NOT NULL,
    data VARCHAR NOT NULL UNIQUE -- should be a md5 or sha256 of the image
);

-- Create Task table
CREATE TABLE Task (
    id UUID PRIMARY KEY NOT NULL,
    completed BOOLEAN DEFAULT FALSE,
    type TEXT CHECK(type IN ('AI', 'Social')),
    desc VARCHAR,
    data VARCHAR,
    classes VARCHAR DEFAULT NULL
);

-- Create User table
CREATE TABLE User (
    name VARCHAR,
    wallet_address VARCHAR PRIMARY KEY UNIQUE NOT NULL,
    badges UUID REFERENCES Badges(id),
    clicks INT DEFAULT 0 NOT NULL,
    email VARCHAR,
    twitter VARCHAR DEFAULT NULL,
    instagram VARCHAR,
    exp INT DEFAULT 0,
    friends UUID REFERENCES User(wallet_address),
    tasks UUID REFERENCES Task(id),
    aliens INT REFERENCES Aliens(id)
);

            COMMIT;
            ",
        )
        .map_err(|e| anyhow!("Failed to create tables: {}", e))?;
    }

    Ok(())
}

pub fn resolve_option<T>(opt: Option<T>) -> String
where
    T: ToString,
{
    match opt {
        Some(o) => o.to_string(),
        None => "NULL".to_string(),
    }
}

use ic_cdk::api::{call::call, management_canister::http_request::{CanisterHttpRequestArgument, HttpMethod, HttpResponse}};

pub async fn backup() {
    let account = std::env::var("STORAGE_ACCOUNT").expect("missing STORAGE_ACCOUNT");
    let access_key = std::env::var("STORAGE_ACCESS_KEY").expect("missing STORAGE_ACCESS_KEY");
    let container = std::env::var("STORAGE_CONTAINER").expect("missing STORAGE_CONTAINER");
    let blob_name = std::env::var("STORAGE_BLOB_NAME").expect("missing STORAGE_BLOB_NAME");
    let file_content = "your file content here";  // You need to serialize the content properly.

    let function_url = format!(
        "https://your-function-url/api/UploadBlob?account={}&accessKey={}&container={}&blobName={}&fileContent={}",
        account, access_key, container, blob_name, file_content
    );

    let request = CanisterHttpRequestArgument {
        url: function_url,
        method: HttpMethod::GET,
        body: None,
        headers: vec![],
        max_response_bytes: None,
        transform: None,
    };

    let (response,): (HttpResponse,) = call(
        Principal::management_canister(),
        "http_request",
        (request,),
    )
    .await
    .unwrap();

    if response.status == candid::Nat::from(200 as usize) {
        ic_cdk::println!("File uploaded successfully.");
    } else {
        ic_cdk::println!(
            "Failed to upload file. Status: {}",
            response.status
        );
    }
}