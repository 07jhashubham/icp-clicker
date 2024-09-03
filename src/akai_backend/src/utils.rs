use std::{env, fs::File, io::{self, BufReader, Read, Write}};
use ic_cdk::api::{call::call, management_canister::http_request::{CanisterHttpRequestArgument, HttpHeader, HttpMethod, HttpResponse}};
use anyhow::{anyhow, Ok, Result};
use base64::{engine::general_purpose, Engine};
use candid::{Nat, Principal};
use flate2::{write::GzEncoder, Compression};
use ic_sqlite::CONN;
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


fn dump_and_compress_database() -> Vec<u8> {
    let conn = CONN.lock().unwrap();
    let mut output = Vec::new();

    // Here we dump the database into memory
    let mut stmt = conn.prepare("PRAGMA database_list;").unwrap();
    let mut rows = stmt.query([]).unwrap();
    while let Some(row) = rows.next().unwrap() {
        let file_name: String = row.get(1).unwrap();
        let sql = format!("BACKUP TO '{}'", file_name);
        conn.execute_batch(&sql).unwrap();
        
        let mut file = File::open(file_name).unwrap();
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).unwrap();
        output.extend(buffer);
    }

    // Compress the data using Gzip
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
    let azure_function_url = env::var("AZURE_BACKUP_FUNCTION_URL").unwrap_or("your-function-url".to_string());

    // Dump the database content to a byte vector
    let file_content = dump_and_compress_database();

    // Convert the byte vector to a base64 encoded string
    let base64_content = general_purpose::STANDARD.encode(file_content);

    // Prepare JSON payload for POST request
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
        headers: vec![HttpHeader{
            name: "Content-Type".to_string(),
            value: "application/json".to_string()
        }],
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

    if response.status == Nat::from(200 as usize) {
        ic_cdk::println!("File uploaded successfully.");
    } else {
        ic_cdk::println!(
            "Failed to upload file. Status: {}",
            response.status
        );
    }
}

const CHUNK_SIZE: usize = 1024 * 1024; // 1 MB chunk size

pub async fn backup_chunked() {
    let account = env::var("STORAGE_ACCOUNT").expect("missing STORAGE_ACCOUNT");
    let access_key = env::var("STORAGE_ACCESS_KEY").expect("missing STORAGE_ACCESS_KEY");
    let container = env::var("STORAGE_CONTAINER").expect("missing STORAGE_CONTAINER");
    let blob_name = env::var("STORAGE_BLOB_NAME").expect("missing STORAGE_BLOB_NAME");
    let azure_function_url = env::var("AZURE_BACKUP_FUNCTION_URL").unwrap_or("your-function-url".to_string());

    let mut reader = stream_and_compress_database().unwrap();

    let mut buffer = vec![0; CHUNK_SIZE];
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

        let (response,): (HttpResponse,) = call(
            Principal::management_canister(),
            "http_request",
            (request,),
        )
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

fn stream_and_compress_database() -> io::Result<impl Read> {
    let conn = CONN.lock().unwrap();

    // Prepare to dump and compress the database
    let mut output = Vec::new();
    let mut stmt = conn.prepare("PRAGMA database_list;").unwrap();
    let mut rows = stmt.query([]).unwrap();

    while let Some(row) = rows.next().unwrap() {
        let file_name: String = row.get(1).unwrap();
        let sql = format!("BACKUP TO '{}'", file_name);
        conn.execute_batch(&sql).unwrap();

        let mut file = File::open(file_name)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;

        output.extend(buffer);
    }

    // Compress the data with GzEncoder
    let pipe_reader = io::Cursor::new(output);
    let encoder = GzEncoder::new(pipe_reader, Compression::best());

    Result::Ok(encoder)
}