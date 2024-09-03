use std::{env, io::Read};

use anyhow::{anyhow, Ok, Result};
use azure_storage::prelude::*;
use azure_storage_blobs::prelude::*;
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

pub async fn backup() {
    let file_name = "main.db";

    // Retrieve account name and access key from environment variables
    let account = env::var("STORAGE_ACCOUNT").expect("missing STORAGE_ACCOUNT");
    let access_key = env::var("STORAGE_ACCESS_KEY").expect("missing STORAGE_ACCOUNT_KEY");
    let container = env::var("STORAGE_CONTAINER").expect("missing STORAGE_CONTAINER");
    let blob_name = env::var("STORAGE_BLOB_NAME").expect("missing STORAGE_BLOB_NAME");

    let storage_credentials = StorageCredentials::access_key(account.clone(), access_key);
    let blob_client =
        ClientBuilder::new(account, storage_credentials).blob_client(&container, blob_name);

    // Read file contents
    let mut file = std::fs::File::open(file_name).unwrap();
    let mut file_contents = Vec::new();
    file.read_to_end(&mut file_contents).unwrap();

    // Upload file content as a block blob
    blob_client
        .put_block_blob(file_contents)
        .content_type("application/octet-stream")
        .await
        .unwrap();

    ic_cdk::println!("File uploaded successfully.");
}
