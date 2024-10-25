use std::io;

use anyhow::{anyhow, Ok, Result};
use ic_cdk::api::stable::{stable_read, stable_size};
use ic_sqlite_features::CONN;
use sha2::{Digest, Sha256};

use crate::MAX_NUMBER_OF_LABELLINGS_PER_TASK;

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
            format!(
                "
                BEGIN;
                PRAGMA foreign_keys = ON;

                -- Create Aliens table
                CREATE TABLE Aliens (
                    id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                    lvl INT NOT NULL,
                    owner VARCHAR NOT NULL,
                    FOREIGN KEY (owner) REFERENCES User(wallet_address)
                );
                
                -- Create Powerups table
                CREATE TABLE Powerups (
                    id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                    type TEXT NOT NULL CHECK(type IN ('Spawner', 'ClickMultiplier', 'AutoFiller', 'AlienMultiplier')),
                    owner VARCHAR NOT NULL,
                    FOREIGN KEY (owner) REFERENCES User(wallet_address)
                );


                -- Create Task table
                CREATE TABLE Task (
                    id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                    completed_times INT DEFAULT 0 CHECK (completed_times <= {}),
                    type TEXT CHECK(type IN ('AI', 'Social')),
                    desc VARCHAR,
                    data VARCHAR,
                    classes VARCHAR DEFAULT NULL,
                    occupancy INT DEFAULT 0 CHECK (occupancy <= {})
                );

                -- Create Task_logs table
                CREATE TABLE Task_logs (
                    id UUID PRIMARY KEY NOT NULL,
                    task_id INTEGER NOT NULL,
                    completed_by TEXT,
                    image_link VARCHAR,
                    FOREIGN KEY (task_id) REFERENCES Task(id),
                    FOREIGN KEY (completed_by) REFERENCES User(wallet_address)
                );

                -- Create User table
                CREATE TABLE User (
                    name TEXT,
                    wallet_address TEXT PRIMARY KEY UNIQUE NOT NULL,
                    clicks INT DEFAULT 0 NOT NULL,
                    email VARCHAR(50) DEFAULT NULL,
                    twitter VARCHAR(15) DEFAULT NULL,
                    instagram VARCHAR(30) DEFAULT NULL,
                    exp INT DEFAULT 0,
                    rating INT DEFAULT 0
                );

                COMMIT;
            ",
                *MAX_NUMBER_OF_LABELLINGS_PER_TASK, *MAX_NUMBER_OF_LABELLINGS_PER_TASK
            )
            .as_str(),
        )
        .map_err(|e| anyhow!("Failed to create tables: {}", e))?;
    }

    Ok(())
}

pub fn read_page_from_vfs(page_number: i64) -> Result<Vec<u8>, io::Error> {
    let page_size = 4096; // Assuming a page size of 4096 bytes as set in your PRAGMA statement
    let offset = (page_number - 1) * page_size as i64;

    let mut buffer = vec![0u8; page_size];

    read(&mut buffer, offset as u64)?;

    Result::Ok(buffer)
}
fn read(buf: &mut [u8], offset: u64) -> Result<(), io::Error> {
    if stable_size() > 0 {
        stable_read(offset + 8, buf);
    }
    Result::Ok(())
}

pub fn generate_hash_id(data: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    let result = hasher.finalize();
    format!("{:x}", result)
}
