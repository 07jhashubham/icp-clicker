use std::fs::File;
use std::io::{self, Read, BufReader};
use candid::{CandidType, Encode, Nat, Principal};
use ic_agent::{identity::AnonymousIdentity, Agent};
use serde::Deserialize;
use tokio::runtime::Runtime;

const CHUNK_SIZE: usize = 1 * 1024 * 1024 / 2; // 2 MB

#[tokio::main]
async fn main() {
    foo().await;
}

#[derive(CandidType)]
struct Arguments {
    data: Vec<u8>,
    last: bool,
}

#[derive(CandidType, Deserialize)]
struct CreateCanisterResult {
    ret: String,
}

async fn foo() {
    let agent = Agent::builder()
        .with_url("https://icp0.io")
        .with_identity(AnonymousIdentity)
        .build()
        .unwrap();
    
    agent.fetch_root_key().await.unwrap(); 
    let effective_canister_id = Principal::from_text("2gsgt-vyaaa-aaaab-qacia-cai").unwrap();
    
    let mut file = BufReader::new(File::open("/home/joel/Documents/rando/akai_backend.wasm").unwrap());
    let mut buffer = vec![0; CHUNK_SIZE];
    let mut total_bytes_read = 0;

    // agent.update(&effective_canister_id, "make_self_controller").call_and_wait().await.unwrap();
    loop {
        let bytes_read = file.read(&mut buffer).unwrap();
        if bytes_read == 0 {
            break; // End of file
        }

        total_bytes_read += bytes_read;
        
        // Check if this is the last chunk
        let is_last_chunk = bytes_read < CHUNK_SIZE;

        println!("Read {} bytes, total: {}, last chunk: {}", bytes_read, total_bytes_read, is_last_chunk);

        let chunk_data = buffer[..bytes_read].to_vec();
        
        // Send the chunk to your update call, including whether it's the last chunk
        let response = agent.update(&effective_canister_id, "receive_wasm")
        .with_arg(Encode!(&chunk_data, &is_last_chunk).unwrap())
        .call_and_wait()
        .await;

        // Handle the response as needed
        match response {
            Ok(result) => println!("Successfully sent chunk"),
            Err(e) => eprintln!("Error: {}", e),
        }

        // If this was the last chunk, exit the loop
        if is_last_chunk {
            break;
        }
    }

    println!("Finished reading the file.");
}
