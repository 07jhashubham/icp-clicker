use serde::{Deserialize, Serialize};

pub mod debug;
pub mod ops;

#[derive(Debug, Serialize, Deserialize)]
pub struct User {
    pub name: Option<String>,
    pub wallet_address: String,
    pub clicks: i32,
    pub email: Option<String>,
    pub twitter: Option<String>,
    pub instagram: Option<String>,
    pub exp: usize,
    pub rating: usize,
}
