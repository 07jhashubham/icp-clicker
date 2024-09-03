use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct Aliens {
    pub id: String,
    pub lvl: usize,
    pub data: String, // md5
}
