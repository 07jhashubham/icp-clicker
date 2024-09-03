use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct Badges {
    pub id: String,
    pub src: String,
    pub lvl: usize,
}
