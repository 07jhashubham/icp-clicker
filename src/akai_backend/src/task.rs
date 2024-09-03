use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct Task {
    pub id: String,
    pub completed: bool,
    pub r#type: TaskType,
    pub desc: String,
    pub data: String,
    pub classes: String,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(rename_all = "lowercase")]
pub enum TaskType {
    AI,
    Social,
}
