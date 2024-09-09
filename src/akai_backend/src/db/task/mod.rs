use candid::CandidType;
use serde::{Deserialize, Serialize};

pub mod ops;
pub mod debug;

#[derive(Serialize, Deserialize, Debug)]
pub struct Task {
    pub id: usize,
    pub completed_times: usize, // completed times basically keeps track of the number of times people have completed the task so that once it reaches the value of the set MAX_NUMBER_OF_LABELLINGS_PER_TASK we can stop giving this task to the users
    pub r#type: TaskType,
    pub desc: String,
    pub data: String,
    pub classes: Option<String>,
    pub occupancy: usize, // occupancy keeps the track of the number of people who have this task assigned to them and the max value of this always equals to MAX_NUMBER_OF_LABELLINGS_PER_TASK
}

#[derive(Serialize, Deserialize, Debug, CandidType)]
pub enum TaskType {
    AI,
    Social,
}

impl ToString for TaskType{
    fn to_string(&self) -> String {
        match &self{
            TaskType::AI => "AI".to_owned(),
            TaskType::Social => "Social".to_owned()
        }
    }
}