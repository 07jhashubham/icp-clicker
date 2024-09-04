use std::{cmp::min, env};

use ic_cdk::update;
use ic_sqlite::CONN;
use serde::{Deserialize, Serialize};

use crate::MAX_NUMBER_OF_LABELLINGS_PER_TASK;

#[derive(Serialize, Deserialize, Debug)]
pub struct Task {
    pub id: String,
    pub completed_times: usize, // completed times basically keeps track of the number of times people have completed the task so that once it reaches the value of the set MAX_NUMBER_OF_LABELLINGS_PER_TASK we can stop giving this task to the users
    pub r#type: TaskType,
    pub desc: String,
    pub data: String,
    pub classes: String,
    pub occupancy: usize, // occupancy keeps the track of the number of people who have this task assigned to them and the max value of this always equals to MAX_NUMBER_OF_LABELLINGS_PER_TASK
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(rename_all = "lowercase")]
pub enum TaskType {
    AI,
    Social,
}

fn get_all_tasks() -> Result<Vec<Task>, String> {
    let conn = CONN.lock().map_err(|e| format!("{}", e))?;
    let max_labellings: u8 = *MAX_NUMBER_OF_LABELLINGS_PER_TASK;

    let mut stmt = conn
        .prepare(
            "SELECT id, completed_times, type, desc, data, classes, occupancy 
                                 FROM Task 
                                 WHERE completed_times < ?1 OR occupancy < ?1 ",
        )
        .map_err(|e| format!("{}", e))?;

    let task_iter = stmt
        .query_map([max_labellings], |row| {
            Ok(Task {
                id: row.get(0)?,
                completed_times: row.get(1)?,
                r#type: {
                    match row.get::<_, String>(2)?.as_str() {
                        "ai" => TaskType::AI,
                        "social" => TaskType::Social,
                        _ => panic!(),
                    }
                },
                desc: row.get(3)?,
                data: row.get(4)?,
                classes: row.get(5)?,
                occupancy: row.get(6)?,
            })
        })
        .map_err(|e| format!("{}", e))?;

    let tasks: Vec<Task> = task_iter
        .into_iter()
        .filter_map(|t| match t {
            Ok(tt) => Some(tt),
            Err(_) => None,
        })
        .collect();

    Ok(tasks)
}

#[update]
pub fn fetch_tasks() -> Result<String, String>{
    let mut available_tasks = get_all_tasks()?;

    available_tasks.sort_by(|a, b| {
        a.completed_times
            .cmp(&b.completed_times)
            .then(a.occupancy.cmp(&b.occupancy))
    });

    let task_to_be_assigned = &available_tasks[0..min(available_tasks.len(), env::var("TASKS_PER_USER").unwrap_or("5".to_string()).parse::<usize>().unwrap())];

    commit_tasks(task_to_be_assigned)?;

    Ok(serde_json::to_string(task_to_be_assigned).unwrap())
}

fn commit_tasks(tasks: &[Task]) -> Result<(), String> {
    let conn = CONN.lock().map_err(|e| format!("{}", e))?;
    
    for task in tasks {
        conn.execute(
            "UPDATE Task SET occupancy = occupancy + 1 WHERE id = ?1",
            [&task.id],
        ).map_err(|e| format!("{}", e))?;
    }

    Ok(())
}

