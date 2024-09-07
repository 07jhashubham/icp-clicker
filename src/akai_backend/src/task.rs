use std::env;

use ic_cdk::update;
use ic_sqlite_features::{params, CONN};
use serde::{Deserialize, Serialize};

use crate::{utils::generate_hash_id, MAX_NUMBER_OF_LABELLINGS_PER_TASK};

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

#[update]
fn fetch_and_commit_tasks() -> Result<String, String> {
    let mut conn = CONN.lock().map_err(|e| format!("{}", e))?;
    let max_labellings: u8 = *MAX_NUMBER_OF_LABELLINGS_PER_TASK;
    let tasks_per_user: usize = env::var("TASKS_PER_USER")
        .unwrap_or("5".to_string())
        .parse()
        .unwrap();

    let tx = conn
        .transaction()
        .map_err(|e| format!("Transaction start failed: {}", e))?;

    let tasks: Vec<Task> = {
        let mut stmt = tx
            .prepare(
                "WITH SelectedTasks AS (
                SELECT id, completed_times, type, desc, data, classes, occupancy
                FROM Task
                WHERE completed_times < ?1 AND occupancy < ?1
                ORDER BY completed_times ASC, occupancy ASC
                LIMIT ?2
            )
            -- Fetch the selected tasks
            SELECT id, completed_times, type, desc, data, classes, occupancy
            FROM SelectedTasks;
            
            -- Update occupancy for the selected tasks
            UPDATE Task
            SET occupancy = occupancy + 1
            WHERE id IN (SELECT id FROM SelectedTasks);",
            )
            .map_err(|e| format!("{}", e))?;

        // Execute the query to fetch tasks
        let task_iter = stmt
            .query_map(params![max_labellings, tasks_per_user], |row| {
                Ok(Task {
                    id: row.get(0)?,
                    completed_times: row.get(1)?,
                    r#type: match row.get::<_, String>(2)?.as_str() {
                        "ai" => TaskType::AI,
                        "social" => TaskType::Social,
                        _ => panic!(),
                    },
                    desc: row.get(3)?,
                    data: row.get(4)?,
                    classes: row.get(5)?,
                    occupancy: row.get(6)?,
                })
            })
            .map_err(|e| format!("Task fetch failed: {}", e))?;

        task_iter.filter_map(|t| t.ok()).collect()
    };

    tx.commit()
        .map_err(|e| format!("Transaction commit failed: {}", e))?;

    Ok(serde_json::to_string(&tasks).unwrap())
}

pub fn clear_tasks_occupancy(t_ids: &[String]) -> Result<(), String> {
    let mut conn = CONN.lock().map_err(|e| format!("{}", e))?;

    let tx = conn
        .transaction()
        .map_err(|e| format!("Transaction start failed: {}", e))?;

    let placeholders = t_ids.iter().map(|_| "?").collect::<Vec<_>>().join(", ");

    let query = format!(
        "UPDATE Task SET occupancy = occupancy - 1 WHERE id IN ({})",
        placeholders
    );

    let params: Vec<&dyn ic_sqlite_features::ToSql> = t_ids
        .iter()
        .map(|id| id as &dyn ic_sqlite_features::ToSql)
        .collect();

    tx.execute(&query, &*params)
        .map_err(|e| format!("Task update failed: {}", e))?;

    tx.commit()
        .map_err(|e| format!("Transaction commit failed: {}", e))?;

    Ok(())
}

#[update]
pub fn complete_tasks(
    t_id: String,
    wallet_address: String,
    date_time: String,
) -> Result<(), String> {
    let mut conn = CONN.lock().map_err(|e| format!("{}", e))?;

    let tx = conn
        .transaction()
        .map_err(|e| format!("Transaction start failed: {}", e))?;

    tx.execute("UPDATE Task SET occupancy = occupancy - 1, completed_times = completed_times + 1 WHERE id = ?1", [&t_id])
        .map_err(|e| format!("Task update failed: {}", e))?;

    let logger_id = generate_hash_id(&(t_id + &wallet_address));
    tx.execute(
        "UPDATE Task_logs
         SET datetime = ?1, completed_by = ?2
         WHERE id = ?3",
        params![date_time, wallet_address, logger_id],
    )
    .map_err(|e| format!("{}", e))?;
    tx.commit()
        .map_err(|e| format!("Transaction commit failed: {}", e))?;

    Ok(())
}
