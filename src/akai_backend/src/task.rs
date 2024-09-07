use std::{collections::HashMap, env};

use ic_cdk::update;
use ic_sqlite_features::{params, CONN};
use serde::{Deserialize, Serialize};

use crate::{
    user::{update_exp, update_rating},
    utils::generate_hash_id,
    MAX_NUMBER_OF_LABELLINGS_PER_TASK,
};

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
            .prepare_cached(
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
pub fn complete_task(
    t_id: String,
    wallet_address: String,
    image_link: String,
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
         SET image_link = ?1, completed_by = ?2
         WHERE id = ?3",
        params![image_link, wallet_address, logger_id],
    )
    .map_err(|e| format!("{}", e))?;

    update_exp(wallet_address, 100, &tx)?; // give 100 exp on completion of a task
    tx.commit()
        .map_err(|e| format!("Transaction commit failed: {}", e))?;

    Ok(())
}

pub async fn settle_tasks() -> Result<(), String> {
    let mut conn = CONN.lock().map_err(|e| e.to_string())?;

    let tx = conn.transaction().map_err(|e| e.to_string())?;

    {
        let mut stmt = tx
            .prepare_cached(
                "SELECT t.id, l.completed_by, l.image_link
                FROM Task t
                JOIN Task_logs l ON t.id = l.task_id
                WHERE t.completed_times = ?1",
            )
            .map_err(|e| e.to_string())?;

        let rows = stmt
            .query_map(params![*MAX_NUMBER_OF_LABELLINGS_PER_TASK], |row| {
                Ok((
                    row.get::<_, String>(0)?, // Task ID
                    row.get::<_, String>(1)?, // Completed By (user)
                    row.get::<_, String>(2)?, // Image Link
                ))
            })
            .map_err(|e| e.to_string())?;

        let mut group_by_id: HashMap<String, Vec<(String, String)>> = HashMap::new(); // task_id -> (image_link, user)

        for r in rows {
            let (task_id, user, image_link): (String, String, String) =
                r.map_err(|e| e.to_string())?;
            group_by_id
                .entry(task_id)
                .or_insert_with(Vec::new)
                .push((image_link, user));
        }

        let mut del_stmt = tx
            .prepare_cached(
                "
            DELETE FROM Task_logs WHERE task_id = ?1;
            DELETE FROM Task WHERE id = ?1;
                ",
            )
            .map_err(|e| e.to_string())?;

        for (id, v) in group_by_id {
            let rating = fetch_images_determine_rating_increment(&v).await?;

            if v.len() == rating.len() {
                return Err("Something went wrong DEVS CHEKCK".to_string());
            };

            for (user, increment) in rating {
                update_rating(user, increment, &tx)?;
            }

            del_stmt.execute(params![id]).map_err(|e| e.to_string())?;
        }
    }

    tx.commit().map_err(|e| e.to_string())?;

    Ok(())
}

// (image_links, wallet_address) -> (wallet_address, increment)
async fn fetch_images_determine_rating_increment(
    image_vec: &Vec<(String, String)>,
) -> Result<Vec<(String, usize)>, String> {
    Ok(image_vec
        .iter()
        .map(|(_, user)| (user.to_owned(), 1))
        .collect())
}
// placeholder
