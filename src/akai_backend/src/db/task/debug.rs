use ic_cdk::update;
use ic_sqlite_features::CONN;
use serde_json::json;

use super::{Task, TaskType};

#[update]
fn get_all_tasks() -> Result<String, String> {
    let conn = CONN.lock().map_err(|e| e.to_string())?;
    let mut stmt = conn
        .prepare("SELECT id, completed_times, type, desc, data, classes, occupancy FROM Task")
        .map_err(|e| e.to_string())?;
    let task_iter = stmt
        .query_map([], |row| {
            Ok(Task {
                id: row.get::<_, usize>(0)?, // explicitly getting id as i64
                completed_times: row.get(1)?,
                r#type: match row.get::<_, String>(2)?.as_str() {
                    "AI" => TaskType::AI,
                    "Social" => TaskType::Social,
                    _ => panic!(),
                },
                desc: row.get(3)?,
                data: row.get(4)?,
                classes: row.get(5)?,
                occupancy: row.get(6)?,
            })
        })
        .map_err(|e| e.to_string())?;

    let mut tasks = Vec::new();
    for task in task_iter {
        tasks.push(task.map_err(|e| e.to_string())?);
    }

    // Serialize tasks vector to JSON
    let json_output = serde_json::to_string(&tasks).map_err(|e| e.to_string())?;
    Ok(json_output)
}

#[update]
fn get_all_tasks_logs() -> Result<String, String> {
    let conn = CONN.lock().map_err(|e| e.to_string())?;
    let mut stmt = conn
        .prepare("SELECT * FROM Task_logs")
        .map_err(|e| e.to_string())?;
    let task_iter = stmt
        .query_map([], |row| {
            Ok(json!({
                "id": row.get::<_, String>(0).unwrap(),
                "task_id": row.get::<_, usize>(1).unwrap(),
                "completed_by": row.get::<_, String>(2).unwrap(),
                "image_link":row.get::<_, String>(3).unwrap(),
            }))
        })
        .map_err(|e| e.to_string())?;

    let mut tasks = Vec::new();
    for task in task_iter {
        tasks.push(task.map_err(|e| e.to_string())?);
    }

    // Serialize tasks vector to JSON
    let json_output = serde_json::to_string(&tasks).map_err(|e| e.to_string())?;
    Ok(json_output)
}
