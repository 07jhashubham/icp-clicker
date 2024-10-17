// use std::{collections::HashMap, env};

// use ic_cdk::update;
// use ic_sqlite_features::{params, params_from_iter, CONN};

// use crate::{
//     db::{
//         user::ops::{update_exp, update_rating},
//         utils::generate_hash_id,
//     },
//     MAX_NUMBER_OF_LABELLINGS_PER_TASK,
// };

// use super::{Task, TaskType};

// #[update]
// fn add_task(
//     r#type: TaskType,
//     desc: String,
//     data: String,
//     classes: Option<String>,
// ) -> Result<(), String> {
//     let mut conn = CONN.lock().map_err(|e| e.to_string())?;

//     let tx = conn.transaction().map_err(|e| e.to_string())?;

//     tx.execute(
//         "INSERT INTO Task (type, desc, data, classes)
// VALUES (?1, ?2, ?3, ?4);",
//         params![r#type.to_string(), desc, data, classes],
//     )
//     .map_err(|e| e.to_string())?;

//     tx.commit().map_err(|e| e.to_string())?;
//     Ok(())
// }
// #[update]
// fn fetch_and_commit_tasks() -> Result<String, String> {
//     let mut conn = CONN.lock().map_err(|e| format!("{}", e))?;
//     let max_labellings: u8 = *MAX_NUMBER_OF_LABELLINGS_PER_TASK;
//     let tasks_per_user: usize = env::var("TASKS_PER_USER")
//         .unwrap_or("5".to_string())
//         .parse()
//         .unwrap();

//     let tx = conn
//         .transaction()
//         .map_err(|e| format!("Transaction start failed: {}", e))?;

//     let tasks: Vec<Task> = {
//         let mut stmt = tx
//             .prepare_cached(
//                 "WITH SelectedTasks AS (
//                 SELECT id, completed_times, type, desc, data, classes, occupancy
//                 FROM Task
//                 WHERE completed_times < ?1 AND occupancy < ?1
//                 ORDER BY completed_times DESC, occupancy DESC
//                 LIMIT ?2
//             )
//             SELECT id, completed_times, type, desc, data, classes, occupancy
//             FROM SelectedTasks;",
//             )
//             .map_err(|e| format!("{}", e))?;

//         // Execute the query to fetch tasks
//         let task_iter = stmt
//             .query_map(params![max_labellings, tasks_per_user], |row| {
//                 Ok(Task {
//                     id: row.get::<_, usize>(0)?, // explicitly getting id as i64
//                     completed_times: row.get(1)?,
//                     r#type: match row.get::<_, String>(2)?.as_str() {
//                         "AI" => TaskType::AI,
//                         "Social" => TaskType::Social,
//                         _ => panic!(),
//                     },
//                     desc: row.get(3)?,
//                     data: row.get(4)?,
//                     classes: row.get(5)?,
//                     occupancy: row.get(6)?,
//                 })
//             })
//             .map_err(|e| format!("Task fetch failed: {}", e))?;

//         task_iter.filter_map(|t| t.ok()).collect()
//     };

//     let task_ids: Vec<usize> = tasks.iter().map(|task| task.id).collect();

//     if !task_ids.is_empty() {
//         let placeholders: Vec<String> = task_ids.iter().map(|_| "?".to_string()).collect();
//         let placeholder_str = placeholders.join(",");

//         let update_query = format!(
//             "UPDATE Task
//             SET occupancy = occupancy + 1
//             WHERE id IN ({})",
//             placeholder_str
//         );

//         let mut update_stmt = tx
//             .prepare_cached(&update_query)
//             .map_err(|e| format!("Failed to prepare UPDATE statement: {}", e))?;

//         // Execute the batched update
//         update_stmt
//             .execute(params_from_iter(task_ids.iter()))
//             .map_err(|e| format!("Update failed: {}", e))?;
//     }

//     tx.commit()
//         .map_err(|e| format!("Transaction commit failed: {}", e))?;

//     Ok(serde_json::to_string(&tasks).unwrap())
// }

// #[update]
// pub fn clear_tasks_occupancy(t_ids: Vec<usize>) -> Result<(), String> {
//     let mut conn = CONN.lock().map_err(|e| format!("{}", e))?;

//     let tx = conn
//         .transaction()
//         .map_err(|e| format!("Transaction start failed: {}", e))?;

//     {
//         let placeholders: Vec<String> = t_ids.iter().map(|_| "?".to_string()).collect();
//         let placeholder_str = placeholders.join(",");

//         let update_query = format!(
//             "UPDATE Task
//             SET occupancy = occupancy - 1
//             WHERE id IN ({})",
//             placeholder_str
//         );

//         let mut update_stmt = tx
//             .prepare_cached(&update_query)
//             .map_err(|e| format!("Failed to prepare UPDATE statement: {}", e))?;

//         // Execute the batched update
//         update_stmt
//             .execute(params_from_iter(t_ids.iter()))
//             .map_err(|e| format!("Update failed: {}", e))?;
//     }

//     tx.commit()
//         .map_err(|e| format!("Transaction commit failed: {}", e))?;

//     Ok(())
// }

// #[update]
// pub fn complete_task(
//     t_id: String,
//     wallet_address: String,
//     image_link: String,
// ) -> Result<(), String> {
//     let mut conn = CONN.lock().map_err(|e| format!("{}", e))?;

//     let tx = conn
//         .transaction()
//         .map_err(|e| format!("Transaction start failed: {}", e))?;

//     tx.execute("UPDATE Task SET occupancy = occupancy - 1, completed_times = completed_times + 1 WHERE id = ?1", [&t_id])
//         .map_err(|e| format!("Task update failed: {}", e))?;

//     let logger_id = generate_hash_id(&(t_id.clone() + &wallet_address));
//     tx.execute(
//         "INSERT INTO Task_logs (id, task_id, completed_by, image_link)
// VALUES (?1, ?2, ?3, ?4);",
//         params![logger_id, t_id, wallet_address, image_link],
//     )
//     .map_err(|e| format!("{}", e))?;

//     update_exp(wallet_address, 100, &tx)?; // give 100 exp on completion of a task
//     tx.commit()
//         .map_err(|e| format!("Transaction commit failed: {}", e))?;

//     Ok(())
// }

// pub async fn settle_tasks() -> Result<(), String> {
//     let mut conn = CONN.lock().map_err(|e| e.to_string())?;

//     let tx = conn.transaction().map_err(|e| e.to_string())?;

//     {
//         let mut stmt = tx
//             .prepare_cached(
//                 "SELECT t.id, l.completed_by, l.image_link
//                 FROM Task t
//                 JOIN Task_logs l ON t.id = l.task_id
//                 WHERE t.completed_times = ?1",
//             )
//             .map_err(|e| e.to_string())?;

//         let rows = stmt
//             .query_map(params![*MAX_NUMBER_OF_LABELLINGS_PER_TASK], |row| {
//                 Ok((
//                     row.get::<_, String>(0)?, // Task ID
//                     row.get::<_, String>(1)?, // Completed By (user)
//                     row.get::<_, String>(2)?, // Image Link
//                 ))
//             })
//             .map_err(|e| e.to_string())?;

//         let mut group_by_id: HashMap<String, Vec<(String, String)>> = HashMap::new(); // task_id -> (image_link, user)

//         for r in rows {
//             let (task_id, user, image_link): (String, String, String) =
//                 r.map_err(|e| e.to_string())?;
//             group_by_id
//                 .entry(task_id)
//                 .or_insert_with(Vec::new)
//                 .push((image_link, user));
//         }

//         let mut del_stmt = tx
//             .prepare_cached(
//                 "
//             DELETE FROM Task_logs WHERE task_id = ?1;
//             DELETE FROM Task WHERE id = ?1;
//                 ",
//             )
//             .map_err(|e| e.to_string())?;

//         for (id, v) in group_by_id {
//             let rating = fetch_images_determine_rating_increment(&v).await?;

//             if v.len() == rating.len() {
//                 return Err("Something went wrong DEVS CHEKCK".to_string());
//             };

//             for (user, increment) in rating {
//                 update_rating(user, increment, &tx)?;
//             }

//             del_stmt.execute(params![id]).map_err(|e| e.to_string())?;
//         }
//     }

//     tx.commit().map_err(|e| e.to_string())?;

//     Ok(())
// }

// // (image_links, wallet_address) -> (wallet_address, increment)
// async fn fetch_images_determine_rating_increment(
//     image_vec: &Vec<(String, String)>,
// ) -> Result<Vec<(String, usize)>, String> {
//     Ok(image_vec
//         .iter()
//         .map(|(_, user)| (user.to_owned(), 1))
//         .collect())
// }
// // placeholder
