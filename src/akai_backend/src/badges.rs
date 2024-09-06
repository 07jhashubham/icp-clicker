// ================================================================================================================================================================================
// >>>>>>>>>>>>>>>>>>>>>>>>>>>> Note: Badges might be replaced by nfts <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
// ================================================================================================================================================================================

use crate::{user::user_exists, utils::generate_hash_id, BADGES};
use ic_cdk::query;
use ic_sqlite_features::{params, CONN};
use serde::{Deserialize, Serialize};
#[derive(Serialize, Deserialize, Debug)]
pub struct Badges {
    pub id: String,
    pub src: String,
    pub lvl: usize,
}
#[query]
pub fn get_user_badges(wallet_address: String) -> Result<String, String> {
    let conn = CONN.lock().map_err(|e| format!("{}", e))?;

    let mut stmt = conn
        .prepare(
            "
        SELECT Badges.id, Badges.src, Badges.lvl
        FROM Badges
        INNER JOIN User ON Badges.owner = User.wallet_address
        WHERE User.wallet_address = ?1
        ",
        )
        .map_err(|e| format!("{}", e))?;

    let badge_iter = stmt
        .query_map(params![wallet_address], |row| {
            Ok(Badges {
                id: row.get(0)?,
                src: row.get(1)?,
                lvl: row.get(2)?,
            })
        })
        .map_err(|e| format!("{}", e))?;

    let badges = badge_iter
        .filter_map(|badge| badge.ok())
        .collect::<Vec<Badges>>();

    serde_json::to_string(&badges).map_err(|e| format!("{}", e))
}
#[allow(dead_code)]
pub fn add_badge(lvl: u32, wallet_address: String) -> Result<(), String> {
    if !user_exists(&wallet_address)? {
        eprintln!(
            "User with wallet address {} does not exist.",
            wallet_address
        );
        return Err("User Not found".to_string());
    }

    let mut conn = CONN.lock().map_err(|x| format!("{}", x))?;
    let img = {
        match BADGES.with(|p| p.borrow().get(&lvl)) {
            Some(s) => s.clone(),
            None => {
                todo!() // call some function that generates the new image and inserts it into the ref
            }
        }
    };

    let unique_id = generate_hash_id(&(wallet_address.clone() + &lvl.to_string()));

    let tx = conn.transaction().map_err(|e| format!("{}", e))?;
    let _ = tx
        .execute(
            "INSERT INTO Badge (id, lvl, src, owner) VALUES ( ?1 , ?2 , ?3 , ?4 )",
            params![unique_id, lvl, img, wallet_address],
        )
        .map_err(|err| format!("{}", err))?;

    tx.commit().map_err(|e| format!("{}", e))?;
    Ok(())
}
