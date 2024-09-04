use ic_cdk::query;
use ic_sqlite::CONN;
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
        WHERE User.wallet_address = ?
        ",
        )
        .map_err(|e| format!("{}", e))?;

    let badge_iter = stmt
        .query_map([wallet_address], |row| {
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

// pub fn add_badge(sha256_src: )
