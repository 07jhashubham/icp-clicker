use std::ops::Add;

use ic_cdk::update;
use ic_rand::rng::RandomNumberGenerator;
use ic_sqlite_features::{params, CONN};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Serialize, Deserialize, Debug)]
pub struct Alien {
    pub id: String,
    pub lvl: usize,
}

impl Add for Alien {
    type Output = Result<Self, String>;

    fn add(self, rhs: Self) -> Self::Output {
        if self.lvl != rhs.lvl {
            return Err("Can't combine aliens with different levels or owners".to_string());
        }

        let mut conn = CONN.lock().map_err(|e| e.to_string())?;

        let tx = conn.transaction().map_err(|e| e.to_string())?;

        tx.execute_batch(&format!(
            "
            BEGIN;
            
            UPDATE Aliens
            SET lvl = lvl + 1
            WHERE id = {}';
            
            DELETE FROM Aliens
            WHERE id = '{}';
            
            COMMIT;
        ",
            self.id, rhs.id
        ))
        .map_err(|e| e.to_string())?;

        Ok(Alien {
            id: self.id,
            lvl: self.lvl + 1,
        })
    }
}
// returns a list of aliens create with their sqlite id and lvl
#[update]
pub fn spawn_aliens(wallet_address: String, slots: usize) -> Result<String, String> {
    let mut conn = CONN.lock().map_err(|e| e.to_string())?;
    let tx = conn.transaction().map_err(|e| e.to_string())?;
    let mut inserted_ids = Vec::with_capacity(slots);
    {
        let mut stmt = tx
            .prepare("INSERT INTO Alien (lvl, owner) VALUES (?1, ?2)")
            .map_err(|e| e.to_string())?;

        for _ in 0..slots {
            let lvl = get_random_alien()?;
            stmt.execute(params![lvl, wallet_address])
                .map_err(|e| e.to_string())?;

            let alien_id = tx.last_insert_rowid();
            inserted_ids.push(json!({ "id": alien_id, "lvl": lvl }));
        }
    }

    tx.commit().map_err(|e| e.to_string())?;

    serde_json::to_string(&inserted_ids).map_err(|e| e.to_string())
}

fn get_random_alien() -> Result<usize, String> {
    return Ok((RandomNumberGenerator::<usize>::new().next() % 5) + 1); // basically gets any number from 1 to 5 levels
}

// Note: lhs alien persists while the rhs alien gets deleted, MUST simulate the same in the frontend
#[update]
pub fn combine_aliens(a: String, b: String) -> Result<String, String> {
    let a_alien: Alien = serde_json::from_str(&a).map_err(|e| e.to_string())?;

    let b_alien: Alien = serde_json::from_str(&b).map_err(|e| e.to_string())?;

    let ret = (a_alien + b_alien)?;

    serde_json::to_string(&ret).map_err(|e| e.to_string())
}
