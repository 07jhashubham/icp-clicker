use ic_cdk::query;
use ic_sqlite_features::CONN;

use super::User;

#[query]
pub fn get_all_users() -> String {
    let conn = CONN.lock().unwrap();
    let mut stmt = conn
        .prepare(
            "SELECT name, wallet_address, clicks, email, twitter, instagram, exp, rating FROM User",
        )
        .unwrap();

    let user_iter = stmt
        .query_map([], |row| {
            Ok(User {
                name: row.get(0).ok(),
                wallet_address: row.get(1)?,
                clicks: row.get(2)?,
                email: row.get(3).ok(),
                twitter: row.get(4).ok(),
                instagram: row.get(5).ok(),
                exp: row.get(6)?,
                rating: row.get(7)?,
            })
        })
        .unwrap();

    let u = user_iter.map(|u| u.unwrap()).collect::<Vec<_>>();

    serde_json::to_string(&u).unwrap()
}
