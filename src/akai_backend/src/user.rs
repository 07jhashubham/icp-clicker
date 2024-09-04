use anyhow::Result;
use ic_cdk::{query, update};
use ic_sqlite::CONN;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct User {
    pub name: Option<String>,
    pub wallet_address: String,
    pub clicks: i32,
    pub email: Option<String>,
    pub twitter: Option<String>,
    pub instagram: Option<String>,
    pub exp: i32,
}

#[update]
pub fn create_new_user(
    wallet_address: String,
    name: Option<String>,
    email: Option<String>,
    twitter: Option<String>,
    instagram: Option<String>,
) -> Result<(), String> {
    let conn = CONN
        .lock()
        .map_err(|err| format!("Failed to acquire database connection lock {}", err))?;

    let _ = conn
        .execute(
            "
        INSERT INTO User (
            name,
            wallet_address,
            clicks,
            email,
            twitter,
            instagram,
            exp,
            friends,
            tasks,
            aliens
        ) VALUES (
            NULLIF($1, 'NULL'),    -- name
            $2,    -- wallet_address
            0,    -- clicks
            NULLIF($5, 'NULL'),    -- email
            NULLIF($6, 'NULL'),    -- twitter or NULL
            NULLIF($7, 'NULL'),    -- instagram
            0,    -- exp
            NULL,    -- friends (UUID) or NULL
            NULL,    -- tasks (UUID) or NULL
            NULL 
        );
        ",
            [
                name.unwrap_or("NULL".to_string()),
                wallet_address,
                email.unwrap_or("NULL".to_string()),
                twitter.unwrap_or("NULL".to_string()),
                instagram.unwrap_or("NULL".to_string()),
            ],
        )
        .map_err(|err| format!("{}", err))?;

    Ok(())
}

fn update_user_field(wallet_address: String, field: &str, value: String) -> Result<(), String> {
    let query = format!("UPDATE User SET {} = $1 WHERE wallet_address = $2;", field);

    let num_rows_affected = CONN
        .lock()
        .map_err(|err| format!("{}", err))?
        .execute(&query, [value, wallet_address])
        .map_err(|err| format!("{}", err))?;

    if num_rows_affected == 0 {
        return Err("Account Not Found".to_string());
    }

    Ok(())
}

#[update]
pub fn update_email(wallet_address: String, email: String) -> Result<(), String> {
    update_user_field(wallet_address, "email", email)
}

#[update]
pub fn update_twitter(wallet_address: String, twitter: String) -> Result<(), String> {
    update_user_field(wallet_address, "twitter", twitter)
}

#[update]
pub fn update_instagram(wallet_address: String, instagram: String) -> Result<(), String> {
    update_user_field(wallet_address, "instagram", instagram)
}

#[query]
pub fn get_all_users() -> String {
    let conn = CONN.lock().unwrap();

    let mut stmt = conn
        .prepare(
            "SELECT name, wallet_address, clicks, email, twitter, instagram, exp, friends, tasks, aliens FROM User"
        )
        .unwrap();

    let user_iter = stmt
        .query_map([], |row| {
            Ok(User {
                name: row.get(0).ok(),
                wallet_address: row.get(1).unwrap(),
                clicks: row.get(2).unwrap(),
                email: row.get(3).ok(),
                twitter: row.get(4).ok(),
                instagram: row.get(5).ok(),
                exp: row.get(6).unwrap(),
            })
        })
        .unwrap();

    let u = user_iter.map(|u| u.unwrap()).collect::<Vec<_>>();

    serde_json::to_string(&u).unwrap()
}

#[query]
pub fn get_user_data(wallet_address: String) -> Result<String, String> {
    let conn = CONN.lock().map_err(|err| format!("{}", err))?;

    let result = conn.query_row(
        "SELECT name, clicks, email, twitter, instagram, exp FROM User WHERE wallet_address = ?1",
        [&wallet_address],
        |row| {
            let user = User {
                wallet_address: wallet_address.clone(),
                name: row.get(0).ok(),
                clicks: row.get(1)?,
                email: row.get(2).ok(),
                twitter: row.get(3).ok(),
                instagram: row.get(4).ok(),
                exp: row.get(5)?,
            };
            Ok(user)
        },
    );
    match result {
        Ok(u) => serde_json::to_string(&u).map_err(|err| format!("{}", err)),
        Err(err) => Err(format!("{}", err)),
    }
}
