use anyhow::{anyhow, Result};
use ic_cdk::{query, update};
use ic_sqlite::CONN;

#[derive(Debug)]
pub struct User {
    name: String,
    wallet_address: String,
    badges: Option<String>, // Adjust types as needed
    clicks: i32,
    email: Option<String>,
    twitter: Option<String>,
    instagram: Option<String>,
    exp: i32,
    friends: Option<String>,
    tasks: Option<String>,
    aliens: Option<String>,
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
            badges,
            clicks,
            email,
            twitter,
            instagram,
            exp,
            friends,
            tasks,
            aliens
        ) VALUES (
            $1,    -- name
            $2,    -- wallet_address
            NULLIF($3, 'NULL'),    -- badges (UUID) or NULL
            $4,    -- clicks
            $5,    -- email
            NULLIF($6, 'NULL'),    -- twitter or NULL
            $7,    -- instagram
            $8,    -- exp
            NULLIF($9, 'NULL'),    -- friends (UUID) or NULL
            NULLIF($10, 'NULL'),    -- tasks (UUID) or NULL
            NULLIF($11, 'NULL')    -- aliens (foreign key referencing Aliens.id) or NULL
        );
        ",
            [
                name.unwrap_or("NULL".to_string()),
                wallet_address,
                "NULL".to_string(), // badges (UUID) or NULL
                "0".to_string(),    // clicks
                email.unwrap_or("NULL".to_string()),
                twitter.unwrap_or("NULL".to_string()),
                instagram.unwrap_or("NULL".to_string()),
                "0".to_string(),    // exp
                "NULL".to_string(), // friends (UUID) or NULL
                "NULL".to_string(), // tasks (UUID) or NULL
                "NULL".to_string(), // aliens (foreign key referencing Aliens.id) or NULL
            ],
        )
        .map_err(|err| format!("{}", err))?;

    Ok(())
}

#[update]
pub fn update_email(wallet_address: String, email: String) -> Result<(), String> {
    let _ = CONN
        .lock()
        .map_err(|err| format!("{}", err))?
        .execute(
            "
        UPDATE User
SET email = $1
WHERE wallet_address = $2 ;
    ",
            [wallet_address, email],
        )
        .map_err(|err| format!("{}", err))?;

    Ok(())
}
#[update]
pub fn update_twitter(wallet_address: String, twitter: String) -> Result<(), String> {
    let _ = CONN
        .lock()
        .map_err(|err| format!("{}", err))?
        .execute(
            "
        UPDATE User
SET twitter = $1
WHERE wallet_address = $2 ;
    ",
            [wallet_address, twitter],
        )
        .map_err(|err| format!("{}", err));

    Ok(())
}

#[query]
pub fn get_all_users() -> String {
    let conn = CONN.lock().unwrap();

    let mut stmt = conn.prepare("SELECT name, wallet_address, badges, clicks, email, twitter, instagram, exp, friends, tasks, aliens FROM User").unwrap();
    let user_iter = stmt
        .query_map([], |row| {
            Ok(User {
                name: row.get(0).unwrap(),
                wallet_address: row.get(1).unwrap(),
                badges: row.get(2).ok(), // Handle potential nulls
                clicks: row.get(3).unwrap(),
                email: row.get(4).ok(),
                twitter: row.get(5).ok(),
                instagram: row.get(6).ok(),
                exp: row.get(7).unwrap(),
                friends: row.get(8).ok(),
                tasks: row.get(9).ok(),
                aliens: row.get(10).ok(),
            })
        })
        .unwrap();

    for user in user_iter {
        return format!("{:?}", user);
    }

    return "".to_string();
}
