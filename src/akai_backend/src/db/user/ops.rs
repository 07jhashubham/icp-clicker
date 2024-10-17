use anyhow::Result;
use ic_cdk::{query, update};
use ic_sqlite_features::{params, ToSql, CONN};

use super::User;

#[update]
pub fn create_new_user(
    wallet_address: String,
    name: Option<String>,
    email: Option<String>,
    twitter: Option<String>,
    instagram: Option<String>,
) -> Result<(), String> {
    let mut conn = CONN
        .lock()
        .map_err(|err| format!("Failed to acquire database connection lock {}", err))?;

    let tx = conn.transaction().map_err(|x| format!("{}", x))?;
    let _ = tx
        .execute(
            "
        INSERT INTO User (
            name,
            wallet_address,
            clicks,
            email,
            twitter,
            instagram
        ) VALUES (
            ?1,    -- name
            ?2,    -- wallet_address
            0,    -- clicks
            ?3,    -- email
            ?4,    -- twitter or NULL
            ?5    -- instagram
        );
        ",
            params![name, wallet_address, email, twitter, instagram,],
        )
        .map_err(|err| format!("{}", err))?;

    tx.commit().map_err(|e| format!("{}", e))?;
    Ok(())
}

// fn update_user_field<T>(wallet_address: String, field: &str, value: T) -> Result<(), String>
// where
//     T: ToSql,
// {
//     // if !user_exists(&wallet_address)? {
//     //     return Err(format!("User doesnt exist for wallet {}", wallet_address));
//     // }
//     let query = format!("UPDATE User SET {} = ?1 WHERE wallet_address = ?2;", field);

//     let mut conn = CONN.lock().map_err(|err| format!("{}", err))?;

//     let tx = conn.transaction().map_err(|e| format!("{}", e))?;

//     tx.execute(&query, params![value, wallet_address])
//         .map_err(|err| format!("{}", err))?;

//     tx.commit().map_err(|e| format!("{}", e))?;

//     Ok(())
// }

// #[update]
// pub fn update_email(wallet_address: String, email: String) -> Result<(), String> {
//     update_user_field(wallet_address, "email", email)
// }

// #[update]
// pub fn update_twitter(wallet_address: String, twitter: String) -> Result<(), String> {
//     update_user_field(wallet_address, "twitter", twitter)
// }

// #[update]
// pub fn update_instagram(wallet_address: String, instagram: String) -> Result<(), String> {
//     update_user_field(wallet_address, "instagram", instagram)
// }

// #[update]
// pub fn update_name(wallet_address: String, name: Option<String>) -> Result<(), String> {
//     update_user_field(wallet_address, "name", name)
// }

fn increment_field(
    wallet_address: String,
    field: &str,
    amt: usize,
    tx: &ic_sqlite_features::Transaction<'_>,
) -> Result<(), String> {
    let query = format!(
        "UPDATE User SET {} = {} + {} WHERE wallet_address = ?1 ;",
        field, field, amt
    );
    tx.execute(&query, params![wallet_address])
        .map_err(|err| format!("{}", err))?;

    Ok(())
}

#[update]
pub(crate) fn update_clicks(
    wallet_address: String,
    amt: usize,
) -> Result<(), String> {
    let mut conn = CONN.lock().map_err(|err| format!("{}", err))?;
    let tx = conn.transaction().map_err(|err| format!("{}", err))?;
    increment_field(wallet_address, "clicks", amt, &tx)?;
    tx.commit().map_err(|err| format!("{}", err))?;
    Ok(())
}
// pub(crate) fn update_exp(
//     wallet_address: String,
//     amt: usize,
//     tx: &ic_sqlite_features::Transaction<'_>,
// ) -> Result<(), String> {
//     increment_field(wallet_address, "exp", amt, tx)
// }

// pub(crate) fn update_rating(
//     wallet_address: String,
//     amt: usize,
//     tx: &ic_sqlite_features::Transaction<'_>,
// ) -> Result<(), String> {
//     increment_field(wallet_address, "rating", amt, tx)
// }

#[query]
pub fn get_user_data(wallet_address: String) -> Result<String, String> {
    let conn = CONN.lock().map_err(|err| format!("{}", err))?;

    let result = conn.query_row(
        "SELECT name, clicks, email, twitter, instagram, exp, rating FROM User WHERE wallet_address = ?1",
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
                rating: row.get(6)?
            };
            Ok(user)
        },
    );
    match result {
        Ok(u) => serde_json::to_string(&u).map_err(|err| format!("{}", err)),
        Err(err) => Err(format!("{}", err)),
    }
}

// pub fn user_exists(wallet_address: &str) -> Result<bool, String> {
//     let conn = CONN.lock().map_err(|x| format!("{}", x))?;
//     let count: i64 = conn
//         .query_row(
//             "SELECT COUNT(*) FROM User WHERE wallet_address = ?1",
//             [wallet_address],
//             |row| row.get(0),
//         )
//         .map_err(|e| format!("{}", e))?;

//     Ok(count > 0)
// }

#[update]
fn reset_clicks(wallet_address: String) -> Result<(), String> {
    let mut conn = CONN.lock().map_err(|err| format!("{}", err))?;
    let tx = conn.transaction().map_err(|err| format!("{}", err))?;
    tx.execute(
        "UPDATE User SET clicks = 0 WHERE wallet_address = ?1",
        params![wallet_address],
    )
    .map_err(|err| format!("{}", err))?;
    tx.commit().map_err(|err| format!("{}", err))?;
    Ok(())
}