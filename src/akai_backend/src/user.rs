use anyhow::{anyhow, Result};
use ic_sqlite::CONN;





pub fn create_new_user(wallet_address: String, name: Option<String>, email: Option<String>, twitter: Option<String>, instagram: Option<String>) -> Result<()> {
    let conn = CONN.lock().map_err(|err| anyhow!("Failed to acquire database connection lock {}", err))?;

    let _ = conn.execute(
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
    $1 ,                   -- name
    $2 ,          -- wallet_address
    $3 , -- badges (UUID) or NULL
    $4 ,                           -- clicks
    $5 ,        -- email
    $6 ,                   -- twitter or NULL
    $7 ,              -- instagram
    $8 ,                          -- exp
    $9 , -- friends (UUID) or NULL
    $10, -- tasks (UUID)
    $11                           -- aliens (foreign key referencing Aliens.id) or NULL
);

        ", 
        [name.unwrap_or("NULL".to_string()), wallet_address, "NULL".to_string(), "0".to_string(), email.unwrap_or("NULL".to_string()), twitter.unwrap_or("NULL".to_string()), instagram.unwrap_or("NULL".to_string()), "0".to_string(), "NULL".to_string(), "NULL".to_string(), "NULL".to_string()]
    ).map_err(|err| anyhow!("{}", err))?;
    Ok(())
}

pub fn update_email(wallet_address: String, email: String) -> Result<()>{
    let _ = CONN.lock().map_err(|err| anyhow!("{}", err))?.execute("
        UPDATE User
SET email = $1
WHERE wallet_address = $2 ;
    ", [wallet_address, email])?;

    Ok(())
}

pub fn update_twitter(wallet_address: String, twitter: String) -> Result<()>{
    let _ = CONN.lock().map_err(|err| anyhow!("{}", err))?.execute("
        UPDATE User
SET twitter = $1
WHERE wallet_address = $2 ;
    ", [wallet_address, twitter])?;

    Ok(())
}

