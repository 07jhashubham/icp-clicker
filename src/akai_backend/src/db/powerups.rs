use std::str::FromStr;

use candid::CandidType;
use ic_cdk::{query, update};
use ic_rand::rng::RandomNumberGenerator;
use ic_sqlite_features::{params, CONN};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, CandidType, Clone, Copy)]
pub enum PowerupType{
    Spawner, ClickMultiplier, AutoFiller, AlienMultiplier
}

impl FromStr for PowerupType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "Spawner" => Ok(PowerupType::Spawner),
            "ClickMultiplier" => Ok(PowerupType::ClickMultiplier),
            "AutoFiller" => Ok(PowerupType::AutoFiller),
            "AlienMultiplier" => Ok(PowerupType::AlienMultiplier),
            _ => Err(format!("Invalid PowerupType: {}", s))
        }
    }
}

impl ToString for PowerupType{
    fn to_string(&self) -> String {
        match self {
            PowerupType::Spawner => "Spawner".to_string(),
            PowerupType::ClickMultiplier => "ClickMultiplier".to_string(),
            PowerupType::AutoFiller => "AutoFiller".to_string(),
            PowerupType::AlienMultiplier => "AlienMultiplier".to_string()
        }
    }
}
#[derive(Serialize, Deserialize, CandidType, Clone)]
pub struct Powerup {
    pub id: usize,
    pub r#type: PowerupType,
}


fn spawn_powerup(wallet_address: String, powerup_type: PowerupType) -> Result<(), String> {
    let mut conn = CONN.lock().map_err(|e| e.to_string())?;
    let tx = conn.transaction().map_err(|e| e.to_string())?;

    tx.execute(
        "INSERT INTO Powerups (type, owner) VALUES (?1, ?2)",
        params![powerup_type.to_string(), wallet_address]
    ).map_err(|e| e.to_string())?;

    tx.commit().map_err(|e| e.to_string())?;
    Ok(())
}

fn get_random_powerup() -> Result<usize, String> {
    return Ok(RandomNumberGenerator::<usize>::new().next() % 3usize); // basically gets any number from 1 to 5 levels
}

#[update]
pub fn spawn_random_powerup(wallet_address: String) -> Result<PowerupType, String>{
    let powerups: Vec<Powerup> = serde_json::from_str(&get_all_powerups(wallet_address.clone())?).map_err(|e| e.to_string())?;
    if powerups.len() >= 3 {
        return Err("You have reached the maximum number of powerups".to_string());
    }
    let powerup_type = match get_random_powerup()? {
        0 => PowerupType::Spawner,
        1 => PowerupType::ClickMultiplier,
        2 => PowerupType::AutoFiller,
        3 => PowerupType::AlienMultiplier,
        _ => return Err("Invalid powerup type".to_string())
    };

    spawn_powerup(wallet_address, powerup_type)?;
    Ok(powerup_type)
}

#[update]
pub fn use_powerup(wallet_address: String, powerup_id: usize) -> Result<(), String> {
    let mut conn = CONN.lock().map_err(|e| e.to_string())?;
    let tx = conn.transaction().map_err(|e| e.to_string())?;

    tx.execute(
        "DELETE FROM Powerups WHERE id = ?1 AND owner = ?2",
        params![powerup_id, wallet_address]
    ).map_err(|e| e.to_string())?;

    tx.commit().map_err(|e| e.to_string())?;
    Ok(())
}

#[query]
pub fn get_all_powerups(wallet_address: String) -> Result<String, String> {
    let conn = CONN.lock().map_err(|e| e.to_string())?;

    let mut stmt = conn
        .prepare("SELECT id, type, owner FROM Powerups WHERE owner = ?1")
        .map_err(|e| e.to_string())?;

    let powerup_iter = stmt.query_map([wallet_address], |row| {
        Ok((row.get::<_, usize>(0)?, row.get::<_, String>(1)?, row.get::<_, String>(2)?))
    }).map_err(|e| e.to_string())?;

    let mut powerups = Vec::new();
    for powerup in powerup_iter {
        powerups.push(Powerup{
            id: powerup.as_ref().map_err(|e| e.to_string())?.0,
            r#type: PowerupType::from_str(&powerup.as_ref().map_err(|e| e.to_string())?.1).map_err(|e| e.to_string())?
        });
    }

    serde_json::to_string(&powerups).map_err(|e| e.to_string())
}