import React, { useEffect, useState, useCallback } from "react";
import Clicker from "./components/Clicker";
import Merging from "./components/Merging";
import SidePanel from "./components/SidePannel";
import UserProfile from "./components/UserProfile";
import "./index.css";
import { useQueryCall, useUpdateCall } from "@ic-reactor/react";

function App() {
  const [clickCount, setClickCount] = useState(0);
  const [boxes, setBoxes] = useState([]);
  const [powerupBoxes, setPowerupBoxes] = useState([]);
  const [selectedIcon, setSelectedIcon] = useState("home");

  // Fetch user data
  const { data, call: refetchData } = useQueryCall({
    functionName: "get_user_data",
    args: ["user1234"],
    refetchOnMount: true,
  });

  // Fetch aliens data (using useQueryCall)
  const {
    data: aliensData,
    call: refetchAliens,
    loading: aliensLoading,
    error: aliensError,
  } = useQueryCall({
    functionName: "get_aliens",
    args: ["user1234"],
    refetchOnMount: true,
  });
  
  const {
    data: powerupsData,
    call: refetchPowerups,
    loading: powerupsLoading,
    error: powerupsError,
  } = useQueryCall({
    functionName: "get_all_powerups",
    args: ["user1234"],
    refetchOnMount: true,
  });

  const { call: spawn_aliens } = useUpdateCall({
    functionName: "spawn_aliens",
    args: ["user1234", 1],
  });

  const { call: spawn_powerup } = useUpdateCall({
    functionName: "spawn_random_powerup",
    args: ["user1234"],
  });

  const { call: reset_clicks } = useUpdateCall({
    functionName: "reset_clicks",
    args: ["user1234"],
  });

  const [user, setUser] = useState(null);

  const { call: increment_clicks } = useUpdateCall({
    functionName: "update_clicks",
    args: ["user1234", 1],
  });

  useEffect(() => {
    if (data && data.Ok) {
      try {
        const newUser = JSON.parse(data.Ok);
  
        setUser((prevUser) => {
          if (JSON.stringify(prevUser) !== JSON.stringify(newUser)) {
            return newUser;
          }
          return prevUser;
        });
      } catch (error) {
        console.error("Failed to parse data.Ok:", error);
      }
    }
  }, [data, data?.Ok]);
  
  useEffect(() => {
    if (aliensData && aliensData.Ok) {
      try {
        const newAliens = JSON.parse(aliensData.Ok);
  
        // Map aliens to include 'index' and 'level'
        const mappedAliens = newAliens.map((alien, idx) => ({
          index: idx, // Assign an index for grid placement
          level: alien.lvl, // Map 'lvl' to 'level' for consistency
          id: alien.id, // Retain the 'id' if needed
        }));
  
        setBoxes((prevBoxes) => {
          if (JSON.stringify(prevBoxes) !== JSON.stringify(mappedAliens)) {
            return mappedAliens;
          }
          return prevBoxes;
        });
      } catch (error) {
        console.error("Failed to parse aliensData.Ok:", error);
      }
    }
  }, [aliensData, aliensData?.Ok]);

  useEffect(() => {
    if (powerupsData && powerupsData.Ok) {
      try {
        const newPowerups = JSON.parse(powerupsData.Ok);
  
        // Map aliens to include 'index' and 'level'
        const mappedPowerups = newPowerups.map((powerup, idx) => ({
          index: idx, // Assign an index for grid placement
          type: powerup.type, // Map 'lvl' to 'level' for consistency
          id: powerup.id, // Retain the 'id' if needed
        }));
  
        setPowerupBoxes((prevBoxes) => {
          if (JSON.stringify(prevBoxes) !== JSON.stringify(mappedPowerups)) {
            return mappedPowerups;
          }
          return prevBoxes;
        });
      } catch (error) {
        console.error("Failed to parse aliensData.Ok:", error);
      }
    }
  }, [powerupsData, powerupsData?.Ok]);

  const handleClick = useCallback(() => {
    // Update UI immediately
    setClickCount((value) => {
      let newValue;
      if (value === 29 && boxes.length <= 12) {
        newValue = 24;

        // Call 'spawn_aliens' and refetch aliens data after it completes
        spawn_aliens()
          .then((response) => {
            console.log(response);
            // After spawning aliens, refetch aliens data
            refetchAliens();
          })
          .catch((error) => {
            console.error("Failed to spawn aliens:", error);
          });
        
        spawn_powerup().then((response) => {
          console.log(response);
          refetchPowerups();
        }).catch((error) => {
          console.error("Failed to spawn powerup:", error);
        });

        reset_clicks().then((res) => setClickCount(0)).catch((error) => {console.error("Failed to reset clicks:", error);});
      } else {
        newValue = value + 1;
      }
      return newValue;
    });

    // Send update to server in background
    increment_clicks()
      .then(refetchData)
      .catch((error) => {
        console.error("Failed to update clicks on server:", error);
        setClickCount((value) => value - 1);
      });
  }, [increment_clicks, refetchData, spawn_aliens, refetchAliens]);

  // Ensure 'user' is not null before accessing 'user.clicks'
  if (!user || aliensLoading || powerupsLoading) {
    return <p>Loading data...</p>;
  }

  if (aliensError) {
    return <p>Error loading aliens data.</p>;
  }

  if (powerupsError) {
    return <p>Error loading aliens data.</p>;
  }
  // Now it's safe to access 'user.clicks'
  console.log(user.clicks);

  return (
    <>
      {/* Conditionally render components based on selectedIcon */}
      {selectedIcon === "home" && (
        <>
          <UserProfile user={user} />
          <Clicker clickCount={clickCount} handleClick={handleClick} />
          <SidePanel powerupBoxes={powerupBoxes} setPowerupBoxes={setPowerupBoxes}/>
          <Merging boxes={boxes} setBoxes={setBoxes} clicks={user?.clicks} exp={user?.exp}/>
        </>
      )}

      {/* Footer section */}
      <div
        className="footer-container"
        style={{
          position: "fixed",
          bottom: 0,
          width: "100%",
          backgroundImage: "url('/Group35.png')",
          backgroundSize: "cover",
          backgroundPosition: "center",
          height: "80px",
          display: "flex",
          justifyContent: "space-around",
          alignItems: "center",
        }}
      >
        {/* Profile icon */}
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
          }}
          onClick={() => handleIconClick("profile")}
        >
          <img
            src="/profile.png"
            alt="Profile"
            style={{
              width: "40px",
              height: "auto",
              zIndex: 2,
            }}
          />
          {/* Conditionally render hover image under profile */}
          {selectedIcon === "profile" && (
            <img
              src="/hover.png"
              alt="Hover"
              style={{
                width: "40px",
                height: "auto",
                marginTop: "5px",
              }}
            />
          )}
        </div>

        {/* Home icon */}
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
          }}
          onClick={() => handleIconClick("home")}
        >
          <img
            src="/home.png"
            alt="Home"
            style={{
              width: "40px",
              height: "auto",
              zIndex: 2,
            }}
          />
          {/* Conditionally render hover image under home */}
          {selectedIcon === "home" && (
            <img
              src="/hover.png"
              alt="Hover"
              style={{
                width: "40px",
                height: "auto",
                marginTop: "5px",
              }}
            />
          )}
        </div>

        {/* Task icon */}
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
          }}
          onClick={() => handleIconClick("task")}
        >
          <img
            src="/task.png"
            alt="Task"
            style={{
              width: "40px",
              height: "auto",
              zIndex: 2,
            }}
          />
          {/* Conditionally render hover image under task */}
          {selectedIcon === "task" && (
            <img
              src="/hover.png"
              alt="Hover"
              style={{
                width: "40px",
                height: "auto",
                marginTop: "5px",
              }}
            />
          )}
        </div>
      </div>
    </>
  );
}

export default App;
