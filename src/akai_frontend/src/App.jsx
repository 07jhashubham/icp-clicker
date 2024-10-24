import React, { useEffect, useState, useCallback } from "react";
import Clicker from "./components/Clicker";
import Merging from "./components/Merging";
import SidePanel from "./components/SidePannel";
import UserProfile from "./components/UserProfile";
import "./index.css";
import { useQueryCall, useUpdateCall } from "@ic-reactor/react";
import ImageLabeler from "./components/ImageLabel";
import Cookies from "js-cookie"; // Import Cookies

async function createNewUserIfNotExists(setWalletAddress, createNewUser) {
  const existingWalletAddress = Cookies.get("WalletAddress");
  if (existingWalletAddress) {
    setWalletAddress(existingWalletAddress);
    return existingWalletAddress;
  }

  const RandomWalletAddress = Math.random().toString(36).substring(2, 15);

  // Use [] to represent optional empty text fields
  const response = await createNewUser([RandomWalletAddress, [], [], [], []]);

  console.log("New User Created with Wallet Address: ", RandomWalletAddress);
  Cookies.set("WalletAddress", RandomWalletAddress);
  setWalletAddress(RandomWalletAddress);
  return RandomWalletAddress;
}

function App() {
  const [walletAddress, setWalletAddress] = useState("user1234");
  const [clickCount, setClickCount] = useState(0);
  const [boxes, setBoxes] = useState([]);
  const [powerupBoxes, setPowerupBoxes] = useState([]);
  const [selectedIcon, setSelectedIcon] = useState("home");
  const [loadingProgress, setLoadingProgress] = useState(0);

  // Fetch user data
  const { data, call: refetchData } = useQueryCall({
    functionName: "get_user_data",
    args: walletAddress ? [walletAddress] : [], // Use empty array instead of null
    enabled: !!walletAddress, // Ensure the query is only enabled when walletAddress is set
  });

  const disableContextMenu = (e) => {
    e.preventDefault(); // Prevent the default context menu from appearing
  };

  // Fetch aliens data (using useQueryCall)
  const {
    data: aliensData,
    call: refetchAliens,
    loading: aliensLoading,
    error: aliensError,
  } = useQueryCall({
    functionName: "get_aliens",
    args: walletAddress ? [walletAddress] : [],
    refetchOnMount: true,
    enabled: !!walletAddress,
  });

  const {
    data: powerupsData,
    call: refetchPowerups,
    loading: powerupsLoading,
    error: powerupsError,
  } = useQueryCall({
    functionName: "get_all_powerups",
    args: walletAddress ? [walletAddress] : [],
    refetchOnMount: true,
    enabled: !!walletAddress,
  });

  const { call: spawn_aliens } = useUpdateCall({
    functionName: "spawn_aliens",
    args: walletAddress ? [walletAddress, 1] : [],
    enabled: !!walletAddress,
  });

  const { call: spawn_powerup } = useUpdateCall({
    functionName: "spawn_random_powerup",
    args: walletAddress ? [walletAddress] : [],
    enabled: !!walletAddress,
  });

  const { call: reset_clicks } = useUpdateCall({
    functionName: "reset_clicks",
    args: walletAddress ? [walletAddress] : [],
    enabled: !!walletAddress,
  });

  const { call: createNewUser } = useUpdateCall({
    functionName: "create_new_user",
  });

  const [user, setUser] = useState(null);

  const { call: increment_clicks } = useUpdateCall({
    functionName: "update_clicks",
    args: walletAddress ? [walletAddress, 1] : [],
    enabled: !!walletAddress,
  });

  useEffect(() => {
    // Create user if not exists when the app loads
    if (!walletAddress) {
      createNewUserIfNotExists(setWalletAddress, createNewUser);
    }
  }, [walletAddress, createNewUser]);

  useEffect(() => {
    let progress = 0;
    const intervalId = setInterval(() => {
      if (progress < 100) {
        progress += 2; // Increment by 2%
        setLoadingProgress(progress);
      } else {
        clearInterval(intervalId); // Clear interval when progress reaches 100%
      }
    }, 100); // Adjust the interval time (e.g., 100ms for smooth animation)

    return () => clearInterval(intervalId); // Cleanup interval on component unmount
  }, []);

  useEffect(() => {
    // Function to handle any interaction (click or back button press)
    const handleInteraction = () => {
      requestFullScreen(); // Request full-screen
      window.removeEventListener("click", handleInteraction); // Remove listener after first interaction
    };

    // Function to handle back navigation and trigger full-screen again
    const handlePopState = () => {
      requestFullScreen();
    };

    // Listen for click events
    window.addEventListener("click", handleInteraction);

    // Listen for back/forward navigation
    window.addEventListener("popstate", handlePopState);

    // Cleanup listeners on component unmount
    return () => {
      window.removeEventListener("click", handleInteraction);
      window.removeEventListener("popstate", handlePopState);
    };
  }, []);

  const requestFullScreen = () => {
    const elem = document.documentElement; // Target the whole document
    if (elem.requestFullscreen) {
      elem.requestFullscreen();
    } else if (elem.mozRequestFullScreen) {
      elem.mozRequestFullScreen(); // Firefox
    } else if (elem.webkitRequestFullscreen) {
      elem.webkitRequestFullscreen(); // Chrome, Safari and Opera
    } else if (elem.msRequestFullscreen) {
      elem.msRequestFullscreen(); // IE/Edge
    }
  };

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
    } else if (walletAddress) {
      // Only refetch data if walletAddress is set
      refetchData();
    }
  }, [data, data?.Ok, walletAddress, refetchData]);

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

        // Map powerups to include 'index' and 'type'
        const mappedPowerups = newPowerups.map((powerup, idx) => ({
          index: idx, // Assign an index for grid placement
          type: powerup.type, // Map 'type' accordingly
          id: powerup.id, // Retain the 'id' if needed
        }));

        setPowerupBoxes((prevBoxes) => {
          if (JSON.stringify(prevBoxes) !== JSON.stringify(mappedPowerups)) {
            return mappedPowerups;
          }
          return prevBoxes;
        });
      } catch (error) {
        console.error("Failed to parse powerupsData.Ok:", error);
      }
    }
  }, [powerupsData, powerupsData?.Ok]);

  const handleClick = useCallback(() => {
    if (!walletAddress) return; // Prevent actions if walletAddress is not set

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

        spawn_powerup()
          .then((response) => {
            console.log(response);
            refetchPowerups();
          })
          .catch((error) => {
            console.error("Failed to spawn powerup:", error);
          });

        reset_clicks().catch((error) => {
          console.error("Failed to reset clicks:", error);
        });
        return 0;
      } else {
        newValue = value + 1;
      }
      return newValue;
    });

    // Send update to server in background
    increment_clicks()
      .then(() => refetchData())
      .catch((error) => {
        console.error("Failed to update clicks on server:", error);
        setClickCount((value) => value - 1);
      });
  }, [
    walletAddress,
    increment_clicks,
    refetchData,
    spawn_aliens,
    refetchAliens,
    spawn_powerup,
    refetchPowerups,
    reset_clicks,
    boxes.length,
  ]);

  // Ensure 'user' is not null before accessing 'user.clicks'
  if (!user || aliensLoading || powerupsLoading) {
    return (
      <div className="loading-screen">
        <div className="loading-container">
          <h1>Loading...</h1>
          <img
            src="/SVGs/preload3.svg"
            alt="Loading Progress"
            className="loading-progress-bar"
            style={{
              width: `${loadingProgress}%`,
              transition: "width 0.3s ease",
            }}
          />
        </div>
      </div>
    );
  }

  if (aliensError) {
    return <p>Error loading aliens data.</p>;
  }

  if (powerupsError) {
    return <p>Error loading powerups data.</p>;
  }
  // Now it's safe to access 'user.clicks'
  console.log(user.clicks);

  const handleIconClick = (icon) => {
    setSelectedIcon(icon); // Set the selected icon state
  };

  return (
    <>
      {/* Conditionally render components based on selectedIcon */}
      {selectedIcon === "home" && (
        <>
          <UserProfile user={user} />
          <Clicker clickCount={clickCount} handleClick={handleClick} />
          <SidePanel
            walletAddress={walletAddress}
            powerupBoxes={powerupBoxes}
            setPowerupBoxes={setPowerupBoxes}
          />
          <Merging
            boxes={boxes}
            setBoxes={setBoxes}
            clicks={user?.clicks}
            exp={user?.exp}
            refetchBoxes={refetchAliens}
          />
        </>
      )}

      {selectedIcon === "task" && (
        <>
          {/* <UserProfile /> */}
          <ImageLabeler />
        </>
      )}

      {/* Footer section */}
      <div
        className="footer-container"
        style={{
          position: "fixed",
          bottom: 0,
          width: "100%",
          backgroundImage: "url('/SVGs/down.svg')",
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
            onContextMenu={disableContextMenu}
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
              onContextMenu={disableContextMenu}
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
            onContextMenu={disableContextMenu}
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
              onContextMenu={disableContextMenu}
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
            onContextMenu={disableContextMenu}
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
              onContextMenu={disableContextMenu}
            />
          )}
        </div>
      </div>
    </>
  );
}

export default App;
