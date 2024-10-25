import React, { useEffect, useState } from "react";
import Clicker from "./components/Clicker";
import Merging from "./components/Merging";
import SidePanel from "./components/SidePannel";
import UserProfile from "./components/UserProfile";
import ImageLabeler from "./components/ImageLabel";

function App() {
  const [clickCount, setClickCount] = useState(0);
  const [boxes, setBoxes] = useState([]);
  const [selectedIcon, setSelectedIcon] = useState("home"); // Default to home icon

  // Full-screen function
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

  // Disable right-click or long-press context menu to prevent copying image address or downloading image
  const disableContextMenu = (e) => {
    e.preventDefault(); // Prevent the default context menu from appearing
  };

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

  const handleClick = () => {
    setClickCount((value) => {
      if (value === 29) {
        setBoxes((prevBoxes) => {
          const nextIndex = prevBoxes.length % 12;
          return [...prevBoxes, { level: 1, index: nextIndex }];
        });
        return 24;
      } else {
        return value + 1;
      }
    });
  };

  // Function to handle icon click
  const handleIconClick = (icon) => {
    setSelectedIcon(icon); // Set the selected icon state
  };

  return (
    <>
      {/* Conditionally render components based on selectedIcon */}
      {selectedIcon === "home" && (
        <>
          <UserProfile />
          <Clicker clickCount={clickCount} handleClick={handleClick} />
          <SidePanel />
          <Merging boxes={boxes} setBoxes={setBoxes} />
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
              width: "40px", // Adjust size as necessary
              height: "auto",
              zIndex: 2,
            }}
            onContextMenu={disableContextMenu} // Disable right-click or long-press context menu on mobile/desktop
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
              onContextMenu={disableContextMenu} // Disable right-click or long-press context menu on mobile/desktop
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
            onContextMenu={disableContextMenu} // Disable right-click or long-press context menu on mobile/desktop
          />
          {selectedIcon === "home" && (
            <img
              src="/hover.png"
              alt="Hover"
              style={{
                width: "40px",
                height: "auto",
                marginTop: "5px",
              }}
              onContextMenu={disableContextMenu} // Disable right-click or long-press context menu on mobile/desktop
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
            onContextMenu={disableContextMenu} // Disable right-click or long-press context menu on mobile/desktop
          />
          {selectedIcon === "task" && (
            <img
              src="/hover.png"
              alt="Hover"
              style={{
                width: "40px",
                height: "auto",
                marginTop: "5px",
              }}
              onContextMenu={disableContextMenu} // Disable right-click or long-press context menu on mobile/desktop
            />
          )}
        </div>
      </div>
    </>
  );
}

export default App;
