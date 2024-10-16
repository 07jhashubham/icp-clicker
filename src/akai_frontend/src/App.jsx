import Clicker from "./components/Clicker";
import Merging from "./components/Merging";
import SidePanel from "./components/SidePannel";
import UserProfile from "./components/UserProfile";
import { useState } from "react";
import './index.css'
function App() {
  const [clickCount, setClickCount] = useState(0);
  const [boxes, setBoxes] = useState([]);
  const [selectedIcon, setSelectedIcon] = useState("home"); // Default to home icon

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

      {/* Footer section */}
      <div
        className="footer-container"
        style={{
          position: "fixed",
          bottom: 0,
          width: "100%",
          backgroundImage: "url('/Group35.png')", // Ensure the correct path
          backgroundSize: "cover",
          backgroundPosition: "center",
          height: "80px", // Adjust height as necessary
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
          />
          {/* Conditionally render hover image under profile */}
          {selectedIcon === "profile" && (
            <img
              src="/hover.png"
              alt="Hover"
              style={{
                width: "40px", // Adjust size as necessary
                height: "auto",
                marginTop: "5px", // Space between profile and hover image
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
              width: "40px", // Adjust size as necessary
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
                width: "40px", // Adjust size as necessary
                height: "auto",
                marginTop: "5px", // Space between home and hover image
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
              width: "40px", // Adjust size as necessary
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
                width: "40px", // Adjust size as necessary
                height: "auto",
                marginTop: "5px", // Space between task and hover image
              }}
            />
          )}
        </div>
      </div>
    </>
  );
}

export default App;
