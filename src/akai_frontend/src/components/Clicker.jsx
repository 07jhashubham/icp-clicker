import React from "react";

export default function Clicker({ clickCount, handleClick }) {
  return (
    <div
      className="incubator-container"
      onClick={handleClick}
      style={{
        position: "relative",
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        width: "100%",
        marginBottom: "20px", // Adjusts spacing between incubator and grid
      }}
    >
      <img
        src="src/images/reactor.png" // Correct path to your incubator image
        alt="Incubator"
        style={{
          width: "80%", // Adjust size as needed
          maxWidth: "400px",
          height: "420px",
          objectFit: "contain",
        }}
      />
      <div
        className="incubator-fill"
        style={{
          position: "absolute",
          bottom: 50,
          borderTopLeftRadius: "10px",
          borderTopRightRadius: "10px",
          left: 150,
          width: "43%",
          height: `${(clickCount / 29) * 73}%`,
          backgroundColor: "rgba(0, 255, 0, 0.5)", // Adjust color and opacity
          transition: "height 0.3s ease",
          zIndex: 1, // Ensure the fill is on top of the background image
          borderBottomLeftRadius: "30px", // Adjust to match the container shape if needed
          borderBottomRightRadius: "30px", // Adjust to match the container shape if needed
        }}
      />
    </div>
  );
}
