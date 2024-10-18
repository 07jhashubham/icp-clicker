import React, { useEffect, useState } from "react";

//chaneg to alien list functionality
const alienImages = [
  "alien-1.png",
  "alien-2.png",
  "alien-3.png",
  "alien-4.png",
];

export default function Merging({ boxes, setBoxes, exp, clicks }) {
  const [draggingBox, setDraggingBox] = useState(null);
  const [lineFillPercentageExp, setLineFillPercentageExp] = useState(0);
  const [lineFillPercentageClicks, setLineFillPercentageClicks] = useState(0);

  // Maximum value for experience, assuming 300 as the maximum level
  const maxExp = 69;
  const maxClicks = 29; // Adjust this value based on the logic of your game

  useEffect(() => {
    const fillPercentage = (exp / maxExp) * 100;
    setLineFillPercentageExp(fillPercentage);
  }, [exp]);

  // Update line fill percentage for clicks
  useEffect(() => {
    const fillPercentage = (clicks / maxClicks) * 100;
    setLineFillPercentageClicks(fillPercentage);
  }, [clicks]);

  const placeholders = Array.from({ length: 12 });

  const handleDragStart = (e, index) => {
    e.dataTransfer.setData("text/plain", index);
    setDraggingBox(index);
  };

  const handleTouchStart = (e, index) => {
    setDraggingBox(index);
  };

  const handleDrop = (e, dropIndex) => {
    e.preventDefault();
    const dragIndex = e.dataTransfer?.getData("text/plain") || draggingBox;
    const draggedBox = boxes.find((b) => b.index === parseInt(dragIndex));
    const droppedBox = boxes.find((b) => b.index === dropIndex);

    if (draggedBox && droppedBox && draggedBox.level === droppedBox.level) {
      const newLevel = draggedBox.level + 1;

      setBoxes((prevBoxes) =>
        prevBoxes
          .filter(
            (b) => b.index !== parseInt(dragIndex) && b.index !== dropIndex
          )
          .concat({ level: newLevel, index: dropIndex })
      );

      // Update backend with merged alien
      updateAlienOnBackend(draggedBox.id, droppedBox.id, newLevel);
    }

    setDraggingBox(null); // Reset dragging box
  };

  const handleTouchMove = (e) => {
    e.preventDefault();
    const touchLocation = e.targetTouches[0];
    const element = document.elementFromPoint(
      touchLocation.clientX,
      touchLocation.clientY
    );

    if (element && element.dataset.index) {
      const dropIndex = parseInt(element.dataset.index);
      if (draggingBox !== null && draggingBox !== dropIndex) {
        handleDrop(e, dropIndex);
      }
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  return (
    <div
      className="frame-container"
      style={{
        position: "relative",
        width: "100%", // Adjust the size to fit the frame image
        height: "455px",
        backgroundImage: "url('frame.png')", // Frame image for parent container
        backgroundSize: "contain",
        backgroundRepeat: "no-repeat",
        backgroundPosition: "center",
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        marginTop: "-60px", // Retain your original marginTop
      }}
    >
      {/* Adding shards.png */}
      <img
        src="shards.png"
        alt="Shards"
        style={{
          position: "absolute",
          top: "45px", // Positioning at the top
          left: "120px", // Aligning at 50px from the left
          width: "200px", // Maintain original aspect ratio
          height: "auto", // Adjust as needed for the image size
          zIndex: 0, // Ensure it's placed below other elements as needed
        }}
      />

      {/* Div for new content inside shards.png */}
      <div
        style={{
          position: "absolute",
          top: "50px", // Aligning inside shards
          left: "140px", // Matching shards position
          display: "flex",
          flexDirection: "row",
          alignItems: "center",
          zIndex: 1, // Ensuring it's on top of shards
        }}
      >
        {/* lostCoin Image */}
        <img
          src="lostCoin.png"
          alt="Lost Coin"
          style={{
            width: "40px", // Adjust size as needed
            height: "auto",
          }}
        />

        {/* Flex column div */}
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            marginLeft: "10px", // Space between lostCoin.png and column
          }}
        >
          {/* Flex row with tpN image and 40/300 text */}
          <div
            style={{
              display: "flex",
              flexDirection: "row",
              alignItems: "center",
            }}
          >
            {/* tpN Image */}
            <img
              src="tpN.png"
              alt="TPN Image"
              style={{
                width: "50px", // Adjust size as needed
                height: "auto",
              }}
            />
            {/* 40/300 text */}
            <p
              style={{
                marginLeft: "13px", // Space between tpN and the text
                color: "#fbbf24", // Tailwind yellow-400 color
                fontSize: "14px",
                margin: "0",
              }}
            >
              {exp}
            </p>
          </div>

          {/* Black Line with filling using line6.png */}
          <div
            style={{
              position: "relative", // To make line6.png absolutely positioned inside
              width: "100%",
              height: "2px",
              backgroundColor: "black",
              marginTop: "5px", // Space between row and line
              overflow: "hidden", // Hide the overflowing part of the fill image
            }}
          >
            {/* line6.png to represent filling */}
            <img
              src="Line6.png"
              alt="Line Fill"
              style={{
                position: "absolute",
                top: 0,
                left: 0,
                width: `${lineFillPercentageExp}%`, // Dynamically update width based on state
                height: "100%",
                objectFit: "cover",
              }}
            />
          </div>
        </div>
      </div>

      {/* Left image (goldRec.png) with tp.png on top */}
      <div
        className="gold-container"
        style={{
          position: "absolute",
          top: "10px",
          left: "10px",
          width: "150px",
          height: "auto",
          marginTop: "75px", // Keep your original marginTop
        }}
      >
        <img
          src="goldRec.png"
          alt="Gold Rectangle"
          style={{
            width: "100%", // Full width of the container
            height: "auto",
            zIndex: 2, // Ensure it's above the background
          }}
        />
        {/* Add tp.png on top of goldRec.png */}
        <img
          src="tp.png"
          alt="TP"
          style={{
            position: "absolute",
            top: "0px", // Adjust positioning relative to goldRec
            left: "10px", // Adjust as necessary to center it
            width: "120px", // Adjust the size of tp.png
            height: "auto",
            zIndex: 3, // Ensure it's on top of goldRec
          }}
        />
      </div>

      {/* Right image (elexerRectange.png) */}
      <div
        className="elexer-container"
        style={{
          position: "absolute",
          top: "10px",
          right: "10px",
          width: "150px", // Adjust size as needed
          height: "auto",
          zIndex: 3, // Ensure it's above the background
          marginTop: "75px", // Retain your original marginTop
        }}
      >
        <img
          src="elexerRectange.png"
          alt="Elexer Rectangle"
          style={{
            width: "100%", // Full width of the container
            height: "auto",
            zIndex: 2,
          }}
        />

        {/* New flex row div with Vector.png and a section with the black line and text */}
        <div
          className="flex-row mx-4"
          style={{
            display: "flex",
            flexDirection: "row",
            alignItems: "center",
            justifyContent: "space-around",
            marginTop: "-30px", // Space between elexerRectange and this div
          }}
        >
          {/* Vector Image */}
          <img
            src="Vector.png"
            alt="Vector Icon"
            style={{
              width: "20px",
              height: "auto",
            }}
          />

          {/* Right-side content: black line and p tag */}
          <div
            className="right-content"
            style={{
              display: "flex",
              flexDirection: "column",
              alignItems: "flex-start",
              marginLeft: "10px", // Space between Vector.png and the right content
            }}
          >
            <div
              style={{
                position: "relative", // To make line6.png absolutely positioned inside
                width: "100%",
                height: "2px",
                backgroundColor: "black",
                marginTop: "5px", // Space between row and line
                overflow: "hidden", // Hide the overflowing part of the fill image
              }}
            >
              {/* line6.png to represent filling */}
              <img
                src="Line6.png"
                alt="Line Fill"
                style={{
                  position: "absolute",
                  top: 0,
                  left: 0,
                  width: `${lineFillPercentageClicks}%`, // Dynamically update width based on state
                  height: "100%",
                  objectFit: "cover",
                }}
              />
            </div>
            {/* Yellow text */}
            <p
              className=" text-white font-mono"
              style={{
                fontSize: "14px",
                margin: "0",
              }}
            >
              {clicks}
            </p>
          </div>
        </div>
      </div>

      {/* Grid of 12 boxes inside the parent frame */}
      <div
        className="grid"
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(4, 1fr)",
          gap: "10px",
          justifyContent: "center",
          alignItems: "center",
        }}
      >
        {placeholders.map((_, index) => {
          const box = boxes.find((b) => b.index === index);
          return (
            <div
              key={index}
              data-index={index}
              className="alien-box"
              style={{
                width: "90px",
                height: "75px",
                backgroundColor: "transparent",
                display: "flex",
                justifyContent: "center",
                alignItems: "center",
                borderRadius: "10px",
                position: "relative",
                overflow: "hidden",
              }}
              draggable={box ? true : false}
              onDragStart={(e) => handleDragStart(e, index)}
              onDrop={(e) => handleDrop(e, index)}
              onDragOver={handleDragOver}
              onTouchStart={(e) => handleTouchStart(e, index)}
              onTouchMove={handleTouchMove}
              onTouchEnd={() => setDraggingBox(null)}
            >
              <img
                src="box.png"
                alt="Container"
                style={{
                  width: "100%",
                  height: "100%",
                  position: "absolute",
                  top: 0,
                  left: 0,
                  zIndex: 1,
                  objectFit: "contain",
                }}
              />
              {box && (
                <img
                  src={alienImages[box.level - 1]}
                  alt={`Alien ${box.level}`}
                  style={{
                    width: "80%",
                    height: "70%",
                    objectFit: "contain",
                    zIndex: 2,
                  }}
                />
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
