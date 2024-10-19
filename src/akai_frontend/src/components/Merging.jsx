import React, { useEffect, useState } from "react";
import { DndProvider, useDrag, useDrop } from "react-dnd";
import { HTML5Backend } from "react-dnd-html5-backend";
import { TouchBackend } from "react-dnd-touch-backend";
import { isMobile } from "react-device-detect";
import { useUpdateCall } from "@ic-reactor/react";
//chaneg to alien list functionality
const alienImages = [
  "alien-1.png",
  "alien-2.png",
  "alien-3.png",
  "alien-4.png",
];

const ItemType = "ALIEN";

function AlienBox({ box, index, onDrop }) {
  const [{ isDragging }, dragRef] = useDrag({
    type: ItemType,
    item: { index },
    collect: (monitor) => ({
      isDragging: monitor.isDragging(),
    }),
  });

  const [{ isOver }, dropRef] = useDrop({
    accept: ItemType,
    drop: (item) => {
      onDrop(item.index, index);
    },
    collect: (monitor) => ({
      isOver: monitor.isOver(),
    }),
  });

  return (
    <div
      ref={(node) => dragRef(dropRef(node))} // Combine refs
      data-index={index}
      className="alien-box"
      style={{
        width: "90px",
        height: "55px",
        backgroundColor: isOver ? "lightblue" : "transparent",
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        borderRadius: "10px",
        position: "relative",
        opacity: isDragging ? 0.5 : 1,
      }}
    >
      <img
        src="box.png"
        alt="Container"
        style={{
          // width: "100%",
          // height: "100%",
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
}

export default function Merging({
  boxes,
  setBoxes,
  exp,
  clicks,
  refetchBoxes,
}) {
  const [lineFillPercentageExp, setLineFillPercentageExp] = useState(0);
  const [lineFillPercentageClicks, setLineFillPercentageClicks] = useState(0);

  const { call: combine_aliens } = useUpdateCall({
    functionName: "combine_aliens",
    // Removed args to allow dynamic arguments
  });

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

  // Updated handleDrop function
  const handleDrop = (dragIndex, dropIndex) => {
    const draggedBox = boxes.find((b) => b.index === dragIndex);
    const droppedBox = boxes.find((b) => b.index === dropIndex);

    if (draggedBox && droppedBox && draggedBox.level === droppedBox.level) {
      const newLevel = draggedBox.level + 1;

      // Update the state with the merged box and remove the dragged and dropped boxes
      setBoxes((prevBoxes) => {
        const updatedBoxes = prevBoxes
          .filter((b) => b.index !== dragIndex && b.index !== dropIndex)
          .concat({ level: newLevel, index: dropIndex, id: draggedBox.id }); // Place the new level alien in the dropped index

        return updatedBoxes;
      });

      // Prepare the arguments to send to the backend
      const argA = JSON.stringify({ lvl: draggedBox.level, id: draggedBox.id });
      const argB = JSON.stringify({ lvl: droppedBox.level, id: droppedBox.id });

      // Call the combine_aliens function with exactly two arguments
      combine_aliens([argA, argB])
        .then((res) => {
          refetchBoxes().then(() => console.log("Boxes updated:", res));
        })
        .catch((err) => {
          console.error("Combine aliens failed:", err);
        });
    }
  };
  const disableContextMenu = (e) => e.preventDefault();

  return (
    <DndProvider
      backend={isMobile ? TouchBackend : HTML5Backend} // Use touch backend for mobile and HTML5 backend for desktop
      options={isMobile ? { enableMouseEvents: true } : {}}
    >
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
          marginTop: "-90px", // Retain your original marginTop
        }}
        onContextMenu={disableContextMenu}
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
          onContextMenu={disableContextMenu}
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
            onContextMenu={disableContextMenu}
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
                onContextMenu={disableContextMenu}
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
                onContextMenu={disableContextMenu}
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
          onContextMenu={disableContextMenu}
        >
          <img
            src="goldRec.png"
            alt="Gold Rectangle"
            style={{
              width: "100%", // Full width of the container
              height: "auto",
              zIndex: 2, // Ensure it's above the background
            }}
            onContextMenu={disableContextMenu}
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
            onContextMenu={disableContextMenu}
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
            onContextMenu={disableContextMenu}
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
              onContextMenu={disableContextMenu}
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
                  onContextMenu={disableContextMenu}
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
          className="grid box12Container"
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(4, 1fr)",
            gap: "10px",
            justifyContent: "center",
            alignItems: "center",
          }}
        >
          {Array.from({ length: 12 }).map((_, index) => {
            const box = boxes.find((b) => b.index === index);
            return (
              <AlienBox
                key={index}
                box={box}
                index={index}
                onDrop={handleDrop}
              />
            );
          })}
        </div>
      </div>
    </DndProvider>
  );
}
