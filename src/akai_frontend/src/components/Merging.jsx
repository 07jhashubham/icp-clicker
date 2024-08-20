import React, { useState } from "react";

const alienImages = [
  "src/images/aliens/alien-1.png",
  "src/images/aliens/alien-2.png",
  "src/images/aliens/alien-3.png",
  "src/images/aliens/alien-4.png",
  // Add more paths as necessary
];

export default function Merging({ boxes, setBoxes }) {
  const [draggingBox, setDraggingBox] = useState(null);

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
      className="grid"
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(4, 1fr)",
        gap: "10px",
        padding: "10px",
        justifyContent: "center",
        marginBottom: "100px",
        marginLeft: "80px",
        marginRight: "80px",
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
              width: "80px",
              height: "80px",
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
              src="src/images/container.png"
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
                  height: "80%",
                  objectFit: "contain",
                  zIndex: 2,
                }}
              />
            )}
          </div>
        );
      })}
    </div>
  );
}
