import React, { useState, useRef, useEffect } from "react";
import "../ImageLabeler.css";

// Hardcoded images array (URLs or paths to the images)
const images = [
  "src/images/image1.png",
  "src/images/image2.png",
  "src/images/image3.png",
  // Add more images as needed
];

const prompts = [
  "Highlight the person wearing a yellow helmet",
  "Highlight the person wearing a yellow helmet",
  "Highlight the person wearing a yellow helmet",
  // Add more prompts as needed
];

export default function ImageLabeler() {
  const [currentImage, setCurrentImage] = useState(
    images[Math.floor(Math.random() * images.length)]
  );
  const [currentPrompt, setCurrentPrompt] = useState(
    prompts[Math.floor(Math.random() * prompts.length)]
  );
  const [isDrawing, setIsDrawing] = useState(false);
  const [startPoint, setStartPoint] = useState({ x: 0, y: 0 });
  const [rectangles, setRectangles] = useState([]);

  const canvasRef = useRef(null);
  const imgRef = useRef(null);

  useEffect(() => {
    const img = imgRef.current;
    const canvas = canvasRef.current;

    // Adjust canvas size to match the image size when the image loads
    const handleImageLoad = () => {
      if (img && canvas) {
        canvas.width = img.width;
        canvas.height = img.height;
      }
    };

    // Attach the load event listener
    img.addEventListener("load", handleImageLoad);

    // Remove the event listener when the component is unmounted
    return () => {
      img.removeEventListener("load", handleImageLoad);
    };
  }, [currentImage]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    // Clear canvas before drawing
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw all rectangles
    rectangles.forEach((rect) => {
      ctx.strokeStyle = "red";
      ctx.lineWidth = 2;
      ctx.strokeRect(rect.x, rect.y, rect.width, rect.height);
    });
  }, [rectangles]);

  const handleMouseDown = (e) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    setStartPoint({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    });
    setIsDrawing(true);
  };

  const handleMouseMove = (e) => {
    if (!isDrawing) return;

    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const newRectangle = {
      x: startPoint.x,
      y: startPoint.y,
      width: x - startPoint.x,
      height: y - startPoint.y,
    };

    // Update the last rectangle being drawn
    const updatedRectangles = [...rectangles];
    updatedRectangles[updatedRectangles.length - 1] = newRectangle;
    setRectangles(updatedRectangles);
  };

  const handleMouseUp = () => {
    setIsDrawing(false);
  };

  const handleNextImage = () => {
    setCurrentImage(images[Math.floor(Math.random() * images.length)]);
    setCurrentPrompt(prompts[Math.floor(Math.random() * prompts.length)]);
    setRectangles([]); // Clear all rectangles when switching images
  };

  const handleCanvasClick = () => {
    // When the canvas is clicked, add a new rectangle
    setRectangles([...rectangles, { x: 0, y: 0, width: 0, height: 0 }]);
  };

  return (
    <div className="image-labeler">
      <div className="prompt">
        <h3>{currentPrompt}</h3>
      </div>
      <div className="image-container">
        <img
          ref={imgRef}
          src={currentImage}
          alt="Labeling"
          className="image"
          style={{ display: "block", maxWidth: "100%", height: "auto" }}
        />
        <canvas
          ref={canvasRef}
          className="canvas"
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onClick={handleCanvasClick}
        />
      </div>
      <button className="next-button" onClick={handleNextImage}>
        Submit
      </button>
    </div>
  );
}
