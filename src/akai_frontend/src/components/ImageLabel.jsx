import React, { useState, useRef, useEffect } from "react";
import "/src/components/ImageLabel.css";

export default function ImageLabeler() {
  const [selectedImages, setSelectedImages] = useState([]); // 3 randomly selected images
  const [currentIndex, setCurrentIndex] = useState(0); // Current image index
  const [isDrawing, setIsDrawing] = useState(false);
  const [startPoint, setStartPoint] = useState({ x: 0, y: 0 });
  const [rectangles, setRectangles] = useState([]);
  const [userId, setUserId] = useState(""); // Store the user ID for updating wallets

  const canvasRef = useRef(null);
  const imgRef = useRef(null);

  // Fetch data from the deployed API and set the user images
  useEffect(() => {
    fetch("https://task-endpoint-production.up.railway.app/getUsers")
      .then((response) => {
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
      })
      .then((fetchedData) => {
        // Randomly select 3 images
        const randomImages = fetchedData
          .sort(() => 0.5 - Math.random())
          .slice(0, 3);
        setSelectedImages(randomImages);
        setUserId(randomImages[0]._id); // Set the user ID for updating wallets
      })
      .catch((error) => {
        console.error("Error fetching data:", error);
      });
  }, []);

  useEffect(() => {
    const img = imgRef.current;
    const canvas = canvasRef.current;

    if (!img || !canvas) {
      return; // Exit if img or canvas is not yet loaded
    }

    // Adjust canvas size to match the image size when the image loads
    const handleImageLoad = () => {
      canvas.width = img.width;
      canvas.height = img.height;
    };

    img.addEventListener("load", handleImageLoad);

    return () => {
      img.removeEventListener("load", handleImageLoad);
    };
  }, [currentIndex, selectedImages]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return; // Exit if canvas is not available
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

  const handleStart = (x, y) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    setStartPoint({
      x: x - rect.left,
      y: y - rect.top,
    });
    setIsDrawing(true);

    // Start a new rectangle
    setRectangles([
      ...rectangles,
      {
        x: x - rect.left,
        y: y - rect.top,
        width: 0,
        height: 0,
      },
    ]);
  };

  const handleMove = (x, y) => {
    if (!isDrawing) return;

    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const width = x - rect.left - startPoint.x;
    const height = y - rect.top - startPoint.y;

    // Update the last rectangle being drawn
    const updatedRectangles = [...rectangles];
    updatedRectangles[updatedRectangles.length - 1] = {
      x: startPoint.x,
      y: startPoint.y,
      width,
      height,
    };
    setRectangles(updatedRectangles);
  };

  const handleMouseDown = (e) => {
    e.preventDefault(); // Prevent default behavior
    handleStart(e.clientX, e.clientY);
  };

  const handleMouseMove = (e) => {
    e.preventDefault(); // Prevent default behavior
    handleMove(e.clientX, e.clientY);
  };

  const handleMouseUp = (e) => {
    e.preventDefault(); // Prevent default behavior
    setIsDrawing(false);
  };

  const handleTouchStart = (e) => {
    e.preventDefault(); // Prevent touch scrolling
    const touch = e.touches[0]; // Only handle single touch
    handleStart(touch.clientX, touch.clientY);
  };

  const handleTouchMove = (e) => {
    e.preventDefault(); // Prevent touch scrolling
    const touch = e.touches[0]; // Only handle single touch
    handleMove(touch.clientX, touch.clientY);
  };

  const handleTouchEnd = (e) => {
    e.preventDefault(); // Prevent default behavior
    setIsDrawing(false);
  };

  // Function to generate a random wallet address (for now)
  const generateRandomWalletAddress = () => {
    return "0x" + Math.random().toString(36).substring(2, 15).toUpperCase();
  };

  const handleSubmit = () => {
    if (rectangles.length === 0) {
      return alert("Please draw a bounding box before submitting.");
    }

    // Get the coordinates of the last bounding box drawn
    const { x, y } = rectangles[rectangles.length - 1];

    // Generate a random wallet address
    const walletAddress = generateRandomWalletAddress();

    // Make a PUT request to update the wallet
    fetch(
      `https://task-endpoint-production.up.railway.app/updateWallet/${userId}`,
      {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          walletAddress,
          x,
          y,
        }),
      }
    )
      .then((response) => response.json())
      .then((data) => {
        console.log("Wallet updated:", data);
        alert("Wallet updated successfully!");
      })
      .catch((error) => {
        console.error("Error updating wallet:", error);
        alert("Error updating wallet.");
      });

    // Move to the next image
    handleNextImage();
  };

  const handleNextImage = () => {
    if (currentIndex < selectedImages.length - 1) {
      setCurrentIndex(currentIndex + 1);
      setRectangles([]); // Clear all rectangles when switching images
      setUserId(selectedImages[currentIndex + 1]._id); // Update user ID for next image
    } else {
      console.log("All images have been labeled");
    }
  };

  const currentImage = selectedImages[currentIndex]?.imageURL || "";
  const currentPrompt = "Highlight the person wearing a yellow helmet"; // Static prompt

  return (
    <div className="image-labeler">
      {selectedImages.length > 0 ? (
        <>
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
              onTouchStart={handleTouchStart}
              onTouchMove={handleTouchMove}
              onTouchEnd={handleTouchEnd}
            />
          </div>
          <button className="next-button" onClick={handleSubmit}>
            Submit
          </button>
        </>
      ) : (
        <p>Loading images...</p>
      )}
    </div>
  );
}
