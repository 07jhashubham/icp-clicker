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

  // Adjust the canvas size based on the image
  useEffect(() => {
    const img = imgRef.current;
    const canvas = canvasRef.current;

    if (!img || !canvas) {
      return; // Exit if img or canvas is not yet loaded
    }

    const handleImageLoad = () => {
      // Ensure canvas size matches the image size
      canvas.width = img.clientWidth;
      canvas.height = img.clientHeight;
    };

    img.addEventListener("load", handleImageLoad);

    return () => {
      img.removeEventListener("load", handleImageLoad);
    };
  }, [currentIndex, selectedImages]);

  // Handle drawing the rectangles on the canvas
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

  // Start drawing a new rectangle
  const handleStart = (x, y) => {
    const canvas = canvasRef.current;
    const img = imgRef.current;

    // Get the canvas and image bounding rectangles
    const canvasRect = canvas.getBoundingClientRect();
    const imgRect = img.getBoundingClientRect();

    // Debug: Log the canvas and image bounding rectangles
    console.log("Canvas bounding rect:", canvasRect);
    console.log("Image bounding rect:", imgRect);

    // Debug: Log the sizes of both the canvas and the image
    console.log("Canvas size:", { width: canvas.width, height: canvas.height });
    console.log("Image size:", { width: img.width, height: img.height });

    // Continue with normal coordinate calculation for canvas-relative position
    const canvasX = x - canvasRect.left;
    const canvasY = y - canvasRect.top;

    console.log("Canvas-relative start coordinates:", { canvasX, canvasY });

    setStartPoint({ x: canvasX, y: canvasY });
    setIsDrawing(true);

    setRectangles([
      ...rectangles,
      {
        x: canvasX,
        y: canvasY,
        width: 0,
        height: 0,
      },
    ]);
  };

  // Update the rectangle as the user moves the mouse
  const handleMove = (x, y) => {
    if (!isDrawing) return;

    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();

    // Adjust for precise coordinates by accounting for canvas offset
    const canvasX = x - rect.left;
    const canvasY = y - rect.top;

    const width = canvasX - startPoint.x;
    const height = canvasY - startPoint.y;

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

  // Handle mouse events for drawing
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

  // Handle touch events for drawing
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

  // Submit the drawn rectangles
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

  // Move to the next image
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
    <div className="image-labeler mt-10 w-full relative">
      {selectedImages.length > 0 ? (
        <>
          {/* Prompt Section */}
          <div className="prompt relative w-full">
            {/* Left side image */}
            <div className="fixed mt-16 top-0 left-0">
              <div className="relative w-48 h-16">
                <img
                  src="t2-p2.png"
                  alt=""
                  className="absolute inset-0 w-full h-full "
                />
                <img
                  src="t2-p6.png"
                  alt=""
                  className="absolute inset-0 w-full h-full "
                />
              </div>
            </div>

            {/* Right side image */}
            <div className="fixed mt-16 top-0 right-0">
              <div className="relative w-48 h-16">
                <img
                  src="t2-p2.png"
                  alt=""
                  className="absolute inset-0 w-full h-full transform scale-x-[-1]"
                />
                <img
                  src="t2-p6.png"
                  alt=""
                  className="absolute inset-0 w-full h-full transform scale-x-[-1]"
                />
              </div>
            </div>

            {/* Center content */}
            <div className="flex-grow flex justify-center items-center z-20 ">
              <div className="bg-[url('/t2-p1.png')] scale-125 bg-contain bg-no-repeat bg-center flex justify-center items-center">
                <h3 className="scale-50 font-mono font-bold text-black">
                  {currentPrompt}
                </h3>
              </div>
            </div>
          </div>

          {/* Image and Canvas Wrapper */}
          <div className="relative image-wrapper bg-[url('t2-p3.png')] bg-center bg-contain bg-no-repeat w-full h-full mb-8">
            <div className="m-8 px-4 mt-16">
              <img
                ref={imgRef}
                src={currentImage}
                alt="Labeling"
                className="image"
                style={{
                  display: "block",
                  width: "auto",
                  height: "auto",
                }}
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
                style={{
                  position: "absolute", // Ensures the canvas overlays the image
                  top: 0,
                  left: 0,
                }}
              />
            </div>
          </div>

          {/* Move buttons outside image-container */}
          <div className="grid grid-cols-3 rp">
            {/* Undo/Redraw Button */}
            <div className=" col-span-1 relative">
              <img
                src="t2-p5.png"
                alt="Redraw"
                className=""
                onClick={() => setRectangles([])}
              />
            </div>

            {/* Submit Button */}
            <div className=" col-span-2 relative">
              <img
                src="t2-p4.png"
                alt="Submit"
                className=" w-full h-auto"
                onClick={handleSubmit}
              />
            </div>
          </div>
        </>
      ) : (
        <p>Loading images...</p>
      )}
    </div>
  );
}
