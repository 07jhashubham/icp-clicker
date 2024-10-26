export default function Clicker({ clickCount, handleClick }) {
  const disableContextMenu = (e) => {
    e.preventDefault(); // Prevent the default context menu from appearing
  };

  // Define the max click count before the incubator reaches 100% height
  const maxClickCount = 30;
  const fillHeightPercentage = (clickCount / maxClickCount) * 70;

  return (
    <div
      className="incubator-container"
      onClick={handleClick}
      style={{
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        width: "100%",
        marginBottom: "20px", // Adjusts spacing between incubator and grid
        position: "relative", // Allows absolute positioning for the fill element
      }}
    >
      <img
        src="storage.png" // Updated path to storage.png
        alt="Incubator"
        style={{
          width: "70%", // Adjust size as needed
          maxWidth: "100%", // Scales with the viewport width
          height: "auto",
          zIndex: "1",
          objectFit: "contain", // Ensures the image maintains aspect ratio
        }}
        className="responsive-image" // Assigning a class for responsive styling
        onContextMenu={disableContextMenu}
      />
      <div
        className="incubator-fill neon-green-bloom"
        style={{
          height: `${fillHeightPercentage}%`,
          transition: "height 0.3s ease-in-out", // Smooth transition for height change
        }}
      />
    </div>
  );
}
