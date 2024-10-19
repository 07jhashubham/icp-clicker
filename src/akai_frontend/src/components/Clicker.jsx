export default function Clicker({ clickCount, handleClick }) {
  const disableContextMenu = (e) => {
    e.preventDefault(); // Prevent the default context menu from appearing
  };
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
        src="storage.png" // Updated path to storage.png
        alt="Incubator"
        style={{
          width: "70%", // Adjust size as needed
          maxWidth: "100%", // Scales with the viewport width
          height: "auto",
          objectFit: "contain", // Ensures the image maintains aspect ratio
        }}
        className="responsive-image" // Assigning a class for responsive styling
        onContextMenu={disableContextMenu}
      />
      <div className="incubator-fill" />
    </div>
  );
}
