export default function Clicker({ clickCount, handleClick }) {
    return (<div className="incubator-container" onClick={handleClick} style={{
            position: "relative",
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            width: "100%",
            marginBottom: "20px", // Adjusts spacing between incubator and grid
        }}>
      <img src="storage.png" // Updated path to storge.png
     alt="Incubator" style={{
            width: "70%", // Adjust size as needed
            maxWidth: "400px",
            height: "480px",
            objectFit: "contain",
        }}/>
      <div className="incubator-fill"/>
    </div>);
}
