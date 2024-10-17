import { useUpdateCall } from "@ic-reactor/react";
import React from "react";

export default function SidePanel({powerupBoxes, setPowerupBoxes}) {
  console.log(powerupBoxes);

  const { call: use_powerup } = useUpdateCall({
    functionName: "use_powerup",
  });

  const usePowerup = (index, powerupId) => {
    // Call the hook with the powerupId dynamically
    use_powerup(["user1234", powerupId]).then(() => {
      setPowerupBoxes((prev) => {
        const newBoxes = [...prev];
        newBoxes.splice(index, 1);
        return newBoxes;
      });
    });
  };

  return (
    <div className="side-panel-container relative -ml-4">
      {/* Main Image */}
      <img src="/check.png" alt="Check Background" className="check-image " />
      
      {/* Slots for the power-ups */}
      <div className="side-panel-icons absolute inset-0 flex flex-col justify-center items-center space-y-4 scale-150">
        {powerupBoxes.map((box, index) => {
          // Render the corresponding image based on the power-up type
          let powerupImage = null;
          switch (box?.type) {
            case "ClickMultiplier":
              powerupImage = "/group1.png";
              break;
            case "AutoFiller":
              powerupImage = "/group2.png";
              break;
            case "Spawner":
              powerupImage = "/group3.png";
              break;
            default:
              powerupImage = null;
          }

          return (
            powerupImage && (
              <img
                key={index}
                src={powerupImage}
                alt={`Powerup ${box.type}`}
                className="w-12 h-12"
                onClick={() => usePowerup(index, box.id)}
              />
            )
          );
        })}
      </div>
    </div>
  );
}
