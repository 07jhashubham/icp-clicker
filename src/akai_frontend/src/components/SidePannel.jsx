import { useUpdateCall } from "@ic-reactor/react";
import React from "react";

export default function SidePanel({
  walletAddress,
  powerupBoxes,
  setPowerupBoxes,
}) {
  const { call: use_powerup } = useUpdateCall({
    functionName: "use_powerup",
  });

  const usePowerup = (index, powerupId) => {
    // Call the hook with the powerupId dynamically
    use_powerup([walletAddress, powerupId]).then(() => {
      setPowerupBoxes((prev) => {
        const newBoxes = [...prev];
        newBoxes.splice(index, 1);
        return newBoxes;
      });
    });
  };

  const disableContextMenu = (e) => {
    e.preventDefault(); // Prevent the default context menu from appearing
  };

  return (
    <div className="side-panel-container relative -ml-4">
      {/* Main Image */}
      <img
        src="/SVGs/powerup-holder.svg"
        alt="Check Background"
        className="check-image"
        onContextMenu={disableContextMenu}
      />

      {/* Slots for the power-ups */}
      <div className="side-panel-icons absolute inset-0 flex flex-col justify-center items-center space-y-4">
        {powerupBoxes.map((box, index) => {
          // Render the corresponding image based on the power-up type
          let powerupImage = null;
          switch (box?.type) {
            case "ClickMultiplier":
              powerupImage = "/SVGs/powerup-1.svg";
              break;
            case "AutoFiller":
              powerupImage = "/SVGs/powerup-2.svg";
              break;
            case "Spawner":
              powerupImage = "/SVGs/powerup-3.svg";
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
                className="w-12 h-12 side-panel-icon"
                onClick={() => usePowerup(index, box.id)}
                onContextMenu={disableContextMenu}
              />
            )
          );
        })}
      </div>
    </div>
  );
}
