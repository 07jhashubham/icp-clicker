import React from "react";
export default function SidePanel() {
    return (<div className="side-panel-container relative -ml-4">
      {/* Main Image */}
      <img src="/check.png" alt="Check Background" className="check-image "/>
      {/* Positioned images inside the check.png */}
      <div className="side-panel-icons absolute inset-0 flex flex-col justify-center items-center space-y-4 scale-150">
        <img src="/group1.png" alt="Image 1" className="w-12 h-12"/>
        <img src="/group2.png" alt="Image 2" className="w-12 h-12"/>
        <img src="/group3.png" alt="Image 3" className="w-12 h-12"/>
      </div>
    </div>);
}
