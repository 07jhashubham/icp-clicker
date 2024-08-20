import React from "react";
import "../Person.css";

export default function Person() {
  console.log("first");
  return (
    <div className="person-container">
      {/* Your existing Person component content goes here */}

      {/* Connect Wallet Button */}
      <button className="connect-wallet-button">Connect Wallet</button>
    </div>
  );
}
