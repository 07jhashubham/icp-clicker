// ConnectScreen.jsx
import React from "react";

function ConnectScreen({ walletAddress, onConnect }) {
  return (
    <div className="connect-screen">
      {walletAddress && <p>Principal ID: {walletAddress}</p>}
      <button onClick={onConnect} className="connect-button">
        Connect
      </button>
    </div>
  );
}

export default ConnectScreen;
