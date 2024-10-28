import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App.jsx";
import "./index.css";
import { ActorProvider, AgentProvider } from "@ic-reactor/react";
import { idlFactory, canisterId } from "../../declarations/akai_backend";

export const LOCAL_BUILD = true;

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <AgentProvider withProcessEnv={LOCAL_BUILD}>
      <ActorProvider idlFactory={idlFactory} canisterId={canisterId}>
        <App />
      </ActorProvider>
    </AgentProvider>
  </StrictMode>
);
