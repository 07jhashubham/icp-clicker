import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App.jsx'
import './index.css'
import { ActorProvider, AgentProvider } from '@ic-reactor/react';
import { idlFactory } from '../../declarations/akai_backend';
import { canisterId } from '../../declarations/deployer';
createRoot(document.getElementById('root')).render(
  <StrictMode>
    <AgentProvider withProcessEnv={false} host='https://icp0.io'>
      <ActorProvider idlFactory={idlFactory} canisterId={canisterId}>
        <App />
      </ActorProvider>
    </AgentProvider>
  </StrictMode>,
)
