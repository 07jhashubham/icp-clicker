import { useAuth } from '@ic-reactor/react'
import React from 'react'

const Login = () => {
    const {
      login,
      logout,
      loginLoading,
      loginError,
      identity,
      authenticating,
      authenticated,
    } = useAuth()
  
    return (
      <div>
        <h2>Login:</h2>
        <div>
          {loginLoading && <div>Loading...</div>}
          {loginError ? <div>{JSON.stringify(loginError)}</div> : null}
          {identity && <div>{identity.getPrincipal().toText()}</div>}
        </div>
        {authenticated ? (
          <div>
            <button onClick={logout}>Logout</button>
          </div>
        ) : (
          <div>
            <button onClick={login} disabled={authenticating}>
              Login
            </button>
          </div>
        )}
      </div>
    )
  }
  
  export default Login