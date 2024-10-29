import { useAuth } from "@ic-reactor/react";
import React from "react";

const Login = () => {
  const {
    login,
    logout,
    loginLoading,
    loginError,
    identity,
    authenticating,
    authenticated,
  } = useAuth();

  return (
    <div className=" flex items-center justify-center h-screen">
      {/* <h2>Login:</h2>
      <div>
        {loginLoading && <div>Loading...</div>}
        {loginError ? <div>{JSON.stringify(loginError)}</div> : null}
        {identity && <div>{identity.getPrincipal().toText()}</div>}
      </div> */}
      {authenticated ? (
        <div>
          <button onClick={logout}>Logout</button>
        </div>
      ) : (
        <div>
          <button
            onClick={login}
            disabled={authenticating}
            className=" bg-blue-500 px-6 py-2 rounded-lg text-3xl max-w-xs"
          >
            Login with Internet Identity
          </button>
        </div>
      )}
    </div>
  );
};

export default Login;
