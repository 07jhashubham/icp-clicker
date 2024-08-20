import React, { useState } from "react";
import { BrowserRouter as Router, Route, Routes, Link } from "react-router-dom";
import "./App.css";
import Clicker from "./components/Clicker";
import Merging from "./components/Merging";
import Task from "./components/Task";
import Person from "./components/Person";

import sawGif from "./images/saw.gif";
import playgroundGif from "./images/Playground.gif";
import personGif from "./images/Person.gif";
import ImageLabeler from "./components/ImageLabeler";
export default function App() {
  const [clickCount, setClickCount] = useState(0);
  const [boxes, setBoxes] = useState([]);

  const handleClick = () => {
    setClickCount((value) => {
      if (value === 29) {
        setBoxes((prevBoxes) => {
          const nextIndex = prevBoxes.length % 12;
          return [...prevBoxes, { level: 1, index: nextIndex }];
        });
        return 24;
      } else {
        return value + 1;
      }
    });
  };

  return (
    <Router>
      <div className="content">
        <Routes>
          <Route
            path="/"
            element={
              <>
                <Clicker clickCount={clickCount} handleClick={handleClick} />
                <Merging boxes={boxes} setBoxes={setBoxes} />
              </>
            }
          />
          <Route path="/task" element={<Task />} />
          <Route path="/person" element={<Person />} />
          <Route path="/image-label" element={<ImageLabeler />} />
        </Routes>
        <div className="bottom-navigation flex flex-row justify-between items-center w-80 px-4 py-2 fixed bottom-0 ">
          <Link to="/task">
            <img src={sawGif} alt="Home" className="h-12" />
          </Link>
          <Link to="/">
            <img src={playgroundGif} alt="Task" className="h-20" />
          </Link>
          <Link to="/person">
            <img src={personGif} alt="Connect" className="h-12" />
          </Link>
        </div>
      </div>
    </Router>
  );
}
