import React, { useState } from "react";
import Navbar from "./components/Navbar";
import Home from "./components/Home";
import About from "./components/About";
import "./App.css";

function App() {
  const [activePage, setActivePage] = useState("home");
  const [useMockMode, setUseMockMode] = useState(false);

  return (
    <div className="app-container">
      <Navbar
        activePage={activePage}
        setActivePage={setActivePage}
        useMockMode={useMockMode}
        setUseMockMode={setUseMockMode}
      />

      <main className="content-area">
        {activePage === "home" ? <Home useMockMode={useMockMode} /> : <About />}
      </main>
    </div>
  );
}

export default App;