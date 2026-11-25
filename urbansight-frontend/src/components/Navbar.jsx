import React from "react";
import { Eye, ShieldCheck, Server } from "lucide-react";

function Navbar({ activePage, setActivePage, useMockMode, setUseMockMode }) {
  return (
    <header className="navbar">
      <div className="navbar-brand" onClick={() => setActivePage("home")}>
        <Eye size={20} />
        <span>UrbanSight AI</span>
      </div>

      <nav className="navbar-links">
        <button
          type="button"
          className={activePage === "home" ? "active" : ""}
          onClick={() => setActivePage("home")}
        >
          Home
        </button>
        <button
          type="button"
          className={activePage === "about" ? "active" : ""}
          onClick={() => setActivePage("about")}
        >
          About
        </button>
      </nav>

      <div className="navbar-actions">
        <span className="privacy-badge">
          <ShieldCheck size={14} /> Privacy Blur
        </span>
        <button type="button" onClick={() => setUseMockMode(!useMockMode)}>
          <Server size={14} /> {useMockMode ? "Demo Mode" : "Live API"}
        </button>
      </div>
    </header>
  );
}

export default Navbar;