import React from "react";

function About() {
  return (
    <section className="card about-card">
      <h2>UrbanSight AI Pipeline (PS 112)</h2>
      <p>
        This frontend connects to a modular Python backend that processes street images in stages:
      </p>
      <ul>
        <li>
          <strong>privacy_blur.py:</strong> Blurs faces and license plates with OpenCV Haar Cascades.
        </li>
        <li>
          <strong>detector.py:</strong> Uses the included YOLOP checkpoint to detect vehicles and return bounding boxes.
        </li>
        <li>
          <strong>vlm_analyzer.py:</strong> Creates a count based summary from YOLOP results. Qwen2.5-VL is not connected.
        </li>
        <li>
          <strong>pdf_generator.py:</strong> Creates a downloadable PDF report from the analysis output.
        </li>
      </ul>
    </section>
  );
}

export default About;
