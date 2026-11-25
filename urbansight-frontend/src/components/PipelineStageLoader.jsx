import React from "react";
import { ScanSearch, BrainCircuit } from "lucide-react";

function PipelineStageLoader({ stage, previewUrl }) {
  if (stage === "idle" || stage === "complete") return null;

  return (
    <section className="card loader-card">
      {stage === "yolo_blur_loading" && (
        <div className="stage-animation">
          <div className="scanner-box">
            {previewUrl && <img src={previewUrl} alt="YOLO scanning" />}
            <div className="scan-line" />
          </div>
          <div className="loader-text">
            <ScanSearch size={20} />
            <div>
              <h3>Running Haar Cascade Blur & YOLO Detection...</h3>
              <p>Blurring faces/plates in privacy_blur.py and detecting objects in detector.py</p>
            </div>
          </div>
        </div>
      )}

      {stage === "vlm_loading" && (
        <div className="stage-animation">
          <div className="vlm-pulse">
            <BrainCircuit size={28} />
          </div>
          <div className="loader-text">
            <div>
              <h3>Running Qwen2.5-VL Scene Analysis & PDF Builder...</h3>
              <p>Generating description in vlm_analyzer.py and preparing report in pdf_generator.py</p>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}

export default PipelineStageLoader;