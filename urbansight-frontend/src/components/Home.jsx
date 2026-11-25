import React, { useState } from "react";
import UploadForm from "./UploadForm";
import PipelineStageLoader from "./PipelineStageLoader";
import AnalysisResult from "./AnalysisResult";

const API_BASE_URL = "http://localhost:8000";

function makeDemoAnnotatedImage(src) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = img.width;
      canvas.height = img.height;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(img, 0, 0);

      ctx.fillStyle = "rgba(15, 23, 42, 0.75)";
      ctx.fillRect(img.width * 0.4, img.height * 0.35, img.width * 0.14, img.height * 0.12);
      ctx.fillStyle = "#38bdf8";
      ctx.font = "bold 16px sans-serif";
      ctx.fillText("BLURRED", img.width * 0.4, img.height * 0.34);

      ctx.strokeStyle = "#10b981";
      ctx.lineWidth = 3;
      ctx.strokeRect(img.width * 0.15, img.height * 0.45, img.width * 0.25, img.height * 0.3);
      ctx.fillStyle = "#10b981";
      ctx.fillText("Car 0.94", img.width * 0.15, img.height * 0.44);

      resolve(canvas.toDataURL("image/jpeg", 0.9));
    };
    img.src = src;
  });
}

function Home({ useMockMode }) {
  const [imageFile, setImageFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [userPrompt, setUserPrompt] = useState(
    "Describe traffic flow, road conditions, and pedestrian safety."
  );
  const [stage, setStage] = useState("idle");
  const [stage1Data, setStage1Data] = useState(null);
  const [stage2Data, setStage2Data] = useState(null);
  const [error, setError] = useState("");

  const handleFileSelect = (file) => {
    setImageFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setStage("idle");
    setStage1Data(null);
    setStage2Data(null);
    setError("");
  };

  const handleClear = () => {
    setImageFile(null);
    setPreviewUrl("");
    setStage("idle");
    setStage1Data(null);
    setStage2Data(null);
    setError("");
  };

  const handleRunPipeline = async () => {
    setError("");
    setStage1Data(null);
    setStage2Data(null);

    try {
      setStage("yolo_blur_loading");
      let s1;

      if (useMockMode) {
        await new Promise((r) => setTimeout(r, 1800));
        const demoImg = await makeDemoAnnotatedImage(previewUrl);
        s1 = {
          session_id: "demo_01",
          blurred_detected_image: demoImg,
          faces_blurred_count: 2,
          plates_blurred_count: 1,
          detections: [
            { label: "Car", confidence: 0.94, count: 3 },
            { label: "Person", confidence: 0.88, count: 2 },
          ],
        };
      } else {
        const formData = new FormData();
        formData.append("image", imageFile);
        const res1 = await fetch(`${API_BASE_URL}/api/preprocess-detect`, {
          method: "POST",
          body: formData,
        });
        if (!res1.ok) throw new Error("Stage 1 failed in Python backend");
        s1 = await res1.json();
      }

      setStage1Data(s1);
      setStage("vlm_loading");

      let s2;
      if (useMockMode) {
        await new Promise((r) => setTimeout(r, 1800));
        s2 = {
          description: `Scene analysis for "${userPrompt}": The street shows moderate vehicle movement with detected cars and pedestrians. Faces and license plates were blurred before analysis.`,
          pdf_download_url: "#demo-pdf",
        };
      } else {
        const res2 = await fetch(`${API_BASE_URL}/api/analyze-vlm`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            session_id: s1.session_id,
            user_prompt: userPrompt,
          }),
        });
        if (!res2.ok) throw new Error("Stage 2 failed in Python backend");
        s2 = await res2.json();
      }

      setStage2Data(s2);
      setStage("complete");
    } catch (err) {
      setError(err.message);
      setStage("idle");
    }
  };

  const isRunning = stage === "yolo_blur_loading" || stage === "vlm_loading";

  return (
    <div className="home-layout">
      <UploadForm
        imageFile={imageFile}
        previewUrl={previewUrl}
        userPrompt={userPrompt}
        setUserPrompt={setUserPrompt}
        onFileSelect={handleFileSelect}
        onClear={handleClear}
        onRunPipeline={handleRunPipeline}
        disabled={isRunning}
      />

      {error && <div className="error-box">{error}</div>}

      <PipelineStageLoader stage={stage} previewUrl={previewUrl} />

      <AnalysisResult
        stage1Data={stage1Data}
        stage2Data={stage2Data}
        userPrompt={userPrompt}
      />
    </div>
  );
}

export default Home;