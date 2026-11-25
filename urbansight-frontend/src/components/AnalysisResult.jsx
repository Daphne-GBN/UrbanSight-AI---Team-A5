import React from "react";
import { Download, ShieldCheck, Tag, FileText } from "lucide-react";
import { jsPDF } from "jspdf";

function AnalysisResult({ stage1Data, stage2Data, userPrompt }) {
  if (!stage1Data) return null;

  const handleDownloadPdf = () => {
    if (stage2Data?.pdf_download_url && stage2Data.pdf_download_url !== "#demo-pdf") {
      window.open(stage2Data.pdf_download_url, "_blank");
      return;
    }

    const doc = new jsPDF();
    doc.setFontSize(16);
    doc.text("UrbanSight AI - Street Scene Report", 14, 20);

    doc.setFontSize(11);
    doc.text(`Focus Prompt: ${userPrompt}`, 14, 32);

    const detText = (stage1Data.detections || [])
      .map((d) => `${d.label} (${d.count || 1})`)
      .join(", ");
    doc.text(`YOLO Detections: ${detText || "None"}`, 14, 42);

    doc.text("Qwen2.5-VL Scene Description:", 14, 56);
    const wrapped = doc.splitTextToSize(stage2Data?.description || "", 180);
    doc.text(wrapped, 14, 64);

    doc.save("UrbanSight_Report.pdf");
  };

  return (
    <div className="results-stack">
      <section className="card result-card">
        <div className="card-top">
          <h2>
            <ShieldCheck size={18} /> Stage 1: Blurred Image & YOLO Detections
          </h2>
          <span>
            {stage1Data.faces_blurred_count ?? 0} Faces Blurred ·{" "}
            {stage1Data.plates_blurred_count ?? 0} Plates Blurred
          </span>
        </div>

        <img
          src={stage1Data.blurred_detected_image}
          alt="Blurred and YOLO annotated"
          className="result-image"
        />

        <div className="tags-row">
          <Tag size={15} />
          {(stage1Data.detections || []).map((item, index) => (
            <span key={index} className="det-tag">
              {item.label} {item.count ? `x${item.count}` : ""} ({Math.round(item.confidence * 100)}%)
            </span>
          ))}
        </div>
      </section>

      {stage2Data?.description && (
        <section className="card result-card">
          <div className="card-top">
            <h2>
              <FileText size={18} /> Stage 2: Qwen2.5-VL Description & PDF
            </h2>
            <button type="button" className="pdf-btn" onClick={handleDownloadPdf}>
              <Download size={15} /> Download PDF
            </button>
          </div>

          <p className="vlm-description">{stage2Data.description}</p>
        </section>
      )}
    </div>
  );
}

export default AnalysisResult;