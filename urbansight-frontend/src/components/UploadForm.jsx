import React, { useRef } from "react";
import { UploadCloud, Sparkles, X } from "lucide-react";

function UploadForm({
  imageFile,
  previewUrl,
  userPrompt,
  setUserPrompt,
  onFileSelect,
  onClear,
  onRunPipeline,
  disabled,
}) {
  const fileInputRef = useRef(null);

  return (
    <section className="card upload-card">
      <div className="section-heading"><span className="step-number">01</span><div><h2>Upload a street scene</h2><p>Choose a clear image to inspect traffic objects and create an annotated preview.</p></div></div>

      {!previewUrl ? (
        <label className="dropzone" htmlFor="streetImage">
          <UploadCloud size={36} />
          <p>Choose a street image</p>
          <small>JPEG, PNG, WebP or AVIF · up to 20 MB</small>
          <input
            ref={fileInputRef}
            id="streetImage"
            type="file"
            accept="image/jpeg,image/png,image/webp,image/avif"
            hidden
            onChange={(e) => e.target.files?.[0] && onFileSelect(e.target.files[0])}
          />
        </label>
      ) : (
        <div className="preview-box">
          <img src={previewUrl} alt="Street preview" className="uploaded-thumb" />
          <div className="preview-footer">
            <span>{imageFile?.name}</span>
            {!disabled && (
              <button type="button" onClick={onClear}>
                <X size={14} /> Remove
              </button>
            )}
          </div>
        </div>
      )}

      <div className="prompt-group">
        <label htmlFor="userPrompt">Inspection focus</label>
        <p className="field-help">The current summary uses YOLOP detections. Visual language analysis is not connected yet.</p>
        <textarea
          id="userPrompt"
          rows={3}
          disabled={disabled}
          value={userPrompt}
          onChange={(e) => setUserPrompt(e.target.value)}
          placeholder="Describe what to check in the street scene (traffic, crossings, road hazards)..."
        />
      </div>

      <button
        type="button"
        className="primary-btn"
        disabled={!previewUrl || !userPrompt.trim() || disabled}
        onClick={onRunPipeline}
      >
        <Sparkles size={16} /> Analyze street scene
      </button>
    </section>
  );
}

export default UploadForm;
