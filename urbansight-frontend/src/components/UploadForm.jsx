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
      <h2>1. Upload Street Image & Focus Prompt</h2>

      {!previewUrl ? (
        <div
          className="dropzone"
          onClick={() => !disabled && fileInputRef.current?.click()}
        >
          <UploadCloud size={36} />
          <p>Click to select a street image</p>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            hidden
            onChange={(e) => e.target.files?.[0] && onFileSelect(e.target.files[0])}
          />
        </div>
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
        <label htmlFor="userPrompt">Specific Description / Focus for Qwen2.5-VL</label>
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
        <Sparkles size={16} /> Run Privacy Blur, YOLO & Qwen Analysis
      </button>
    </section>
  );
}

export default UploadForm;