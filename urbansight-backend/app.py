"""
UrbanSight AI - Backend Server (FastAPI)
Architecture PS 112:
- Stage 2: OpenCV image decoding & preprocessing
- Stage 3: Privacy blurring for human faces and vehicle license plates (privacy_blur.py)
- Object Detection: YOLOP (You Only Look Once for Panoptic Driving Perception) bounding boxes (detector.py)
- PDF Generation: ReportLab inspection report export (pdf_generator.py)
- Scene Analysis: Factual summary module (vlm_analyzer.py)
"""

import uuid
import base64
import io
import logging
import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

from privacy_blur import anonymize_image
from detector import detect_objects
from vlm_analyzer import generate_scene_summary
from pdf_generator import generate_report_pdf


app = FastAPI(
    title="UrbanSight AI Backend",
    description="Privacy-First Street Scene Analysis with YOLOP Bounding Box Prediction",
    version="1.0.0"
)
logger = logging.getLogger(__name__)

# Enable CORS for the React/Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session cache for processed street analyses
SESSIONS: dict[str, dict] = {}


class VlmAnalysisRequest(BaseModel):
    session_id: str
    user_prompt: str = "Describe traffic flow, road conditions, and pedestrian safety."


@app.get("/")
def health_check():
    return {
        "service": "UrbanSight AI Backend",
        "status": "online",
        "detector_model": "YOLOP (Panoptic Driving Perception)",
        "privacy_blur": "Best-effort face and plate blurring",
        "qwen_vlm_stage": "Not connected; count-based summary only"
    }


@app.post("/api/preprocess-detect")
async def preprocess_and_detect(image: UploadFile = File(...)):
    """
    Stage 1 to Stage 3 Pipeline:
    1. Validates and decodes image.
    2. Applies best-effort blur to detected faces and plates.
    3. Runs YOLOP object detection on the blurred image and draws bounding boxes.
    4. Caches session data and returns base64 annotated preview for frontend.
    """
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a valid image format.")

    try:
        contents = await image.read()
        if not contents or len(contents) > 20 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Upload an image smaller than 20 MB.")
        nparr = np.frombuffer(contents, np.uint8)
        cv_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if cv_img is None:
            try:
                with Image.open(io.BytesIO(contents)) as pil_image:
                    pil_image = ImageOps.exif_transpose(pil_image).convert("RGB")
                    cv_img = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                raise HTTPException(status_code=400, detail="Could not decode this image. Try JPEG or PNG, or install Pillow with AVIF support.") from exc

        # Stage 3: Privacy Blurring (Faces + Vehicle Plates only)
        blurred_img, faces_blurred, plates_blurred, blurred_regions = anonymize_image(cv_img)

        # Object Detection: YOLOP model predicts bounding boxes
        annotated_img, detections_list, counts_summary = detect_objects(blurred_img)

        # Encode annotated image to Base64 Data URL for immediate display
        success, encoded_buf = cv2.imencode(".jpg", annotated_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not success:
            raise HTTPException(status_code=500, detail="Failed to encode annotated preview.")

        b64_image = base64.b64encode(encoded_buf.tobytes()).decode("utf-8")
        data_url = f"data:image/jpeg;base64,{b64_image}"

        # Store session context
        session_id = f"urbansight_{uuid.uuid4().hex[:8]}"
        SESSIONS[session_id] = {
            "faces_blurred": faces_blurred,
            "plates_blurred": plates_blurred,
            "blurred_regions": blurred_regions,
            "detections": detections_list,
            "detection_summary": counts_summary,
            "annotated_bytes": encoded_buf.tobytes(),
            "user_prompt": ""
        }

        return {
            "session_id": session_id,
            "blurred_detected_image": data_url,
            "faces_blurred_count": faces_blurred,
            "plates_blurred_count": plates_blurred,
            "detections": detections_list,
            "detection_summary": counts_summary,
            "privacy_status": "Best-effort blur applied to detected faces and plates; review before sharing"
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Image pipeline failed")
        raise HTTPException(status_code=500, detail=f"Image pipeline failed: {exc}") from exc


@app.post("/api/analyze-vlm")
def analyze_vlm_stage(payload: VlmAnalysisRequest):
    """
    Stage 4 / VLM Analysis step:
    Synthesizes YOLOP observations with the user's focus prompt.
    Keeps frontend buttons active and provides the PDF download link.
    """
    session = SESSIONS.get(payload.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Analysis session expired. Upload the image again.")

    session["user_prompt"] = payload.user_prompt
    summary = generate_scene_summary(
        user_prompt=payload.user_prompt,
        detection_summary=session.get("detection_summary", {}),
        faces_blurred=session.get("faces_blurred", 0),
        plates_blurred=session.get("plates_blurred", 0)
    )
    session["vlm_description"] = summary["description"]

    return {
        "session_id": payload.session_id,
        "description": summary["description"],
        "pdf_download_url": f"http://localhost:8000/api/download-report/{payload.session_id}",
        "traffic_density": summary["traffic_density"],
        "limitations": summary["limitations"]
    }


@app.get("/api/download-report/{session_id}")
def download_pdf_report(session_id: str):
    """
    Stage 6: Generates and serves a downloadable PDF report.
    """
    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Analysis session expired. Upload the image again.")
    user_prompt = session.get("user_prompt", "Street scene safety and density inspection.")
    detection_summary = session.get("detection_summary", {})
    faces_blurred = session.get("faces_blurred", 0)
    plates_blurred = session.get("plates_blurred", 0)
    description = session.get(
        "vlm_description",
        "Street inspection conducted with privacy anonymization and YOLOP bounding box detection."
    )

    pdf_bytes = generate_report_pdf(
        session_id=session_id,
        user_prompt=user_prompt,
        detection_summary=detection_summary,
        faces_blurred=faces_blurred,
        plates_blurred=plates_blurred,
        description=description
    )

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename=UrbanSight_Report_{session_id}.pdf"
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
