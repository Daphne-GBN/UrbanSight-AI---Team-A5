"""YOLOP object detection on privacy processed BGR images."""

from pathlib import Path
import sys

import cv2
import numpy as np
import torch

YOLOP_DIR = Path(__file__).resolve().parent.parent / "YOLOP"
WEIGHTS = YOLOP_DIR / "weights" / "End-to-end.pth"
# This checkpoint has nc=1 in YOLOP/lib/models/YOLOP.py: vehicle only.
CLASSES = ("Vehicle",)
COLOR = (90, 209, 125)  # BGR


def letterbox(image, size=640):
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    new_width, new_height = round(width * scale), round(height * scale)
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    left = (size - new_width) // 2
    top = (size - new_height) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[top:top + new_height, left:left + new_width] = resized
    return canvas, scale, left, top


class YOLOPDetector:
    def __init__(self):
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(self):
        if self.model is not None:
            return
        if not WEIGHTS.is_file():
            raise RuntimeError(f"YOLOP weights missing: {WEIGHTS}")
        if str(YOLOP_DIR) not in sys.path:
            sys.path.insert(0, str(YOLOP_DIR))
        from lib.config import cfg
        from lib.models import get_net
        from lib.core.general import non_max_suppression

        model = get_net(cfg)
        checkpoint = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint.get("state_dict", checkpoint))
        self.model = model.to(self.device).eval()
        self.nms = non_max_suppression

    def predict(self, image, confidence=0.25):
        self.load()
        prepared, scale, left, top = letterbox(image)
        rgb = cv2.cvtColor(prepared, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
            [0.229, 0.224, 0.225], dtype=np.float32)
        chw = rgb.transpose(2, 0, 1).copy()
        tensor = torch.frombuffer(bytearray(chw.tobytes()), dtype=torch.float32).reshape(1, 3, 640, 640).to(self.device)
        with torch.inference_mode():
            detection_head, _, _ = self.model(tensor)
            predictions = detection_head[0] if isinstance(detection_head, (tuple, list)) else detection_head
            boxes = self.nms(predictions, conf_thres=confidence, iou_thres=0.45)[0]

        height, width = image.shape[:2]
        results = []
        for x1, y1, x2, y2, score, class_id in boxes.cpu().tolist():
            coords = [max(0, min(width, round((x - left) / scale))) for x in (x1, x2)]
            rows = [max(0, min(height, round((y - top) / scale))) for y in (y1, y2)]
            if coords[1] <= coords[0] or rows[1] <= rows[0]:
                continue
            index = int(class_id)
            results.append({
                "label": CLASSES[index] if 0 <= index < len(CLASSES) else f"Class {index}",
                "confidence": round(score, 3),
                "bbox": [coords[0], rows[0], coords[1], rows[1]],
            })
        return results

    def detect_and_annotate(self, image):
        raw = self.predict(image)
        annotated = image.copy()
        grouped = {}
        counts = {}
        for detection in raw:
            label = detection["label"]
            score = detection["confidence"]
            x1, y1, x2, y2 = detection["bbox"]
            counts[label] = counts.get(label, 0) + 1
            entry = grouped.setdefault(label, {"label": label, "confidence": 0, "count": 0, "bboxes": []})
            entry["count"] += 1
            entry["confidence"] = max(entry["confidence"], score)
            entry["bboxes"].append(detection["bbox"])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), COLOR, 2)
            caption = f"{label} {score:.0%}"
            text_size = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            badge_top = max(0, y1 - text_size[1] - 8)
            cv2.rectangle(annotated, (x1, badge_top), (min(annotated.shape[1], x1 + text_size[0] + 8), y1), COLOR, -1)
            cv2.putText(annotated, caption, (x1 + 4, max(text_size[1], y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (12, 24, 30), 1, cv2.LINE_AA)
        return annotated, list(grouped.values()), counts


_detector = YOLOPDetector()


def detect_objects(image):
    return _detector.detect_and_annotate(image)
