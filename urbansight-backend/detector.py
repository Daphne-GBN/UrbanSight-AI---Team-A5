
"""YOLO11n object detection on privacy-processed BGR images."""

from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# YOLO11n pretrained model.
# Ultralytics automatically downloads this file if it is missing.
MODEL_NAME = "yolo11n.pt"

# COCO vehicle classes.
VEHICLE_CLASSES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
}

COLOR = (90, 209, 125)  # BGR
CONFIDENCE_THRESHOLD = 0.25


class YOLO11Detector:

    def __init__(self):
        self.model = None

    def load(self):
        if self.model is None:
            self.model = YOLO(MODEL_NAME)

    def predict(self, image, confidence=CONFIDENCE_THRESHOLD):
        self.load()

        results = self.model.predict(
            source=image,
            conf=confidence,
            imgsz=640,
            device="cpu",
            verbose=False
        )

        detections = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                class_id = int(box.cls.item())

                # Only retain vehicle detections.
                if class_id not in VEHICLE_CLASSES:
                    continue

                score = float(box.conf.item())

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                height, width = image.shape[:2]

                x1 = max(0, min(width, round(x1)))
                x2 = max(0, min(width, round(x2)))
                y1 = max(0, min(height, round(y1)))
                y2 = max(0, min(height, round(y2)))

                if x2 <= x1 or y2 <= y1:
                    continue

                detections.append({
                    "label": VEHICLE_CLASSES[class_id],
                    "confidence": round(score, 3),
                    "bbox": [x1, y1, x2, y2],
                })

        return detections

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

            entry = grouped.setdefault(
                label,
                {
                    "label": label,
                    "confidence": 0,
                    "count": 0,
                    "bboxes": []
                }
            )

            entry["count"] += 1
            entry["confidence"] = max(
                entry["confidence"], score
            )
            entry["bboxes"].append(detection["bbox"])

            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                COLOR,
                2
            )

            caption = f"{label} {score:.0%}"

            text_size = cv2.getTextSize(
                caption,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                1
            )[0]

            badge_top = max(0, y1 - text_size[1] - 8)

            cv2.rectangle(
                annotated,
                (x1, badge_top),
                (
                    min(annotated.shape[1], x1 + text_size[0] + 8),
                    y1
                ),
                COLOR,
                -1
            )

            cv2.putText(
                annotated,
                caption,
                (x1 + 4, max(text_size[1], y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (12, 24, 30),
                1,
                cv2.LINE_AA
            )

        return annotated, list(grouped.values()), counts


# Shared detector instance.
_detector = YOLO11Detector()


def detect_objects(image):
    return _detector.detect_and_annotate(image)
