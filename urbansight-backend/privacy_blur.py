"""
UrbanSight AI - Privacy Blurring Module (Stage 3)
Implements best-effort privacy preprocessing:
- Blurs human faces using Haar Cascades.
- Blurs vehicle number plates ONLY (never the entire car).
- Blurs detected regions before image data reaches object detection.
  Automatic detection can miss small, angled, or obscured details.
"""

import os
import cv2
import numpy as np


class PrivacyBlurEngine:
    def __init__(self):
        # Load OpenCV built-in cascades
        cascade_dir = cv2.data.haarcascades
        face_cascade_path = os.path.join(cascade_dir, "haarcascade_frontalface_default.xml")
        plate_cascade_path = os.path.join(cascade_dir, "haarcascade_russian_plate_number.xml")

        self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
        self.plate_cascade = cv2.CascadeClassifier(plate_cascade_path)

        # Fallback profile face cascade for angled pedestrians
        profile_path = os.path.join(cascade_dir, "haarcascade_profileface.xml")
        self.profile_cascade = cv2.CascadeClassifier(profile_path) if os.path.exists(profile_path) else None

    def _apply_strong_blur(self, image: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
        """Applies a heavy Gaussian blur to a specific region of interest."""
        h_img, w_img = image.shape[:2]
        # Boundary constraints
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w_img, x + w)
        y2 = min(h_img, y + h)

        if x2 <= x1 or y2 <= y1:
            return image

        roi = image[y1:y2, x1:x2]

        # Calculate kernel size based on ROI dimensions (must be odd)
        k_w = max(15, (roi.shape[1] // 2) * 2 + 1)
        k_h = max(15, (roi.shape[0] // 2) * 2 + 1)

        # Double-pass heavy blur for guaranteed privacy compliance
        blurred_roi = cv2.GaussianBlur(roi, (k_w, k_h), 30)
        blurred_roi = cv2.GaussianBlur(blurred_roi, (k_w, k_h), 30)

        image[y1:y2, x1:x2] = blurred_roi
        return image

    def _detect_plate_contours(self, gray: np.ndarray) -> list:
        """
        Morphological license plate region extraction to catch plates
        that may be missed by Haar cascades.
        Plates typically have aspect ratios between 2.0 and 5.5 and horizontal edge gradients.
        """
        plate_boxes = []
        try:
            # Sobel horizontal gradient
            sobelx = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
            _, thresh = cv2.threshold(sobelx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Morphological close to join characters of plate
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
            closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            h_img, w_img = gray.shape

            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w) / max(1, h)
                area = w * h

                # Typical street photograph license plate dimensions
                if 2.0 <= aspect_ratio <= 5.8 and (0.0003 * w_img * h_img <= area <= 0.05 * w_img * h_img):
                    # Plate is usually located on the road / vehicle plane, not in the upper sky
                    if y > h_img * 0.25:
                        plate_boxes.append((x, y, w, h))
        except Exception:
            pass
        return plate_boxes

    def anonymize_image(self, image: np.ndarray) -> tuple[np.ndarray, int, int, list]:
        """
        Processes the input BGR image:
        - Detects human faces and applies Gaussian blur.
        - Detects license plates and blurs ONLY the plate region (not the car).

        Returns:
            processed_image: Anonymized BGR image.
            faces_blurred_count: Number of faces anonymized.
            plates_blurred_count: Number of plates anonymized.
            blurred_regions: List of dicts describing blurred bounding boxes.
        """
        output_image = image.copy()
        gray = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)
        gray_eq = cv2.equalizeHist(gray)

        blurred_regions = []

        # 1. Detect and blur human faces
        faces = self.face_cascade.detectMultiScale(
            gray_eq, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20)
        )
        face_list = list(faces) if len(faces) > 0 else []

        if self.profile_cascade is not None:
            profiles = self.profile_cascade.detectMultiScale(
                gray_eq, scaleFactor=1.15, minNeighbors=4, minSize=(20, 20)
            )
            if len(profiles) > 0:
                face_list.extend(list(profiles))

        # Deduplicate and apply blur to faces
        unique_faces = []
        for (fx, fy, fw, fh) in face_list:
            # Check overlap with existing faces
            is_dup = False
            for (ux, uy, uw, uh) in unique_faces:
                if abs(fx - ux) < 20 and abs(fy - uy) < 20:
                    is_dup = True
                    break
            if not is_dup:
                unique_faces.append((fx, fy, fw, fh))
                output_image = self._apply_strong_blur(output_image, fx, fy, fw, fh)
                blurred_regions.append({"type": "face", "box": [int(fx), int(fy), int(fw), int(fh)]})

        # 2. Detect and blur vehicle number plates ONLY
        plates = self.plate_cascade.detectMultiScale(
            gray_eq, scaleFactor=1.1, minNeighbors=3, minSize=(25, 10)
        )
        plate_list = list(plates) if len(plates) > 0 else []

        # Supplementary contour-based plate detection
        contour_plates = self._detect_plate_contours(gray)
        plate_list.extend(contour_plates)

        unique_plates = []
        for (px, py, pw, ph) in plate_list:
            # Ensure it is not overlapping an already blurred face
            is_face = any(abs(px - fx) < 25 and abs(py - fy) < 25 for (fx, fy, _, _) in unique_faces)
            if is_face:
                continue

            # Deduplicate plate boxes
            is_dup = False
            for (ux, uy, uw, uh) in unique_plates:
                if abs(px - ux) < 20 and abs(py - uy) < 20:
                    is_dup = True
                    break
            if not is_dup:
                unique_plates.append((px, py, pw, ph))
                # Blur ONLY the number plate region, never the full car
                output_image = self._apply_strong_blur(output_image, px, py, pw, ph)
                blurred_regions.append({"type": "plate", "box": [int(px), int(py), int(pw), int(ph)]})

        return output_image, len(unique_faces), len(unique_plates), blurred_regions


# Global single-instance helper
_engine = PrivacyBlurEngine()

def anonymize_image(image: np.ndarray) -> tuple[np.ndarray, int, int, list]:
    """Public helper function adhering to architectural interface."""
    return _engine.anonymize_image(image)
