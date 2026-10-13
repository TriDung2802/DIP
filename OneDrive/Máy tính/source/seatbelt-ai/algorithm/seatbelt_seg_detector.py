from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "seatbelt-seg-v2"
    / "weights"
    / "best_roboflow.pt"
)

class SeatbeltSegDetector:
    def __init__(self, min_confidence=0.5):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Khong tim thay model: {MODEL_PATH}"
            )

        self.model = YOLO(str(MODEL_PATH))
        self.min_confidence = min_confidence

    def _ket_qua(self, seatbelt, polygon=None, confidence=0.0):
        return {
            "seatbelt": seatbelt,
            "detected": polygon is not None,
            "polygon": polygon,
            "confidence": confidence,
            "debug_image": None,
        }

    def detect(self, person_image):
        if person_image is None or person_image.size == 0:
            return self._ket_qua(False)

        results = self.model.predict(
            source=person_image,
            conf=self.min_confidence,
            retina_masks=True,
            verbose=False,
        )

        result = results[0]

        if result.masks is None or result.boxes is None:
            return self._ket_qua(False)

        if len(result.boxes) == 0:
            return self._ket_qua(False)

        seatbelt_indices = [
            i
            for i, cls in enumerate(result.boxes.cls.tolist())
            if self.model.names[int(cls)] == "seatbelt"
        ]

        if not seatbelt_indices:
            return self._ket_qua(False)

        best_index = max(
            seatbelt_indices,
            key=lambda i: float(result.boxes.conf[i]),
        )

        confidence = float(result.boxes.conf[best_index])
        mask = result.masks.data[best_index].cpu().numpy()

        if mask.shape != person_image.shape[:2]:
            mask = cv2.resize(
                mask,
                (person_image.shape[1], person_image.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        binary_mask = (mask > 0.5).astype(np.uint8)
        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5, 5),
        )
        binary_mask = cv2.morphologyEx(
            binary_mask,
            cv2.MORPH_CLOSE,
            close_kernel,
        )
        smoothed_mask = cv2.GaussianBlur(
            binary_mask.astype(np.float32),
            (5, 5),
            0,
        )
        binary_mask = (smoothed_mask >= 0.45).astype(np.uint8)

        contours, _ = cv2.findContours(
            binary_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            return self._ket_qua(False, confidence=confidence)

        largest_area = max(cv2.contourArea(contour) for contour in contours)
        min_area = max(16.0, largest_area * 0.03)
        polygons = []
        for contour in contours:
            if cv2.contourArea(contour) < min_area:
                continue

            perimeter = cv2.arcLength(contour, closed=True)
            smooth_contour = cv2.approxPolyDP(
                contour,
                epsilon=max(1.0, perimeter * 0.0015),
                closed=True,
            )
            polygon = smooth_contour.reshape(-1, 2)
            if len(polygon) >= 3:
                polygons.append(polygon)

        if not polygons:
            return self._ket_qua(False, confidence=confidence)

        polygons.sort(
            key=lambda polygon: abs(cv2.contourArea(polygon)),
            reverse=True,
        )
        polygon = polygons[0]
        result = self._ket_qua(
            seatbelt=True,
            polygon=polygon,
            confidence=confidence,
        )
        result["polygons"] = polygons
        return result
