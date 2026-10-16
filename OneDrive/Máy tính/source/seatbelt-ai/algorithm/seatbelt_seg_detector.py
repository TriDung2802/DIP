from pathlib import Path
from collections import deque
import cv2
import numpy as np
import math
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (BASE_DIR/ "models"/ "seatbelt-seg-v2"/ "weights"/ "best_roboflow.pt")

class SeatbeltSegDetector:
    def __init__(self, min_confidence=0.5):
        self.history = deque(maxlen=10)
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Khong tim thay model: {MODEL_PATH}"
            )

        self.model = YOLO(str(MODEL_PATH))
        print("Model dang dung:", MODEL_PATH)
        print("Danh sach class:", self.model.names)
        self.min_confidence = min_confidence
    
    def _ket_qua(self, seatbelt, polygon=None, confidence=0.0):
        return {
            "seatbelt": seatbelt,
            "detected": polygon is not None,
            "polygon": polygon,
            "confidence": confidence,
            "debug_image": None,
        }
    
    def _mask_co_hinh_dang_hop_ly(self, contour, image_shape):
        h, w = image_shape[:2]
        area = cv2.contourArea(contour)
        if area < max(20.0, h * w * 0.0005):
            print("[LOC] loai: dien tich nho", area)
            return False

        (_, _), (rw, rh), _ = cv2.minAreaRect(contour)
        long_side, short_side = max(rw, rh), max(1.0, min(rw, rh))
        ratio = long_side / short_side

        vx, vy, _, _ = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        angle = abs(math.degrees(math.atan2(vy, vx)))
        if angle > 90:
            angle = 180 - angle

        print(f"[LOC] ratio={ratio:.2f} long={long_side:.0f}/{h} angle={angle:.0f}")

        if ratio < 1.8:
            return False
        if long_side < h * 0.15:
            return False
        return 15 <= angle <= 80

    def detect(self, person_image):
        if person_image is None or person_image.size == 0:
            return self._lam_muot(False)
        
        H = person_image.shape[0]
        y_off = 0
        roi = person_image
        if roi.size == 0:
            return self._lam_muot(False)

        results = self.model.predict(
            source=roi,                      # đưa roi vào model
            conf=self.min_confidence,
            retina_masks=True,
            verbose=False,
        )
        result = results[0]

        if result.masks is None or result.boxes is None or len(result.boxes) == 0:
            return self._lam_muot(False)

        idx = [i for i, c in enumerate(result.boxes.cls.tolist())
            if self.model.names[int(c)] == "seatbelt"]
        if not idx:
            return self._lam_muot(False)

        best = max(idx, key=lambda i: float(result.boxes.conf[i]))
        confidence = float(result.boxes.conf[best])
        mask = result.masks.data[best].cpu().numpy()

        # resize về ĐÚNG kích thước roi (không phải person_image)
        if mask.shape != roi.shape[:2]:
            mask = cv2.resize(mask, (roi.shape[1], roi.shape[0]),
                            interpolation=cv2.INTER_LINEAR)

        binary = (mask > 0.5).astype(np.uint8)
        binary = cv2.morphologyEx(
            binary, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        smooth = cv2.GaussianBlur(binary.astype(np.float32), (5, 5), 0)
        binary = (smooth >= 0.45).astype(np.uint8)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return self._lam_muot(False, confidence=confidence)

        largest = max(cv2.contourArea(c) for c in contours)
        min_area = max(16.0, largest * 0.03)
        polygons = []
        for c in contours:
            if cv2.contourArea(c) < min_area:
                continue
            if not self._mask_co_hinh_dang_hop_ly(c, roi.shape):   # roi.shape
                continue
            eps = max(1.0, cv2.arcLength(c, True) * 0.0015)
            poly = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
            if len(poly) >= 3:
                polygons.append(poly + np.array([0, y_off]))       # cộng offset 1 lần duy nhất

        if not polygons:
            return self._lam_muot(False, confidence=confidence)

        polygons.sort(key=lambda p: abs(cv2.contourArea(p.astype(np.int32))),
                    reverse=True)
        return self._lam_muot(True, polygons, confidence)

    def _lam_muot(self, detected, polygons=None, confidence=0.0):
        self.history.append(1 if detected else 0)
        stable = len(self.history) < 5 or sum(self.history) >= 4

        if detected and stable:
            res = self._ket_qua(True, polygons[0], confidence)
            res["polygons"] = polygons
            return res
        return self._ket_qua(False, confidence=confidence)