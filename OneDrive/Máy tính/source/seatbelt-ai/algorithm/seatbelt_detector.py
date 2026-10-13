import warnings
from pathlib import Path

import cv2
import numpy as np

from .image_processing import preprocessing_roi
from .feature_extraction import seatbelt_line_extraction


class Seatbelt_Detector:

    def __init__(self, min_confidence=0.5):
        self.min_confidence = min_confidence
        self.model = None
        
        self.model_path = (
            Path(__file__).resolve().parent.parent
            / "models"
            / "seatbelt-seg-v2"
            / "weights"
            / "best_roboflow.pt"
        )


        if self.model_path.is_file():
            from ultralytics import YOLO

            self.model = YOLO(str(self.model_path))
            print("MODEL DANG DUNG:", self.model_path)
        else:
            warnings.warn(
                f"YOLO-seg model not found at {self.model_path}; "
                "using the legacy Hough detector until the model is trained.",
                RuntimeWarning,
                stacklevel=2
            )

    def detect(self, chest_roi):

        if chest_roi is None or chest_roi.size == 0:
            return {
                "seatbelt": False,
                "confidence": 0.0,
                "debug_image": None,
                "polygon": None
            }

        if self.model is not None:
            return self._detect_with_segmentation(chest_roi)
        binary_img, debug_mode = preprocessing_roi(chest_roi)
        lines = seatbelt_line_extraction(binary_img)

        print("Số đường tìm được:", len(lines))

        for line in lines:
            print(
                "angle =", line["angle"],
                "length =", line["length"],
                "coords =", line["coords"]
            )
        debug_img = chest_roi.copy()
        for line in lines:

            x1, y1, x2, y2 = line["coords"]

            cv2.line(
                debug_img,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
            )

        seatbelt_detected = False
        confidence = 0.0
        polygon = None
        if len(lines) >= 2:

            pair = self._find_best_pair(
                lines,
                chest_roi.shape
            )

            if pair is not None:

                line1, line2 = pair
                polygon = self._make_polygon(
                    line1["coords"],
                    line2["coords"]
                )

                seatbelt_detected = True
                cv2.polylines(
                    debug_img,
                    [polygon],
                    isClosed=True,
                    color=(0, 255, 0),
                    thickness=3
                )
                for line in pair:

                    x1, y1, x2, y2 = line["coords"]

                    cv2.line(
                        debug_img,
                        (x1, y1),
                        (x2, y2),
                        (0, 0, 255),
                        3
                    )
                length1 = line1["length"]
                length2 = line2["length"]

                roi_height = chest_roi.shape[0]

                length_score = min(
                    ((length1 + length2) / 2) / roi_height,
                    1.0
                )

                angle_difference = abs(
                    line1["angle"] - line2["angle"]
                )

                parallel_score = max(
                    0.0,
                    1.0 - angle_difference / 20
                )

                confidence = (
                    0.6 * length_score
                    + 0.4 * parallel_score
                )

                confidence = round(
                    confidence,
                    2
                )
        return {
            "seatbelt": (
                seatbelt_detected
                and confidence >= self.min_confidence
            ),

            "confidence": (
                confidence
                if seatbelt_detected
                else 0.0
            ),

            "debug_image": debug_img,

            "polygon": polygon
        }

    def _detect_with_segmentation(self, chest_roi):
        result = self.model(
            chest_roi,
            conf=self.min_confidence,
            verbose=False
        )[0]

        print("CHEST ROI:", chest_roi.shape)
        print(
            "SO VUNG DU DOAN:",
            0 if result.masks is None else len(result.masks)
        )
        print(
            "CONFIDENCE:",
            [] if result.boxes is None
            else [round(float(c), 3) for c in result.boxes.conf]
        )

        debug_img = chest_roi.copy()
        debug_img = chest_roi.copy()

        if result.masks is None or result.boxes is None:
            return {
                "seatbelt": False,
                "confidence": 0.0,
                "debug_image": debug_img,
                "polygon": None
            }

        detections = []

        for polygon_xy, box in zip(result.masks.xy, result.boxes):
            if self.model.names[int(box.cls[0].item())] != "seatbelt":
                continue

            polygon = np.asarray(
                polygon_xy,
                dtype=np.float32
            )

            if len(polygon) < 3:
                continue

            polygon = np.rint(polygon).astype(np.int32)
            confidence = float(box.conf[0].item())
            detections.append((confidence, polygon))

        if not detections:
            return {
                "seatbelt": False,
                "confidence": 0.0,
                "debug_image": debug_img,
                "polygon": None
            }

        confidence, polygon = max(detections, key=lambda item: item[0])

        cv2.polylines(
            debug_img,
            [polygon],
            isClosed=True,
            color=(0, 255, 0),
            thickness=3
        )

        return {
            "seatbelt": confidence >= self.min_confidence,
            "confidence": round(confidence, 2),
            "debug_image": debug_img,
            "polygon": polygon
        }

    def _find_best_pair(self, lines, roi_shape):

        best_pair = None
        best_score = 0.0

        roi_width = roi_shape[1]

        for i in range(len(lines)):

            for j in range(i + 1, len(lines)):

                line1 = lines[i]
                line2 = lines[j]

                # -------------------------
                # So sánh góc
                # -------------------------

                angle1 = line1["angle"]
                angle2 = line2["angle"]

                angle_diff = abs(
                    angle1 - angle2
                )

                if angle_diff > 15:
                    continue

                x1, y1, x2, y2 = line1["coords"]

                a1, b1, a2, b2 = line2["coords"]

                mx1 = (x1 + x2) / 2
                my1 = (y1 + y2) / 2

                mx2 = (a1 + a2) / 2
                my2 = (b1 + b2) / 2

                distance = np.sqrt(
                    (mx1 - mx2) ** 2
                    + (my1 - my2) ** 2
                )

                if distance < roi_width * 0.03:
                    continue

                if distance > roi_width * 0.35:
                    continue

                parallel_score = (
                    1 - angle_diff / 15
                )

                length_score = min(
                    (
                        line1["length"]
                        + line2["length"]
                    ) / (2 * roi_width),
                    1.0
                )

                score = (
                    0.6 * parallel_score
                    + 0.4 * length_score
                )

                if score > best_score:

                    best_score = score

                    best_pair = (
                        line1,
                        line2
                    )

        return best_pair

    def _make_polygon(self, line1, line2):

        x1, y1, x2, y2 = line1

        a1, b1, a2, b2 = line2

        polygon = np.array(
            [
                [x1, y1],
                [x2, y2],
                [a2, b2],
                [a1, b1]
            ],
            dtype=np.int32
        )

        return polygon