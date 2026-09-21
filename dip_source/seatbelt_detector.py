import cv2

import numpy as np

from dip_and_seatbelt_detection.dip_source.image_processing import preprocessing_roi

from dip_and_seatbelt_detection.dip_source.feature_extraction import seatbelt_line_extraction



class Seatbelt_Detector:

    def __init__(self, min_confidence=0.5):

        self.min_confidence=min_confidence



    def detect(self, chest_roi):



        if chest_roi is None or chest_roi.size == 0:

            return {"seatbelt": False, "confidence": 0.0, "debug_image": None}



        binary_img, debug_mode = preprocessing_roi(chest_roi)





        lines = seatbelt_line_extraction(binary_img)



        debug_img = chest_roi.copy()

        seatbelt_detected = False

        confidence = 0.0



        if len(lines) > 0:

            seatbelt_detected = True

           

            for line in lines:

                x1, y1, x2, y2 = line["coords"]

                cv2.line(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

           

            max_len = max([l["length"] for l in lines])

            roi_height = chest_roi.shape[0]

           

            confidence = min(1.0, (max_len / roi_height) * 0.8 + len(lines) * 0.1)

            confidence = round(confidence, 2)



        return {

            "seatbelt": seatbelt_detected and (confidence >= self.min_confidence),

            "confidence": confidence if seatbelt_detected else 0.0,

            "debug_image": debug_img

        }



if __name__ == "__main__":

    detector = Seatbelt_Detector()

    # test_roi = cv2.imread("path_to_test_roi.jpg")

    # result = detector.detect(test_roi)

    # print(result) //seatbelt_detector.py