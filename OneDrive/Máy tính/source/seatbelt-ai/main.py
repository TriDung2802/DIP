import tkinter as tk

from person_detection.camera import open_camera
from person_detection.person_detector import detect_people
from person_detection.dataset_saver import (
    save_seatbelt_image,
    save_no_seatbelt_image,
)

from algorithm.seatbelt_seg_detector import SeatbeltSegDetector
from UI.logger import Logger
from UI.ui import SeatbeltUI

detector = SeatbeltSegDetector(min_confidence=0.5)

def pipeline(frame):
    persons = detect_people(frame)
    # không phát hiện người -> không nhận diện dây
    if not persons:
        return[]
    results = []

    for person in persons:
        person_image = person.get("person_image")
        if person_image is None or person_image.size ==0:
            continue
        result = detector.detect(person["person_image"])
        results.append((person, result))

    return results

def main():
    cap = open_camera()
    logger = Logger()
    root = tk.Tk()

    SeatbeltUI(
        root,
        cap,
        pipeline,
        khi_co_su_kien=logger.ghi,
        luu_co_day=save_seatbelt_image,
        luu_khong_day=save_no_seatbelt_image,
    )
    root.mainloop()
if __name__ == "__main__":
    main()
