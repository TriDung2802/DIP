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


detector = SeatbeltSegDetector(min_confidence=0.08)


def pipeline(frame):
    persons = detect_people(frame)
    results = []

    for person in persons:
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
