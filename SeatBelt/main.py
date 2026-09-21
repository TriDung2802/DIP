"""
main.py chỉ làm 4 việc: mở camera, định nghĩa pipeline, tạo logger, mở giao diện.
Chỉ 2 dòng import mock_* là thứ cần đổi khi P1, P2 làm xong.
"""
import tkinter as tk

import cv2

from mock_p1 import detect_people      # sau này: from roi import detect_people (tên do Dũng báo)
from mock_p2 import detect_seatbelt    # sau này: from seatbelt_detector import detect_seatbelt
from logger import Logger
from ui import SeatbeltUI


def pipeline(frame):
    """frame -> danh sách (person, result). Đây là toàn bộ 'luồng integration'."""
    ket_qua = []
    for person in detect_people(frame):                 # gọi P1
        result = detect_seatbelt(person["chest_roi"])   # gọi P2
        ket_qua.append((person, result))
    return ket_qua


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không mở được camera. Thử đổi số 0 thành 1.")
        return

    logger = Logger()
    root = tk.Tk()
    SeatbeltUI(root, cap, pipeline, khi_co_su_kien=logger.ghi)
    root.mainloop()


if __name__ == "__main__":
    main()