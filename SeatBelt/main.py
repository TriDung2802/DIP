import tkinter as tk
from collections import deque

import cv2

from trial_p1 import detect_people      
from trial_p2 import detect_seatbelt    
from logger import Logger
from ui import SeatbeltUI

SO_KHUNG_LAM_MUOT = 7      
lich_su = {}               


def lam_muot(i, result):
    """Một khung hình nhận sai không được làm đổi kết luận: lấy đa số của vài khung gần nhất."""
    hs = lich_su.setdefault(i, deque(maxlen=SO_KHUNG_LAM_MUOT))
    hs.append((result["seatbelt"], result["confidence"]))
    co_day = sum(1 for c, _ in hs if c) * 2 > len(hs)         # hòa -> coi là chưa chắc có dây
    dong_y = [tc for c, tc in hs if c == co_day]
    moi = dict(result)
    moi["seatbelt"] = co_day
    moi["confidence"] = sum(dong_y) / len(dong_y)
    return moi


def pipeline(frame):
    """frame -> danh sách (person, result). Đây là toàn bộ 'luồng integration'."""
    cac_nguoi = detect_people(frame)                          # gọi P1
    if not cac_nguoi:
        lich_su.clear()                                       # không còn ai -> quên lịch sử
    ket_qua = []
    for i, person in enumerate(cac_nguoi):
        result = detect_seatbelt(person["chest_roi"])         # gọi P2
        ket_qua.append((person, lam_muot(i, result)))
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
