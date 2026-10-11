"""
GIẢ LẬP module của Đô (P2: DIP + Seatbelt Detection).
Nhận chest_roi, trả về kết quả theo "hợp đồng":
    {"seatbelt": bool, "confidence": float (0..1), "debug_image": ndarray hoặc None}

Để bạn thấy được cả 2 trạng thái, cứ mỗi 3 giây kết quả đổi qua lại.
"""
import time


def detect_seatbelt(chest_roi):
    co_day = (int(time.time()) // 3) % 2 == 0
    return {
        "seatbelt": co_day,
        "confidence": 0.87 if co_day else 0.62,
        "debug_image": None,
    }
