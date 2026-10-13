"""
Mỗi vi phạm tạo ra:
  - 1 dòng trong  logs/vi_pham.csv   (mở được bằng Excel)
  - 1 ảnh chụp    logs/anh/vi_pham_YYYYMMDD_HHMMSS.jpg

NGUYÊN TẮC VÀNG: ghi log hỏng thì hệ thống vẫn phải chạy tiếp.
"""
import csv
from datetime import datetime
from pathlib import Path

import cv2

THU_MUC_LOG = Path(__file__).resolve().parent / "logs"   # cạnh file logger.py
THU_MUC_ANH = THU_MUC_LOG / "anh"
FILE_CSV = THU_MUC_LOG / "vi_pham.csv"
CAC_COT = ["thoi_gian", "camera", "loai_vi_pham", "do_tin_cay", "file_anh"]


class Logger:
    def __init__(self):
        THU_MUC_ANH.mkdir(parents=True, exist_ok=True)   # chưa có thư mục thì tự tạo

    def ghi(self, su_kien):
        """Nhận dict sự kiện từ ui.py, ghi ảnh + 1 dòng CSV. Không bao giờ làm sập chương trình."""
        try:
            ten_anh = self._luu_anh(su_kien)
            self._ghi_dong_csv(su_kien, ten_anh)
        except Exception as loi:
            print(f"[Logger] Không ghi được log: {loi}")

    def _luu_anh(self, sk):
        moc = datetime.strptime(sk["thoi_gian"], "%d/%m/%Y %H:%M:%S")
        ten = f"vi_pham_{moc:%Y%m%d_%H%M%S}.jpg"
        ok, bo_dem = cv2.imencode(".jpg", sk["anh"])
        if not ok:
            raise RuntimeError("Không mã hóa được ảnh")
        (THU_MUC_ANH / ten).write_bytes(bo_dem.tobytes())
        return ten

    def _ghi_dong_csv(self, sk, ten_anh):
        file_moi = not FILE_CSV.exists()
        # utf-8-sig: để Excel đọc đúng tiếng Việt có dấu
        with open(FILE_CSV, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            if file_moi:
                w.writerow(CAC_COT)                       # dòng tiêu đề, chỉ ghi 1 lần
            w.writerow([sk["thoi_gian"], sk["camera"], sk["loai"],
                        f"{sk['tin_cay']:.4f}", ten_anh])