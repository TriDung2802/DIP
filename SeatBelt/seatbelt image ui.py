"""Chế độ CHỌN ẢNH: kiểm tra thắt dây an toàn trên 1 ảnh có sẵn (không cần camera).

Bên trái : ảnh đã chọn (có khung người, khung ngực).
Bên phải : THẺ KẾT QUẢ (giống thẻ cảnh báo bên camera)
           -> nút CHỌN ẢNH, XEM CHI TIẾT, QUAY LẠI CAMERA.

Chạy riêng được:  python seatbelt_image_ui.py
Hoặc mở từ nút "CHỌN ẢNH" bên màn hình camera (ui.py).
"""
import subprocess
import sys
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

from trial_p1_yolo import detect_people   # cùng P1 với màn hình camera
from trial_p2 import detect_seatbelt      # cùng P2 với màn hình camera
from ui import (DO, FONT, MAU_DO, MAU_XAM, MAU_XANH, VAI_TRO, VANG, XANH,
                SeatbeltUI, thanh_photoimage)

THU_MUC = Path(__file__).resolve().parent
THU_MUC_MO_DAU = THU_MUC / "test"           # hộp thoại chọn ảnh mở sẵn ở thư mục test/
TEN_NGUON = "Ảnh tải lên"                   # thay cho "CAM-01"
KHUNG_TOI_DA = (640, 480)                   # ảnh hiển thị bên trái không vượt quá kích thước này
LOAI_FILE = [("Ảnh", "*.jpg *.jpeg *.png *.bmp *.webp"), ("Tất cả", "*.*")]


def doc_anh(duong_dan):
    """cv2.imread hay lỗi với đường dẫn có dấu tiếng Việt trên Windows -> đọc bytes rồi giải mã."""
    du_lieu = np.fromfile(str(duong_dan), dtype=np.uint8)
    return cv2.imdecode(du_lieu, cv2.IMREAD_COLOR)


def thu_vua_khung(anh, rong_max, cao_max):
    """Ảnh OpenCV -> ảnh Tkinter, thu nhỏ để vừa khung (giữ tỉ lệ, không phóng to)."""
    h, w = anh.shape[:2]
    ti_le = min(rong_max / w, cao_max / h, 1.0)
    nho = cv2.resize(anh, (max(1, int(w * ti_le)), max(1, int(h * ti_le))))
    return ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(nho, cv2.COLOR_BGR2RGB)))


class SeatbeltImageUI:
    def __init__(self, root):
        self.root = root
        self.su_kien = None                 # kết quả của ảnh đang xem (để XEM CHI TIẾT dùng)
        self.thu_muc_gan_nhat = THU_MUC_MO_DAU if THU_MUC_MO_DAU.exists() else THU_MUC

        root.title("Seatbelt AI - Chọn ảnh")
        root.protocol("WM_DELETE_WINDOW", self.dong)

        SeatbeltUI.dung_header(self, root)  # dùng lại thanh tiêu đề của màn hình camera

        # --- Bên trái: ảnh ---
        rong, cao = KHUNG_TOI_DA
        self.anh_cho = ImageTk.PhotoImage(Image.new("RGB", (rong, cao), "#2b2b2b"))
        self.khung_hinh = tk.Label(root, image=self.anh_cho, compound="center", fg="white",
                                   font=(FONT, 12), text="Bấm  CHỌN ẢNH  để bắt đầu")
        self.khung_hinh.pack(side="left", padx=10, pady=10)

        # --- Bên phải: tên file + thẻ kết quả ---
        cot_phai = tk.Frame(root)
        cot_phai.pack(side="right", fill="y", padx=10, pady=10)

        self.nhan_trang_thai = tk.Label(cot_phai, text="● Chưa chọn ảnh",
                                        font=(FONT, 11, "bold"), anchor="w", fg=MAU_XAM)
        self.nhan_trang_thai.pack(fill="x", pady=(0, 6))

        self.the = tk.Frame(cot_phai, highlightthickness=2, highlightbackground=MAU_XAM)
        self.the.pack(fill="both", expand=True)
        self.dung_the()
        self.hien_the_rong()

        root.bind("<Control-o>", lambda e: self.chon_anh())

    # ================= dựng thẻ kết quả (1 lần) =================
    def dung_the(self):
        the = self.the
        kw = dict(anchor="w", font=(FONT, 11))

        self.nhan_tieu_de = tk.Label(the, font=(FONT, 13, "bold"), anchor="w")
        self.nhan_tieu_de.pack(fill="x", padx=12, pady=(10, 6))

        self.nhan_file = tk.Label(the, font=(FONT, 10), anchor="w", fg=MAU_XAM,
                                  wraplength=270, justify="left")
        self.nhan_file.pack(fill="x", padx=12)

        o = tk.Frame(the, highlightthickness=1, highlightbackground="gray")
        o.pack(anchor="w", padx=12, pady=8)
        self.nhan_vai_tro = tk.Label(o, font=(FONT, 11), anchor="w", width=24)
        self.nhan_vai_tro.pack(padx=6, pady=(4, 0))
        self.nhan_hanh_vi = tk.Label(o, font=(FONT, 11), anchor="w", width=24)
        self.nhan_hanh_vi.pack(padx=6, pady=(0, 4))

        self.nhan_so_nguoi = tk.Label(the, **kw)
        self.nhan_loai = tk.Label(the, **kw)
        self.nhan_tin_cay = tk.Label(the, **kw)
        self.nhan_thoi_gian = tk.Label(the, **kw)
        for nhan in (self.nhan_so_nguoi, self.nhan_loai, self.nhan_tin_cay, self.nhan_thoi_gian):
            nhan.pack(fill="x", padx=12, pady=1)

        # Hàng nút
        kieu_nut = dict(font=(FONT, 11, "bold"), width=13, pady=4, cursor="hand2")
        hang_nut = tk.Frame(the)
        hang_nut.pack(fill="x", padx=12, pady=(12, 4))

        self.nut_chon = tk.Button(hang_nut, text="CHỌN ẢNH", command=self.chon_anh,
                                  bg=MAU_XANH, fg="white", activebackground="#167a2c",
                                  activeforeground="white", **kieu_nut)
        self.nut_chon.pack(side="left", expand=True, padx=(0, 4))

        self.nut = tk.Button(hang_nut, text="XEM CHI TIẾT", command=self.xem_chi_tiet,
                             bg=MAU_DO, fg="white", activebackground="#a61e1e",
                             activeforeground="white", disabledforeground="#e8b4b4",
                             **kieu_nut)
        self.nut.pack(side="left", expand=True, padx=(4, 0))

        tk.Button(the, text="◀ QUAY LẠI CAMERA", command=self.ve_camera,
                  font=(FONT, 10), cursor="hand2").pack(fill="x", padx=12, pady=(4, 12))

    # ================= các trạng thái của thẻ =================
    def hien_the_rong(self):
        self.the.configure(highlightbackground=MAU_XAM)
        self.nhan_tieu_de.configure(text="Chưa có kết quả", fg=MAU_XAM)
        self.nhan_file.configure(text="Chưa chọn ảnh")
        self.nhan_vai_tro.configure(text="—")
        self.nhan_hanh_vi.configure(text="—")
        self.nhan_so_nguoi.configure(text="Số người: —")
        self.nhan_loai.configure(text="Loại vi phạm: —")
        self.nhan_tin_cay.configure(text="Độ tin cậy: —")
        self.nhan_thoi_gian.configure(text="Thời gian: —")
        self.nut.configure(state="disabled")

    def hien_ket_qua(self, ten_file, so_nguoi, sk):
        self.nhan_file.configure(text=f"File: {ten_file}")
        self.nhan_so_nguoi.configure(text=f"Số người: {so_nguoi}")
        self.nhan_thoi_gian.configure(text=f"Thời gian: {sk['thoi_gian']}")

        if so_nguoi == 0:
            self.the.configure(highlightbackground=MAU_XAM)
            self.nhan_tieu_de.configure(text="Không thấy người trong ảnh", fg=MAU_XAM)
            self.nhan_trang_thai.configure(text="● Không thấy người", fg=MAU_XAM)
            self.nhan_vai_tro.configure(text="—")
            self.nhan_hanh_vi.configure(text="—")
            self.nhan_loai.configure(text="Loại vi phạm: —")
            self.nhan_tin_cay.configure(text="Độ tin cậy: —")
            self.nut.configure(state="disabled")
            return

        mau = sk["mau"]
        self.the.configure(highlightbackground=mau)
        self.nhan_tieu_de.configure(text=sk["tieu_de"], fg=mau)
        self.nhan_vai_tro.configure(text=VAI_TRO)
        self.nhan_hanh_vi.configure(text=sk["loai"], fg=mau)
        self.nhan_loai.configure(text=f"Loại vi phạm: {sk['loai']}")
        self.nhan_tin_cay.configure(text=f"Độ tin cậy: {sk['tin_cay']:.1%}")
        if sk["vi_pham"]:
            self.nhan_trang_thai.configure(text="● KHÔNG thắt dây", fg=MAU_DO)
        else:
            self.nhan_trang_thai.configure(text="● Có thắt dây an toàn", fg=MAU_XANH)
        # XEM CHI TIẾT: đỏ nếu vi phạm, xanh nếu an toàn
        self.nut.configure(state="normal", bg=mau,
                           activebackground="#a61e1e" if sk["vi_pham"] else "#167a2c")

    # ================= xử lý ảnh =================
    def chon_anh(self):
        duong_dan = filedialog.askopenfilename(parent=self.root, title="Chọn ảnh cần kiểm tra",
                                               initialdir=str(self.thu_muc_gan_nhat),
                                               filetypes=LOAI_FILE)
        if not duong_dan:
            return                                         # bấm Hủy
        duong_dan = Path(duong_dan)
        self.thu_muc_gan_nhat = duong_dan.parent

        anh = doc_anh(duong_dan)
        if anh is None:
            messagebox.showerror("Không đọc được ảnh",
                                 f"File này không phải ảnh hợp lệ:\n{duong_dan}", parent=self.root)
            return

        self.root.configure(cursor="watch")
        self.nhan_trang_thai.configure(text="● Đang phân tích...", fg=MAU_XAM)
        self.root.update_idletasks()
        try:
            self.phan_tich(anh, duong_dan.name)
        except Exception as loi:
            messagebox.showerror("Lỗi khi phân tích", str(loi), parent=self.root)
            self.nhan_trang_thai.configure(text="● Lỗi khi phân tích", fg=MAU_DO)
        finally:
            self.root.configure(cursor="")

    def phan_tich(self, anh, ten_file):
        """Ảnh -> P1 (tìm người) -> P2 (từng vùng ngực) -> vẽ khung + cập nhật thẻ."""
        cac_nguoi = detect_people(anh)
        ket_qua = [(p, detect_seatbelt(p["chest_roi"])) for p in cac_nguoi]

        anh_ve = anh.copy()
        day = max(2, round(max(anh.shape[:2]) / 400))      # ảnh to thì nét khung dày hơn
        for p, r in ket_qua:
            x1, y1, x2, y2 = p["person_box"]
            cx1, cy1, cx2, cy2 = p["chest_box"]
            cv2.rectangle(anh_ve, (x1, y1), (x2, y2), XANH if r["seatbelt"] else DO, day)
            cv2.rectangle(anh_ve, (cx1, cy1), (cx2, cy2), VANG, day)

        rong, cao = KHUNG_TOI_DA
        hien = thu_vua_khung(anh_ve, rong, cao)
        self.khung_hinh.configure(image=hien, text="")
        self.khung_hinh.image = hien

        # Ưu tiên báo người VI PHẠM chắc chắn nhất; không ai vi phạm -> người an toàn chắc nhất
        vi_pham = [r for _, r in ket_qua if not r["seatbelt"]]
        nhom = vi_pham or [r for _, r in ket_qua]
        r = max(nhom, key=lambda x: x["confidence"]) if nhom else None
        co_vi_pham = bool(vi_pham)

        self.su_kien = {
            "anh": anh_ve,
            "loai": "Không thắt dây" if co_vi_pham else "Không vi phạm (có thắt dây)",
            "tin_cay": r["confidence"] if r else 0.0,
            "camera": TEN_NGUON,
            "thoi_gian": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            "debug_image": r.get("debug_image") if r else None,
            "vi_pham": co_vi_pham,
            "tieu_de": "⚠ PHÁT HIỆN VI PHẠM" if co_vi_pham else "✔ ĐÃ THẮT DÂY AN TOÀN",
            "mau": MAU_DO if co_vi_pham else MAU_XANH,
        }
        self.hien_ket_qua(ten_file, len(ket_qua), self.su_kien)

    # ================= nút khác =================
    def xem_chi_tiet(self):
        SeatbeltUI.xem_chi_tiet(self)       # dùng lại đúng cửa sổ chi tiết bên camera

    def ve_camera(self):
        """Đóng màn hình này, mở lại màn hình camera (main.py)."""
        file_camera = THU_MUC / "main.py"
        if not file_camera.exists():
            messagebox.showerror("Không tìm thấy file", f"Không có file:\n{file_camera}",
                                 parent=self.root)
            return
        self.dong()
        subprocess.Popen([sys.executable, str(file_camera)], cwd=str(THU_MUC))

    def dong(self):
        if getattr(self, "da_dong", False):
            return
        self.da_dong = True
        self.root.destroy()


def main():
    root = tk.Tk()
    SeatbeltImageUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
