"""
ui.py của Lộc (P3) - BƯỚC 2 (bản 2): giao diện theo mẫu "thẻ cảnh báo vi phạm".

Bên trái : camera trực tiếp (có khung người, khung ngực).
Bên phải : THẺ CẢNH BÁO - hiện vi phạm gần nhất, đúng thứ tự như bản thiết kế:
           tiêu đề -> ảnh -> đối tượng/hành vi -> loại vi phạm -> độ tin cậy
           -> camera -> thời gian -> nút XEM CHI TIẾT.

ui.py CHỈ lo hiển thị. Nó không biết YOLO hay DIP - chỉ gọi hàm `pipeline` của main.py.
"""
import time
import tkinter as tk
from datetime import datetime

import cv2
from PIL import Image, ImageTk

XANH = (0, 200, 0)   # BGR
DO = (0, 0, 255)
VANG = (0, 220, 220)

LAT_HINH = 1          # 1 = gương, 0 = lật dọc, -1 = cả hai, None = không lật
DO_TRE_MS = 30        # ~30 khung hình/giây

TEN_CAMERA = "CAM-01"
VAI_TRO = "Người lái"     # tạm cố định; xem ghi chú về P1 trong hướng dẫn
COOLDOWN_GIAY = 3         # ít nhất 3 giây mới tạo thêm 1 cảnh báo mới
FONT = "Segoe UI"

MAU_DO = "#d62828"
MAU_XANH = "#1e9e3a"
MAU_XAM = "gray"


def thanh_photoimage(anh, rong):
    """Ảnh OpenCV (BGR hoặc xám) -> ảnh Tkinter, thu về chiều rộng `rong`."""
    if anh.ndim == 2:
        rgb = cv2.cvtColor(anh, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(anh, cv2.COLOR_BGR2RGB)   # OpenCV=BGR, Tkinter=RGB
    h, w = rgb.shape[:2]
    rgb = cv2.resize(rgb, (rong, max(1, int(h * rong / w))))
    return ImageTk.PhotoImage(Image.fromarray(rgb))


class SeatbeltUI:
    def __init__(self, root, cap, pipeline, khi_co_su_kien=None):
        self.root = root
        self.cap = cap
        self.pipeline = pipeline          # hàm: frame -> [(person, result), ...]
        self.khi_co_su_kien = khi_co_su_kien   # hàm gọi mỗi khi có cảnh báo mới (vd: logger.ghi)
        self.su_kien = None               # cảnh báo gần nhất
        self.lan_bao_cuoi = 0.0

        root.title("Seatbelt AI")
        root.protocol("WM_DELETE_WINDOW", self.dong)

        # --- Bên trái: camera ---
        self.khung_hinh = tk.Label(root)
        self.khung_hinh.pack(side="left", padx=10, pady=10)

        # --- Bên phải: trạng thái trực tiếp + thẻ cảnh báo ---
        cot_phai = tk.Frame(root)
        cot_phai.pack(side="right", fill="y", padx=10, pady=10)

        self.nhan_trang_thai = tk.Label(cot_phai, text="● Đang khởi động...",
                                        font=(FONT, 11, "bold"), anchor="w", fg=MAU_XAM)
        self.nhan_trang_thai.pack(fill="x", pady=(0, 6))

        self.the = tk.Frame(cot_phai, highlightthickness=2, highlightbackground=MAU_XAM)
        self.the.pack(fill="both", expand=True)
        self.dung_the()
        self.hien_the_rong()

        self.cap_nhat()

    # ================= dựng thẻ cảnh báo (1 lần) =================
    def dung_the(self):
        the = self.the
        kw = dict(anchor="w", font=(FONT, 11))

        self.nhan_tieu_de = tk.Label(the, font=(FONT, 13, "bold"), anchor="w")
        self.nhan_tieu_de.pack(fill="x", padx=12, pady=(10, 6))

        # Ảnh nền xám làm chỗ giữ chỗ "[ ẢNH CAMERA ]"
        self.anh_rong = ImageTk.PhotoImage(Image.new("RGB", (270, 200), "#2b2b2b"))
        self.nhan_anh = tk.Label(the, image=self.anh_rong, text="[ ẢNH CAMERA ]",
                                 compound="center", fg="white", font=(FONT, 11))
        self.nhan_anh.pack(padx=12, pady=6)

        # Ô nhỏ: Người lái / Không thắt dây
        o = tk.Frame(the, highlightthickness=1, highlightbackground="gray")
        o.pack(anchor="w", padx=12, pady=6)
        self.nhan_vai_tro = tk.Label(o, font=(FONT, 11), anchor="w", width=20)
        self.nhan_vai_tro.pack(padx=6, pady=(4, 0))
        self.nhan_hanh_vi = tk.Label(o, font=(FONT, 11), anchor="w", width=20)
        self.nhan_hanh_vi.pack(padx=6, pady=(0, 4))

        self.nhan_loai = tk.Label(the, **kw)
        self.nhan_tin_cay = tk.Label(the, **kw)
        self.nhan_camera = tk.Label(the, **kw)
        self.nhan_thoi_gian = tk.Label(the, **kw)
        for nhan in (self.nhan_loai, self.nhan_tin_cay, self.nhan_camera, self.nhan_thoi_gian):
            nhan.pack(fill="x", padx=12, pady=1)

        self.nut = tk.Button(the, text="XEM CHI TIẾT", font=(FONT, 11, "bold"),
                             command=self.xem_chi_tiet)
        self.nut.pack(pady=12)

    # ================= 2 trạng thái của thẻ =================
    def hien_the_rong(self):
        """Chưa có vi phạm nào."""
        self.the.configure(highlightbackground=MAU_XAM)
        self.nhan_tieu_de.configure(text="Chưa có vi phạm", fg=MAU_XAM)
        self.nhan_anh.configure(image=self.anh_rong, text="[ ẢNH CAMERA ]")
        self.nhan_anh.image = self.anh_rong
        self.nhan_vai_tro.configure(text="—")
        self.nhan_hanh_vi.configure(text="—")
        self.nhan_loai.configure(text="Loại vi phạm: —")
        self.nhan_tin_cay.configure(text="Độ tin cậy: —")
        self.nhan_camera.configure(text=f"Camera: {TEN_CAMERA}")
        self.nhan_thoi_gian.configure(text="Thời gian: —")
        self.nut.configure(state="disabled")

    def hien_the_vi_pham(self, sk):
        self.the.configure(highlightbackground=MAU_DO)
        self.nhan_tieu_de.configure(text="⚠ PHÁT HIỆN VI PHẠM", fg=MAU_DO)
        anh = thanh_photoimage(sk["anh"], 270)
        self.nhan_anh.configure(image=anh, text="")
        self.nhan_anh.image = anh                 # giữ tham chiếu (nếu không ảnh biến mất)
        self.nhan_vai_tro.configure(text=VAI_TRO)
        self.nhan_hanh_vi.configure(text=sk["loai"])
        self.nhan_loai.configure(text=f"Loại vi phạm: {sk['loai']}")
        self.nhan_tin_cay.configure(text=f"Độ tin cậy: {sk['tin_cay']:.1%}")
        self.nhan_camera.configure(text=f"Camera: {sk['camera']}")
        self.nhan_thoi_gian.configure(text=f"Thời gian: {sk['thoi_gian']}")
        self.nut.configure(state="normal")

    # ================= vòng lặp chính =================
    def cap_nhat(self):
        ok, frame = self.cap.read()
        if ok:
            if LAT_HINH is not None:
                frame = cv2.flip(frame, LAT_HINH)

            cac_nguoi = self.pipeline(frame)              # gọi P1 rồi P2
            for person, result in cac_nguoi:
                self.ve_khung(frame, person, result)

            self.xu_ly_vi_pham(frame, cac_nguoi)
            self.hien_thi_hinh(frame)
            self.hien_trang_thai(cac_nguoi)

        self.root.after(DO_TRE_MS, self.cap_nhat)         # không dùng while True

    def xu_ly_vi_pham(self, frame, cac_nguoi):
        """Nếu có người không thắt dây -> tạo cảnh báo (có giãn cách COOLDOWN)."""
        vi_pham = [r for _, r in cac_nguoi if not r["seatbelt"]]
        if not vi_pham:
            return
        bay_gio = time.time()
        if bay_gio - self.lan_bao_cuoi < COOLDOWN_GIAY:
            return                                        # vừa báo xong, chưa báo lại
        self.lan_bao_cuoi = bay_gio

        r = max(vi_pham, key=lambda x: x["confidence"])
        self.su_kien = {
            "anh": frame.copy(),                          # chụp lại khoảnh khắc vi phạm
            "loai": "Không thắt dây",
            "tin_cay": r["confidence"],
            "camera": TEN_CAMERA,
            "thoi_gian": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            "debug_image": r.get("debug_image"),
        }
        self.hien_the_vi_pham(self.su_kien)
        if self.khi_co_su_kien is not None:
            self.khi_co_su_kien(self.su_kien)             # báo cho bên ngoài (logger...)

    # ================= các hàm con =================
    def ve_khung(self, frame, person, result):
        x1, y1, x2, y2 = person["person_box"]
        cx1, cy1, cx2, cy2 = person["chest_box"]
        mau = XANH if result["seatbelt"] else DO
        cv2.rectangle(frame, (x1, y1), (x2, y2), mau, 2)
        cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), VANG, 2)

    def hien_thi_hinh(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        anh = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.khung_hinh.configure(image=anh)
        self.khung_hinh.image = anh

    def hien_trang_thai(self, cac_nguoi):
        if not cac_nguoi:
            chu, mau = "● Không thấy người", MAU_XAM
        elif all(r["seatbelt"] for _, r in cac_nguoi):
            chu, mau = "● Đang thắt dây an toàn", MAU_XANH
        else:
            chu, mau = "● Đang KHÔNG thắt dây", MAU_DO
        self.nhan_trang_thai.configure(text=chu, fg=mau)

    def xem_chi_tiet(self):
        sk = self.su_kien
        if sk is None:
            return
        cua_so = tk.Toplevel(self.root)
        cua_so.title("Chi tiết vi phạm")

        anh = thanh_photoimage(sk["anh"], 640)
        nhan = tk.Label(cua_so, image=anh)
        nhan.image = anh
        nhan.pack(padx=10, pady=10)

        chu = (f"Đối tượng: {VAI_TRO}\n"
               f"Loại vi phạm: {sk['loai']}\n"
               f"Độ tin cậy: {sk['tin_cay']:.1%}\n"
               f"Camera: {sk['camera']}\n"
               f"Thời gian: {sk['thoi_gian']}")
        tk.Label(cua_so, text=chu, font=(FONT, 12), justify="left").pack(padx=10, anchor="w")

        if sk["debug_image"] is not None:                 # ảnh trung gian của Đô (nếu có)
            tk.Label(cua_so, text="Ảnh trung gian (xử lý ảnh):",
                     font=(FONT, 11, "bold")).pack(pady=(10, 0))
            dbg = thanh_photoimage(sk["debug_image"], 320)
            nd = tk.Label(cua_so, image=dbg)
            nd.image = dbg
            nd.pack(pady=(0, 10))

    def dong(self):
        self.cap.release()
        self.root.destroy()
