import time
import tkinter as tk
from datetime import datetime

import cv2
from PIL import Image, ImageTk

LAT_HINH = 1          # 1 = gương, 0 = lật dọc, -1 = cả hai, None = không lật
DO_TRE_MS = 30        # ~30 khung hình/giây

TEN_CAMERA = "CAM-01"
VAI_TRO = "Người lái"
COOLDOWN_GIAY = 3         # ít nhất 3 giây mới tạo thêm 1 cảnh báo mới
FONT = "Segoe UI"

TEN_MON_HOC = "Xử lý ảnh số"
TEN_DE_TAI = "Hệ thống phát hiện thắt dây an toàn qua camera"
TEN_NHOM = "Nhóm 11"

DANH_SACH_THANH_VIEN = [
    ("Nguyễn Công Đại Lộc", "24110106"),
    ("Cao Đỗ Trí Dũng", "24110083"),
    ("Ngô Trần Ngân Đô", "24110088"),
]

MAU_DO = "#d62828"
MAU_XANH = "#1e9e3a"
MAU_XAM = "gray"


def thanh_photoimage(anh, rong):
    """Ảnh OpenCV (BGR hoặc xám) -> ảnh Tkinter, thu về chiều rộng `rong`."""
    if anh.ndim == 2:
        rgb = cv2.cvtColor(anh, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(anh, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    rgb = cv2.resize(rgb, (rong, max(1, int(h * rong / w))))
    return ImageTk.PhotoImage(Image.fromarray(rgb))


def ve_nhan(frame, text, origin, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )
    height, width = frame.shape[:2]
    x = max(0, min(origin[0], width - text_width - 8))
    y = max(0, min(origin[1], height - text_height - baseline - 8))
    cv2.rectangle(
        frame,
        (x, y),
        (x + text_width + 8, y + text_height + baseline + 8),
        color,
        cv2.FILLED,
    )
    cv2.putText(
        frame,
        text,
        (x + 4, y + text_height + 4),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


class SeatbeltUI:

    def __init__(self, root, cap, pipeline, khi_co_su_kien=None, luu_co_day=None, luu_khong_day=None):
        self.root = root
        self.cap = cap
        self.pipeline = pipeline
        self.khi_co_su_kien = khi_co_su_kien

        self.ham_luu_co_day = luu_co_day
        self.ham_luu_khong_day = luu_khong_day

        self.su_kien = None
        self.lan_bao_cuoi = 0.0
        
        root.title("Seatbelt AI")
        root.state("zoomed")
        root.minsize(900, 600) 
        root.protocol("WM_DELETE_WINDOW", self.dong)
        root.bind("<KeyPress-s>", self.luu_co_day)
        root.bind("<KeyPress-n>", self.luu_khong_day)
        self.dung_header(root)

        # --- Bên trái: camera ---
        self.khung_hinh = tk.Label(root, bg="black")
        self.khung_hinh.pack(
            side="left",
            padx=10,
            pady=10,
            fill="both",
            expand=True
        )

        # --- Bên phải: trạng thái trực tiếp + thẻ cảnh báo ---
        cot_phai = tk.Frame(root, width=340)
        cot_phai.pack_propagate(False)
        cot_phai.pack(
            side="right",
            fill="both",
            expand=False,
            padx=10,
            pady=10
        )

        self.nhan_trang_thai = tk.Label(
            cot_phai, text="● Đang khởi động...", font=(FONT, 11, "bold"), anchor="w", fg=MAU_XAM
        )
        self.nhan_trang_thai.pack(fill="x", pady=(0, 6))

        self.the = tk.Frame(cot_phai, highlightthickness=2, highlightbackground=MAU_XAM)
        self.the.pack(fill="both", expand=True)
        self.dung_the()
        self.hien_the_rong()

        self.cap_nhat()

    def luu_co_day(self, event=None):
        if self.ham_luu_co_day is None:
            return

        cac_nguoi = self.pipeline(self.frame_hien_tai)
        if not cac_nguoi:
            return

        person = cac_nguoi[0][0]
        x1, y1, x2, y2 = person["person_box"]
        roi = self.frame_hien_tai[y1:y2, x1:x2]
        self.ham_luu_co_day(roi)

    def luu_khong_day(self, event=None):
        if self.ham_luu_khong_day is None:
            return

        cac_nguoi = self.pipeline(self.frame_hien_tai)
        if not cac_nguoi:
            return

        person = cac_nguoi[0][0]
        x1, y1, x2, y2 = person["person_box"]
        roi = self.frame_hien_tai[y1:y2, x1:x2]
        self.ham_luu_khong_day(roi)

    def dung_header(self, root):
        header = tk.Frame(
            root,
            bg="#1b2a4a",
            highlightthickness=1,
            highlightbackground="gray",
        )
        header.pack(side="top", fill="x")

        trai = tk.Frame(header, bg="#1b2a4a")
        trai.pack(
            side="left",
            fill="both",
            expand=True,
            padx=14,
            pady=8,
        )

        tk.Label(
            trai,
            text=TEN_MON_HOC,
            font=(FONT, 13, "bold"),
            bg="#1b2a4a",
            fg="white",
            anchor="w",
        ).pack(anchor="w")

        tk.Label(
            trai,
            text=TEN_DE_TAI,
            font=(FONT, 10),
            bg="#1b2a4a",
            fg="#d8dee9",
            anchor="w",
            justify="left",
            wraplength=320,
        ).pack(anchor="w")

        tk.Frame(
            header,
            bg="gray",
            width=1,
        ).pack(side="left", fill="y", pady=8)

        phai = tk.Frame(header, bg="#1b2a4a")
        phai.pack(
            side="right",
            fill="both",
            expand=True,
            padx=14,
            pady=8,
        )

        tk.Label(
            phai,
            text=TEN_NHOM,
            font=(FONT, 13, "bold"),
            bg="#1b2a4a",
            fg="white",
            anchor="e",
        ).pack(anchor="e")

        for ho_ten, mssv in DANH_SACH_THANH_VIEN:
            tk.Label(
                phai,
                text=f"{ho_ten} - {mssv}",
                font=(FONT, 10),
                bg="#1b2a4a",
                fg="#d8dee9",
                anchor="e",
            ).pack(anchor="e")

    def dung_the(self):
        the = self.the
        kw = dict(anchor="w", font=(FONT, 10))

        # 1. HÀNG NÚT BẤM DƯỚI CÙNG (Luôn cố định vị trí ở đáy)
        hang_nut = tk.Frame(the)
        hang_nut.pack(side="bottom", fill="x", padx=10, pady=10)

        self.nut = tk.Button(
            hang_nut,
            text="XEM CHI TIẾT",
            font=(FONT, 10, "bold"),
            bg=MAU_DO,
            fg="white",
            activebackground="#a61e1e",
            activeforeground="white",
            disabledforeground="#e8b4b4",
            cursor="hand2",
            command=self.xem_chi_tiet,
        )
        self.nut.pack(side="left", expand=True, fill="x", padx=(0, 4), ipady=6)

        self.nut_chon_anh = tk.Button(
            hang_nut,
            text="CHỌN ẢNH",
            font=(FONT, 10, "bold"),
            bg="#e1e1e1",
            fg="black",
            activebackground="#cccccc",
            cursor="hand2",
            command=self.mo_chon_anh,
        )
        self.nut_chon_anh.pack(side="right", expand=True, fill="x", padx=(4, 0), ipady=6)

        # 2. KHUNG NỘI DUNG PHÍA TRÊN
        khung_tren = tk.Frame(the)
        khung_tren.pack(side="top", fill="both", expand=True)

        self.nhan_tieu_de = tk.Label(khung_tren, font=(FONT, 11, "bold"), anchor="w")
        self.nhan_tieu_de.pack(fill="x", padx=10, pady=(6, 2))

        self.anh_rong = ImageTk.PhotoImage(Image.new("RGB", (240, 130), "#2b2b2b"))
        self.nhan_anh = tk.Label(
            khung_tren, image=self.anh_rong, text="[ ẢNH CAMERA ]",
            compound="center", fg="white", font=(FONT, 9)
        )
        self.nhan_anh.pack(padx=10, pady=2)

        o = tk.Frame(khung_tren, highlightthickness=1, highlightbackground="gray")
        o.pack(anchor="w", padx=10, pady=4)
        self.nhan_vai_tro = tk.Label(o, font=(FONT, 10), anchor="w", width=22)
        self.nhan_vai_tro.pack(padx=6, pady=(2, 0))
        self.nhan_hanh_vi = tk.Label(o, font=(FONT, 10), anchor="w", width=22)
        self.nhan_hanh_vi.pack(padx=6, pady=(0, 2))

        self.nhan_loai = tk.Label(khung_tren, **kw)
        self.nhan_tin_cay = tk.Label(khung_tren, **kw)
        self.nhan_camera = tk.Label(khung_tren, **kw)
        self.nhan_thoi_gian = tk.Label(khung_tren, **kw)
        for nhan in (self.nhan_loai, self.nhan_tin_cay, self.nhan_camera, self.nhan_thoi_gian):
            nhan.pack(fill="x", padx=10, pady=1)

    def hien_the_rong(self):
        """Chưa có vi phạm nào."""
        self.the.configure(highlightbackground=MAU_XAM)
        self.nhan_tieu_de.configure(text="Chưa có vi phạm", fg=MAU_XAM)
        self.nhan_anh.configure(image=self.anh_rong, text="[ ẢNH CAMERA ]")
        self.nhan_anh.image = self.anh_rong
        self.nhan_vai_tro.configure(text="Người lái")
        self.nhan_hanh_vi.configure(text="Chưa xác định")
        self.nhan_loai.configure(text="Loại vi phạm: —")
        self.nhan_tin_cay.configure(text="Độ tin cậy: —")
        self.nhan_camera.configure(text=f"Camera: {TEN_CAMERA}")
        self.nhan_thoi_gian.configure(text="Thời gian: —")
        self.nut.configure(state="disabled")

    def hien_the_vi_pham(self, sk):
        self.the.configure(highlightbackground=MAU_DO)
        self.nhan_tieu_de.configure(text="⚠ PHÁT HIỆN VI PHẠM", fg=MAU_DO)
        
        anh = thanh_photoimage(sk["anh"], 240)
        self.nhan_anh.configure(image=anh, text="")
        self.nhan_anh.image = anh

        self.nhan_vai_tro.configure(text=VAI_TRO)
        self.nhan_hanh_vi.configure(text=sk["loai"])
        self.nhan_loai.configure(text=f"Loại vi phạm: {sk['loai']}")
        self.nhan_tin_cay.configure(text=f"Độ tin cậy: {sk['tin_cay']:.1%}")
        self.nhan_camera.configure(text=f"Camera: {sk['camera']}")
        self.nhan_thoi_gian.configure(text=f"Thời gian: {sk['thoi_gian']}")
        self.nut.configure(state="normal")

    def cap_nhat(self):
        ok, frame = self.cap.read()
        if ok:
            if LAT_HINH is not None:
                frame = cv2.flip(frame, LAT_HINH)
            self.frame_hien_tai = frame.copy()

            cac_nguoi = self.pipeline(frame)
            for person, result in cac_nguoi:
                self.ve_khung(frame, person, result)

            self.xu_ly_vi_pham(frame, cac_nguoi)
            self.hien_thi_hinh(frame)
            self.hien_trang_thai(cac_nguoi)

        self.root.after(DO_TRE_MS, self.cap_nhat)

    def xu_ly_vi_pham(self, frame, cac_nguoi):
        vi_pham = [r for _, r in cac_nguoi if not r["seatbelt"]]
        if not vi_pham:
            return
        bay_gio = time.time()
        if bay_gio - self.lan_bao_cuoi < COOLDOWN_GIAY:
            return
        self.lan_bao_cuoi = bay_gio

        r = max(vi_pham, key=lambda x: x["confidence"])
        self.su_kien = {
            "anh": frame.copy(),
            "loai": "Không thắt dây",
            "tin_cay": r["confidence"],
            "camera": TEN_CAMERA,
            "thoi_gian": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            "debug_image": r.get("debug_image"),
        }
        self.hien_the_vi_pham(self.su_kien)
        if self.khi_co_su_kien is not None:
            self.khi_co_su_kien(self.su_kien)

    def ve_khung(self, frame, person, result):
        x1, y1, x2, y2 = person["person_box"]
        person_color = (255, 0, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), person_color, 2)
        ve_nhan(frame, "person", (x1, max(0, y1 - 28)), person_color)

        polygons = result.get("polygons")
        if polygons is None:
            polygon = result.get("polygon")
            polygons = [] if polygon is None else [polygon]

        if polygons:
            largest_polygon = max(
                polygons,
                key=lambda item: abs(cv2.contourArea(item.astype("int32"))),
            )
            label_position = None
            for polygon in polygons:
                if len(polygon) < 3:
                    continue

                polygon_on_frame = polygon.astype("int32", copy=True)
                polygon_on_frame[:, 0] += x1
                polygon_on_frame[:, 1] += y1

                cv2.polylines(
                    frame,
                    [polygon_on_frame],
                    True,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                if polygon is largest_polygon:
                    label_position = (
                        int(polygon_on_frame[:, 0].min()),
                        int(polygon_on_frame[:, 1].min()),
                    )

            if label_position is not None:
                label_x, label_y = label_position
                ve_nhan(
                    frame,
                    "seatbelt",
                    (label_x, max(0, label_y - 28)),
                    (0, 160, 0),
                )

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

    def mo_chon_anh(self):
        import subprocess
        import sys
        from pathlib import Path

        file_anh_ui = Path(__file__).resolve().parent.parent / "seatbelt_image_ui.py"
        if not file_anh_ui.exists():
            from tkinter import messagebox
            messagebox.showerror("Lỗi", f"Không tìm thấy file:\n{file_anh_ui}")
            return

        self.dong()
        subprocess.Popen([sys.executable, str(file_anh_ui)])

    def xem_chi_tiet(self):
        sk = self.su_kien
        if sk is None:
            return

        cua_so = tk.Toplevel(self.root)
        cua_so.title("Chi tiết vi phạm")
        cua_so.geometry("700x650")
        cua_so.minsize(500, 450)

        main_frame = tk.Frame(cua_so, padx=15, pady=15)
        main_frame.pack(fill="both", expand=True)

        anh = thanh_photoimage(sk["anh"], 600)
        nhan_anh = tk.Label(main_frame, image=anh)
        nhan_anh.image = anh
        nhan_anh.pack(pady=(0, 10))

        chu = (
            f"Đối tượng: {VAI_TRO}\n"
            f"Loại vi phạm: {sk['loai']}\n"
            f"Độ tin cậy: {sk['tin_cay']:.1%}\n"
            f"Camera: {sk['camera']}\n"
            f"Thời gian: {sk['thoi_gian']}"
        )
        tk.Label(
            main_frame, text=chu, font=(FONT, 11), justify="left", anchor="w"
        ).pack(fill="x", pady=5)

        if sk.get("debug_image") is not None:
            tk.Label(
                main_frame,
                text="Ảnh trung gian (xử lý ảnh):",
                font=(FONT, 11, "bold"),
                anchor="w",
            ).pack(fill="x", pady=(10, 5))

            dbg = thanh_photoimage(sk["debug_image"], 300)
            nd = tk.Label(main_frame, image=dbg)
            nd.image = dbg
            nd.pack(pady=(0, 10))

    def dong(self):
        if hasattr(self, "cap") and self.cap is not None:
            self.cap.release()
        self.root.destroy()