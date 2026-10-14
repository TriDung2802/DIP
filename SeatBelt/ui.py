""" giao diện theo mẫu "thẻ cảnh báo vi phạm".

Bên trái : camera trực tiếp (có khung người, khung ngực).
Bên phải : THẺ CẢNH BÁO - hiện vi phạm gần nhất, đúng thứ tự như bản thiết kế:
           tiêu đề -> ảnh -> đối tượng/hành vi -> loại vi phạm -> độ tin cậy
           -> camera -> thời gian -> nút XEM CHI TIẾT + nút CHỌN ẢNH.
"""
import subprocess
import sys
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox

import cv2
from PIL import Image, ImageTk

XANH = (0, 200, 0)   # BGR
DO = (0, 0, 255)
VANG = (0, 220, 220)

LAT_HINH = 1          # 1 = gương, 0 = lật dọc, -1 = cả hai, None = không lật
DO_TRE_MS = 30        # ~30 khung hình/giây

TEN_CAMERA = "CAM-01"
VAI_TRO = "Người lái"     # tạm cố định; xem ghi chú về P1 trong hướng dẫn


TEN_MON_HOC = "Xử lý ảnh số"              # <-- đổi đúng tên môn học
TEN_DE_TAI = "Hệ thống phát hiện thắt dây an toàn qua camera"   # <-- đổi đúng tên đề tài
TEN_NHOM = "Nhóm 11"                       # <-- đổi đúng tên/số nhóm
DANH_SACH_THANH_VIEN = [                   # <-- điền đủ họ tên + MSSV từng thành viên
    ("Nguyễn Công Đại Lộc", "24110106"),
    ("Cao Đỗ Trí Dũng", "24110083"),
    ("Ngô Trần Ngân Đô", "24110088"),
]
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

        self.dung_header(root)     # thanh tiêu đề trên cùng, theo yêu cầu của thầy

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

    # ================= thanh tiêu đề trên cùng =================
    def dung_header(self, root):
        """Hình chữ nhật trên cùng, chia 2 nửa: trái = môn học/đề tài, phải = nhóm/thành viên."""
        header = tk.Frame(root, bg="#1b2a4a", highlightthickness=1, highlightbackground="gray")
        header.pack(side="top", fill="x")

        # --- Nửa trái: tên môn học + tên đề tài ---
        trai = tk.Frame(header, bg="#1b2a4a")
        trai.pack(side="left", fill="both", expand=True, padx=14, pady=8)
        tk.Label(trai, text=TEN_MON_HOC, font=(FONT, 13, "bold"),
                 bg="#1b2a4a", fg="white", anchor="w", justify="left").pack(anchor="w")
        tk.Label(trai, text=TEN_DE_TAI, font=(FONT, 10),
                 bg="#1b2a4a", fg="#d8dee9", anchor="w", justify="left",
                 wraplength=320).pack(anchor="w")

        # --- Đường kẻ chia đôi ở giữa ---
        tk.Frame(header, bg="gray", width=1).pack(side="left", fill="y", pady=8)

        # --- Nửa phải: tên nhóm + danh sách thành viên (họ tên + MSSV) ---
        phai = tk.Frame(header, bg="#1b2a4a")
        phai.pack(side="right", fill="both", expand=True, padx=14, pady=8)
        tk.Label(phai, text=TEN_NHOM, font=(FONT, 13, "bold"),
                 bg="#1b2a4a", fg="white", anchor="e", justify="right").pack(anchor="e")
        for ho_ten, mssv in DANH_SACH_THANH_VIEN:
            tk.Label(phai, text=f"{ho_ten} - {mssv}", font=(FONT, 10),
                     bg="#1b2a4a", fg="#d8dee9", anchor="e", justify="right").pack(anchor="e")

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

        # Hàng nút: XEM CHI TIẾT | CHỌN ẢNH
        hang_nut = tk.Frame(the)
        hang_nut.pack(fill="x", padx=12, pady=12)
        kieu_nut = dict(font=(FONT, 11, "bold"), width=13, pady=4, cursor="hand2")

        self.nut = tk.Button(hang_nut, text="XEM CHI TIẾT", command=self.xem_chi_tiet,
                             bg=MAU_DO, fg="white", activebackground="#a61e1e",
                             activeforeground="white", disabledforeground="#e8b4b4",
                             **kieu_nut)
        self.nut.pack(side="left", expand=True, padx=(0, 4))

        self.nut_chon_anh = tk.Button(hang_nut, text="CHỌN ẢNH", command=self.mo_chon_anh,
                                      **kieu_nut)
        self.nut_chon_anh.pack(side="left", expand=True, padx=(4, 0))

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
        """Mở (hoặc làm mới) MỘT cửa sổ chi tiết duy nhất cho vi phạm gần nhất."""
        sk = self.su_kien
        if sk is None:
            return

        # Đã mở rồi thì dùng lại, không đẻ thêm cửa sổ mới mỗi lần bấm
        cua_so = getattr(self, "cua_so_chi_tiet", None)
        if cua_so is not None and cua_so.winfo_exists():
            for con in cua_so.winfo_children():
                con.destroy()
        else:
            cua_so = tk.Toplevel(self.root)
            cua_so.title("Chi tiết vi phạm")
            cua_so.resizable(False, False)
            cua_so.transient(self.root)               # luôn nằm trên cửa sổ chính
            cua_so.bind("<Escape>", lambda e: cua_so.destroy())
            self.cua_so_chi_tiet = cua_so

        # --- Tiêu đề (mặc định: đỏ "vi phạm"; màn hình chọn ảnh có thể truyền khác) ---
        mau = sk.get("mau", MAU_DO)
        tk.Label(cua_so, text=sk.get("tieu_de", "⚠ CHI TIẾT VI PHẠM"), font=(FONT, 14, "bold"),
                 fg=mau, anchor="w").pack(fill="x", padx=14, pady=(12, 4))

        than = tk.Frame(cua_so)
        than.pack(padx=14, pady=4)

        # --- Trái: ảnh chụp lúc vi phạm ---
        anh = thanh_photoimage(sk["anh"], 520)
        nhan = tk.Label(than, image=anh, highlightthickness=2, highlightbackground=mau)
        nhan.image = anh                              # giữ tham chiếu
        nhan.grid(row=0, column=0, rowspan=2, sticky="n")

        # --- Phải: bảng thông tin ---
        bang = tk.Frame(than, highlightthickness=1, highlightbackground="gray")
        bang.grid(row=0, column=1, sticky="nw", padx=(14, 0))
        dong_tt = [
            ("Đối tượng", VAI_TRO),
            ("Loại vi phạm", sk["loai"]),
            ("Độ tin cậy", f"{sk['tin_cay']:.1%}"),
            ("Camera", sk["camera"]),
            ("Thời gian", sk["thoi_gian"]),
        ]
        for i, (ten, gia_tri) in enumerate(dong_tt):
            tk.Label(bang, text=f"{ten}:", font=(FONT, 11, "bold"), anchor="w"
                     ).grid(row=i, column=0, sticky="w", padx=(8, 6), pady=3)
            tk.Label(bang, text=gia_tri, font=(FONT, 11), anchor="w",
                     fg=mau if ten == "Loại vi phạm" else "black"
                     ).grid(row=i, column=1, sticky="w", padx=(0, 8), pady=3)

        # --- Phải, bên dưới: ảnh trung gian của bước xử lý ảnh (nếu có) ---
        dbg_goc = sk.get("debug_image")
        if dbg_goc is not None:
            khung_dbg = tk.Frame(than)
            khung_dbg.grid(row=1, column=1, sticky="nw", padx=(14, 0), pady=(10, 0))
            tk.Label(khung_dbg, text="Ảnh trung gian (xử lý ảnh):",
                     font=(FONT, 10, "bold"), anchor="w").pack(anchor="w")
            try:
                dbg = thanh_photoimage(dbg_goc, 260)
                nd = tk.Label(khung_dbg, image=dbg)
                nd.image = dbg
                nd.pack(anchor="w")
            except Exception as loi:                  # ảnh debug lỗi thì vẫn hiện phần còn lại
                tk.Label(khung_dbg, text=f"(không hiển thị được: {loi})",
                         fg=MAU_XAM).pack(anchor="w")

        tk.Button(cua_so, text="ĐÓNG", font=(FONT, 11, "bold"), width=12,
                  command=cua_so.destroy).pack(pady=(8, 12))

        cua_so.deiconify()
        cua_so.lift()
        cua_so.focus_force()

    def mo_chon_anh(self):
        """Đóng cửa sổ camera hiện tại rồi mở seatbelt_image_ui.py (chế độ chọn ảnh)."""
        file_moi = Path(__file__).resolve().parent / "seatbelt_image_ui.py"
        if not file_moi.exists():
            messagebox.showerror("Không tìm thấy file",
                                 f"Không có file:\n{file_moi}", parent=self.root)
            return
        self.dong()                                   # nhả camera + đóng cửa sổ này trước
        # Chạy bằng đúng trình thông dịch Python đang dùng (venv nào cũng được)
        subprocess.Popen([sys.executable, str(file_moi)], cwd=str(file_moi.parent))

    def dong(self):
        if getattr(self, "da_dong", False):
            return
        self.da_dong = True
        self.cap.release()
        self.root.destroy()
