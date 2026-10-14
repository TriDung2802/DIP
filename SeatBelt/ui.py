import time
import tkinter as tk
from datetime import datetime
from tkinter import filedialog  # Thêm thư viện chọn file

import cv2
from PIL import Image, ImageTk

XANH = (0, 200, 0)   # BGR
DO = (0, 0, 255)
VANG = (0, 220, 220)

LAT_HINH = 1          # 1 = gương, 0 = lật dọc, -1 = cả hai, None = không lật
DO_TRE_MS = 30        # ~30 khung hình/giây

TEN_CAMERA = "CAM-01"
VAI_TRO = "Người lái"     # tạm cố định

TEN_MON_HOC = "Xử lý ảnh số"
TEN_DE_TAI = "Hệ thống phát hiện thắt dây an toàn qua camera"
TEN_NHOM = "Nhóm 11"
DANH_SACH_THANH_VIEN = [
    ("Cao Đỗ Trí Dũng", "24110083"),
    ("Ngô Trần Ngân Đô", "24110088"),
    ("Nguyễn Công Đại Lộc", "24110106"),
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
        self.khi_co_su_kien = khi_co_su_kien   # hàm gọi mỗi khi có cảnh báo mới
        self.su_kien = None               # cảnh báo gần nhất
        self.lan_bao_cuoi = 0.0

        # MỚI: Biến lưu trữ chế độ ảnh tĩnh hoặc camera
        self.anh_file_cv = None           # Lưu frame ảnh tải lên từ file
        self.che_do_anh = False           # False: Camera live, True: Đang xem ảnh file

        root.title("Seatbelt AI")
        root.protocol("WM_DELETE_WINDOW", self.dong)

        self.dung_header(root)     # thanh tiêu đề trên cùng

        # --- Bên trái: camera ---
        self.khung_hinh = tk.Label(root)
        self.khung_hinh.pack(side="left", padx=10, pady=10)

        # --- Bên phải: trạng thái trực tiếp + thẻ cảnh báo ---
        cot_phai = tk.Frame(root)
        cot_phai.pack(side="right", fill="y", padx=10, pady=10)

        # MỚI: Khung chứa nút điều khiển nguồn đầu vào (Camera / Ảnh file)
        khung_nguon = tk.LabelFrame(cot_phai, text="Nguồn đầu vào", font=(FONT, 9, "bold"), fg="#333333")
        khung_nguon.pack(fill="x", pady=(0, 8))

        btn_chonn_anh = tk.Button(khung_nguon, text="📁 Chọn ảnh file", font=(FONT, 9), command=self.chon_anh_file)
        btn_chonn_anh.pack(side="left", padx=5, pady=5, expand=True, fill="x")

        btn_cam = tk.Button(khung_nguon, text="📷 Live Camera", font=(FONT, 9), command=self.dung_camera)
        btn_cam.pack(side="right", padx=5, pady=5, expand=True, fill="x")

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
        header = tk.Frame(root, bg="#1b2a4a", highlightthickness=1, highlightbackground="gray")
        header.pack(side="top", fill="x")

        trai = tk.Frame(header, bg="#1b2a4a")
        trai.pack(side="left", fill="both", expand=True, padx=14, pady=8)
        tk.Label(trai, text=TEN_MON_HOC, font=(FONT, 13, "bold"),
                 bg="#1b2a4a", fg="white", anchor="w", justify="left").pack(anchor="w")
        tk.Label(trai, text=TEN_DE_TAI, font=(FONT, 10),
                 bg="#1b2a4a", fg="#d8dee9", anchor="w", justify="left",
                 wraplength=320).pack(anchor="w")

        tk.Frame(header, bg="gray", width=1).pack(side="left", fill="y", pady=8)

        phai = tk.Frame(header, bg="#1b2a4a")
        phai.pack(side="right", fill="both", expand=True, padx=14, pady=8)
        tk.Label(phai, text=TEN_NHOM, font=(FONT, 13, "bold"),
                 bg="#1b2a4a", fg="white", anchor="e", justify="right").pack(anchor="e")
        for ho_ten, mssv in DANH_SACH_THANH_VIEN:
            # Sửa lỗi SyntaxError bằng việc gộp dòng chính xác
            tk.Label(phai, text=f"{ho_ten} - {mssv}", font=(FONT, 10),
                     bg="#1b2a4a", fg="#d8dee9", anchor="e", justify="right").pack(anchor="e")

    # ================= dựng thẻ cảnh báo (1 lần) =================
    def dung_the(self):
        the = self.the
        kw = dict(anchor="w", font=(FONT, 11))

        self.nhan_tieu_de = tk.Label(the, font=(FONT, 13, "bold"), anchor="w")
        self.nhan_tieu_de.pack(fill="x", padx=12, pady=(10, 6))

        self.anh_rong = ImageTk.PhotoImage(Image.new("RGB", (270, 200), "#2b2b2b"))
        self.nhan_anh = tk.Label(the, image=self.anh_rong, text="[ ẢNH CAMERA ]",
                                 compound="center", fg="white", font=(FONT, 11))
        self.nhan_anh.pack(padx=12, pady=6)

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
        self.nhan_anh.image = anh
        self.nhan_vai_tro.configure(text=VAI_TRO)
        self.nhan_hanh_vi.configure(text=sk["loai"])
        self.nhan_loai.configure(text=f"Loại vi phạm: {sk['loai']}")
        self.nhan_tin_cay.configure(text=f"Độ tin cậy: {sk['tin_cay']:.1%}")
        self.nhan_camera.configure(text=f"Camera: {sk['camera']}")
        self.nhan_thoi_gian.configure(text=f"Thời gian: {sk['thoi_gian']}")
        self.nut.configure(state="normal")

    # ================= MỚI: Xử lý Tải / Chuyển Đổi Nguồn Ảnh =================
    def chon_anh_file(self):
        duong_dan = filedialog.askopenfilename(
            title="Chọn ảnh cần kiểm tra",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if duong_dan:
            frame = cv2.imread(duong_dan)
            if frame is not None:
                self.anh_file_cv = frame
                self.che_do_anh = True
                self.lan_bao_cuoi = 0  # Đặt lại cooldown để ép nhận diện hình mới

    def dung_camera(self):
        self.che_do_anh = False
        self.anh_file_cv = None

    # ================= vòng lặp chính =================
    def cap_nhat(self):
        if self.che_do_anh and self.anh_file_cv is not None:
            # Sử dụng ảnh đã tải lên
            frame = self.anh_file_cv.copy()
            ok = True
        else:
            # Đọc từ Webcam
            ok, frame = self.cap.read()
            if ok and LAT_HINH is not None:
                frame = cv2.flip(frame, LAT_HINH)

        if ok:
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
        # Đối với ảnh tải lên thì bỏ qua đếm cooldown
        if not self.che_do_anh and (bay_gio - self.lan_bao_cuoi < COOLDOWN_GIAY):
            return
        self.lan_bao_cuoi = bay_gio

        r = max(vi_pham, key=lambda x: x["confidence"])
        self.su_kien = {
            "anh": frame.copy(),
            "loai": "Không thắt dây",
            "tin_cay": r["confidence"],
            "camera": "FILE_IMAGE" if self.che_do_anh else TEN_CAMERA,
            "thoi_gian": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            "debug_image": r.get("debug_image"),
        }
        self.hien_the_vi_pham(self.su_kien)
        if self.khi_co_su_kien is not None:
            self.khi_co_su_kien(self.su_kien)

    # ================= các hàm con =================
    def ve_khung(self, frame, person, result):
        x1, y1, x2, y2 = person["person_box"]
        cx1, cy1, cx2, cy2 = person["chest_box"]
        mau = XANH if result["seatbelt"] else DO
        cv2.rectangle(frame, (x1, y1), (x2, y2), mau, 2)
        cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), VANG, 2)

    def hien_thi_hinh(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        anh = thanh_photoimage(rgb, 640)  # Chuẩn hóa độ rộng hiển thị khung bên trái
        self.khung_hinh.configure(image=anh)
        self.khung_hinh.image = anh

    def hien_trang_thai(self, cac_nguoi):
        if not cac_nguoi:
            chu, mau = "● Không thấy người", MAU_XAM
        elif all(r["seatbelt"] for _, r in cac_nguoi):
            chu, mau = "● Đang thắt dây an toàn", MAU_XANH
        else:
            chu, mau = "● Đang KHÔNG thắt dây", MAU_DO
        
        if self.che_do_anh:
            chu += " (Chế độ Ảnh)"
            
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

        if sk["debug_image"] is not None:
            tk.Label(cua_so, text="Ảnh trung gian (xử lý ảnh):",
                     font=(FONT, 11, "bold")).pack(pady=(10, 0))
            dbg = thanh_photoimage(sk["debug_image"], 320)
            nd = tk.Label(cua_so, image=dbg)
            nd.image = dbg
            nd.pack(pady=(0, 10))

    def dong(self):
        self.cap.release()
        self.root.destroy()
