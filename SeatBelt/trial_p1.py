import cv2

TI_LE_NGUC = dict(
    trai_phai=0.9,      # ngực rộng thêm 0.9 lần bề rộng mặt về mỗi bên
    tren=1.15,          # bắt đầu ngay dưới cằm (1.15 lần chiều cao mặt kể từ đỉnh mặt)
    duoi=3.0,           # kết thúc ở 3 lần chiều cao mặt
)


def _tao_bo_do_mat():
    if not hasattr(cv2, "CascadeClassifier") or not hasattr(cv2, "data"):
        raise RuntimeError(
            "OpenCV %s không có bộ dò khuôn mặt (OpenCV 5 đã bỏ). "
            "Chạy: py -3.13 -m pip install \"opencv-python<5\"" % cv2.__version__)
    bo = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if bo.empty():
        raise RuntimeError("Không nạp được file haarcascade_frontalface_default.xml")
    return bo


_BO_DO_MAT = _tao_bo_do_mat()     # tạo ngay khi import: lỗi thì báo lúc khởi động, không phải giữa chừng


def _tim_mat(xam, kich_thuoc_toi_thieu):
    return _BO_DO_MAT.detectMultiScale(xam, scaleFactor=1.1, minNeighbors=5,
                                       minSize=(kich_thuoc_toi_thieu, kich_thuoc_toi_thieu))


def detect_people(frame):
    h, w = frame.shape[:2]
    kich_thuoc_toi_thieu = max(40, w // 14)   # mặt phải chiếm ít nhất khoảng này

    xam_thuong = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cac_mat = _tim_mat(xam_thuong, kich_thuoc_toi_thieu)
    if len(cac_mat) == 0:
        # Ảnh thiếu sáng (ngược sáng, tối) -> thử lại với ảnh đã cân bằng độ tương phản.
        # LƯU Ý: chỉ dùng equalizeHist làm PHƯƠNG ÁN DỰ PHÒNG, không dùng mặc định,
        # vì ở phòng có nền sáng (bảng trắng, cửa sổ) nó từng làm co nhỏ khung mặt
        # và khiến khung bị loại nhầm ở đúng ngưỡng minSize (đã kiểm chứng bằng ảnh thật).
        xam_can_bang = cv2.equalizeHist(xam_thuong)
        cac_mat = _tim_mat(xam_can_bang, kich_thuoc_toi_thieu)
    ket_qua = []
    for fx, fy, fw, fh in cac_mat:
        person_box = (max(0, int(fx - 1.2 * fw)), max(0, int(fy - 0.3 * fh)),
                      min(w, int(fx + 2.2 * fw)), min(h, int(fy + 4.0 * fh)))

        r = TI_LE_NGUC
        cx1, cx2 = max(0, int(fx - r["trai_phai"] * fw)), min(w, int(fx + fw + r["trai_phai"] * fw))
        cy1, cy2 = int(fy + r["tren"] * fh), min(h, int(fy + r["duoi"] * fh))
        if cx2 - cx1 < 40 or cy2 - cy1 < 40:     # mặt sát đáy khung hình -> không còn chỗ cho ngực
            continue
        ket_qua.append({
            "person_box": person_box,
            "chest_box": (cx1, cy1, cx2, cy2),
            "chest_roi": frame[cy1:cy2, cx1:cx2].copy(),
        })
    return ket_qua
