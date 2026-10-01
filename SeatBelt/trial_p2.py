"""
trial_p2.py - BẢN THỬ của khâu P2 (DIP + nhận diện dây an toàn).

Dùng để chạy thử toàn hệ thống trước khi có bản chính thức của Đô. Giữ nguyên
"hợp đồng dữ liệu", nên khi Đô xong chỉ cần đổi 1 dòng import trong main.py.

Pipeline (đúng gợi ý trong file phân công):
  chest_roi -> Grayscale -> Gaussian blur -> Canny -> Morphology (closing)
            -> HoughLinesP -> lọc theo GÓC + ĐỘ DÀI -> tìm CẶP CẠNH SONG SONG -> quyết định

Ý tưởng chính: một sợi dây an toàn có 2 mép song song, chéo, dài. Một đường kẻ
mảnh, đường viền cổ áo (ngang/dọc) hay nếp gấp ngắn đều KHÔNG thỏa cả 3 điều kiện.
"""
import math

import cv2
import numpy as np

# ====== CÁC "NÚM CHỈNH" - đổi số ở đây để thử nghiệm ======
RONG_CHUAN = 320                       # thu ROI về rộng 320px để số liệu không phụ thuộc kích thước ảnh
BLUR_KICH_THUOC = (5, 5)
CANNY_THAP, CANNY_CAO = 50, 150
GOC_MIN, GOC_MAX = 25, 75              # độ, so với phương ngang: chỉ nhận đường CHÉO
DO_DAI_MIN = 0.45                      # đoạn thẳng phải dài >= 45% chiều cao ROI
BE_RONG_MIN, BE_RONG_MAX = 0.03, 0.25  # khoảng cách giữa 2 mép dây / chiều rộng ROI
LECH_GOC_TOI_DA = 10                   # 2 mép phải song song (lệch <= 10 độ)
CHONG_LAN_MIN = 0.5                    # 2 mép phải nằm cạnh nhau >= 50% chiều dài
NGUONG_QUYET_DINH = 0.5                # điểm tối thiểu để kết luận "có dây"

# --- Bằng chứng thứ 2: DẢI TỐI (threshold + morphology + contour) ---
SO_OTSU = 0.65                         # ngưỡng "tối" = 0.65 x ngưỡng Otsu (để tách dây đen khỏi nền xám)
DIEN_TICH_MIN = 0.03                   # vùng tối phải chiếm >= 3% diện tích ROI (bỏ nhiễu)
TI_LE_DAI_MIN = 1.8                    # dài / rộng >= 1.8 (phải "thon dài", không phải mảng tròn)
BE_RONG_TOI_MAX = 0.30                 # bề rộng dải tối <= 30% chiều rộng ROI
DAI_TOI_MIN = 0.8                      # chiều dài dải >= 0.8 x chiều cao ROI


def tien_xu_ly(roi):
    """Bước 1-4: xám -> mờ -> cạnh -> nối cạnh đứt. Trả về (ảnh nhỏ màu, ảnh cạnh)."""
    if roi.ndim == 2:
        roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    h, w = roi.shape[:2]
    nho = cv2.resize(roi, (RONG_CHUAN, max(1, int(h * RONG_CHUAN / w))))
    xam = cv2.cvtColor(nho, cv2.COLOR_BGR2GRAY)
    mo = cv2.GaussianBlur(xam, BLUR_KICH_THUOC, 0)
    canh = cv2.Canny(mo, CANNY_THAP, CANNY_CAO)
    canh = cv2.morphologyEx(canh, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return nho, canh


def tim_doan_thang(canh):
    """Bước 5: Hough tìm các đoạn thẳng, kèm độ dài và góc (0..180 độ)."""
    cao = canh.shape[0]
    doan = cv2.HoughLinesP(canh, 1, np.pi / 180, threshold=40,
                           minLineLength=int(DO_DAI_MIN * cao), maxLineGap=12)
    ket_qua = []
    if doan is None:
        return ket_qua
    # OpenCV 4 trả mảng (N,1,4), OpenCV 5 trả (N,4) -> ép về (N,4) để chạy được cả hai
    for x1, y1, x2, y2 in np.asarray(doan).reshape(-1, 4):
        ket_qua.append({
            "p": (int(x1), int(y1), int(x2), int(y2)),
            "dai": math.hypot(x2 - x1, y2 - y1),
            "goc": math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180,
        })
    return ket_qua


def la_cheo(d):
    g = d["goc"]   # cho phép cả 2 chiều nghiêng (camera có thể bị lật gương)
    return GOC_MIN <= g <= GOC_MAX or 180 - GOC_MAX <= g <= 180 - GOC_MIN


def tim_cap_mep_day(cac_doan, rong, cao):
    """Bước 6: tìm 2 đoạn song song, cách nhau bề rộng hợp lý, nằm cạnh nhau. Trả về (điểm, a, b)."""
    tot_nhat = None
    for i in range(len(cac_doan)):
        for j in range(i + 1, len(cac_doan)):
            a, b = cac_doan[i], cac_doan[j]
            lech = abs(a["goc"] - b["goc"])
            lech = min(lech, 180 - lech)
            if lech > LECH_GOC_TOI_DA:
                continue
            ax1, ay1, ax2, ay2 = a["p"]
            ux, uy = (ax2 - ax1) / a["dai"], (ay2 - ay1) / a["dai"]   # hướng dọc dây
            bx, by = (b["p"][0] + b["p"][2]) / 2, (b["p"][1] + b["p"][3]) / 2
            khoang_cach = abs((bx - ax1) * uy - (by - ay1) * ux) / rong   # cách nhau bao nhiêu
            if not (BE_RONG_MIN <= khoang_cach <= BE_RONG_MAX):
                continue
            ta = sorted([0.0, (ax2 - ax1) * ux + (ay2 - ay1) * uy])
            tb = sorted([(b["p"][0] - ax1) * ux + (b["p"][1] - ay1) * uy,
                         (b["p"][2] - ax1) * ux + (b["p"][3] - ay1) * uy])
            chong = max(0.0, min(ta[1], tb[1]) - max(ta[0], tb[0])) / min(a["dai"], b["dai"])
            if chong < CHONG_LAN_MIN:
                continue
            diem = (0.5 * min(1.0, (a["dai"] + b["dai"]) / 2 / cao)   # dài
                    + 0.3 * min(1.0, chong)                           # nằm cạnh nhau
                    + 0.2 * (1 - lech / LECH_GOC_TOI_DA))             # song song
            if tot_nhat is None or diem > tot_nhat[0]:
                tot_nhat = (diem, a, b)
    return tot_nhat


def tim_dai_toi(nho):
    """Bằng chứng 2: tách vùng TỐI dạng dải chéo, dài. Trả về (điểm, hộp xoay, ảnh mặt nạ) hoặc (0, None, mặt nạ)."""
    cao, rong = nho.shape[:2]
    xam = cv2.GaussianBlur(cv2.cvtColor(nho, cv2.COLOR_BGR2GRAY), BLUR_KICH_THUOC, 0)
    t_otsu, _ = cv2.threshold(xam, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, mat_na = cv2.threshold(xam, t_otsu * SO_OTSU, 255, cv2.THRESH_BINARY_INV)
    mat_na = cv2.morphologyEx(mat_na, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    mat_na = cv2.morphologyEx(mat_na, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))

    tot_nhat = (0.0, None)
    cac_vung, _ = cv2.findContours(mat_na, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cac_vung:
        if cv2.contourArea(c) < DIEN_TICH_MIN * cao * rong:
            continue
        hop = cv2.minAreaRect(c)
        (_, _), (w, h), ang = hop
        dai, ngan = max(w, h), max(min(w, h), 1.0)
        goc = (ang if w >= h else ang + 90) % 180
        if dai / ngan < TI_LE_DAI_MIN or dai < DAI_TOI_MIN * cao:
            continue
        if not la_cheo({"goc": goc}) or not (BE_RONG_MIN <= ngan / rong <= BE_RONG_TOI_MAX):
            continue
        diem = 0.6 * min(1.0, dai / (1.3 * cao)) + 0.4 * min(1.0, dai / ngan / 4)
        if diem > tot_nhat[0]:
            tot_nhat = (diem, hop)
    return tot_nhat[0], tot_nhat[1], mat_na


def ve_debug(nho, canh, moi_doan, cheo, cap, mat_na, hop_toi):
    """Ảnh trung gian để 'chứng minh DIP': trên = ảnh gốc, giữa = cạnh + đoạn thẳng, dưới = mặt nạ vùng tối."""
    tren = nho.copy()
    duoi = cv2.cvtColor(canh, cv2.COLOR_GRAY2BGR)
    for d in moi_doan:
        x1, y1, x2, y2 = d["p"]
        cv2.line(duoi, (x1, y1), (x2, y2), (255, 120, 0), 1)       # xanh dương: mọi đoạn Hough
    for d in cheo:
        x1, y1, x2, y2 = d["p"]
        cv2.line(duoi, (x1, y1), (x2, y2), (0, 220, 220), 2)       # vàng: đoạn chéo đủ dài
    if cap:
        for d in (cap[1], cap[2]):
            x1, y1, x2, y2 = d["p"]
            cv2.line(duoi, (x1, y1), (x2, y2), (0, 220, 0), 3)     # xanh lá: cặp mép dây
            cv2.line(tren, (x1, y1), (x2, y2), (0, 220, 0), 2)
    panel_toi = cv2.cvtColor(mat_na, cv2.COLOR_GRAY2BGR)
    if hop_toi is not None:
        khung = np.intp(cv2.boxPoints(hop_toi))
        cv2.polylines(panel_toi, [khung], True, (0, 220, 0), 2)
        cv2.polylines(tren, [khung], True, (0, 220, 0), 2)
    return np.vstack([tren, duoi, panel_toi])


def detect_seatbelt(chest_roi):
    """HỢP ĐỒNG: nhận chest_roi -> {"seatbelt": bool, "confidence": 0..1, "debug_image": ảnh}"""
    if chest_roi is None or chest_roi.size == 0 or min(chest_roi.shape[:2]) < 20:
        return {"seatbelt": False, "confidence": 0.0, "debug_image": None}

    nho, canh = tien_xu_ly(chest_roi)
    cao, rong = canh.shape
    moi_doan = tim_doan_thang(canh)
    cheo = [d for d in moi_doan if la_cheo(d)]
    cap = tim_cap_mep_day(cheo, rong, cao)

    diem_toi, hop_toi, mat_na = tim_dai_toi(nho)
    diem_mep = cap[0] if cap is not None else 0.0

    diem = max(diem_mep, diem_toi)            # chỉ cần 1 trong 2 bằng chứng đủ mạnh
    co_day = diem >= NGUONG_QUYET_DINH
    if co_day:
        tin_cay = 0.5 + 0.5 * diem
    else:
        tin_cay = 0.70 if cheo else 0.90     # thấy đường chéo nhưng không thành dây -> kém chắc hơn
    return {"seatbelt": bool(co_day), "confidence": float(tin_cay),
            "debug_image": ve_debug(nho, canh, moi_doan, cheo,
                                    cap if diem_mep >= NGUONG_QUYET_DINH else None,
                                    mat_na, hop_toi if diem_toi >= NGUONG_QUYET_DINH else None)}
