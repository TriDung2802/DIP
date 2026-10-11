import sys
import traceback

import cv2
import numpy as np

from mock_p1 import detect_people      # đổi thành: from trial_p1 import detect_people
from mock_p2 import detect_seatbelt    # đổi thành: from trial_p2 import detect_seatbelt

SO_CA_DUNG = 0
SO_CA_SAI = 0


def kiem_tra(mo_ta, dieu_kien, ghi_chu=""):
    """1 khối kiểm tra: in OK/FAIL, không cho phép Exception làm dừng cả bài test."""
    global SO_CA_DUNG, SO_CA_SAI
    try:
        ok = bool(dieu_kien)
    except Exception:
        ok = False
        ghi_chu = f"Ném lỗi: {traceback.format_exc(limit=1).strip().splitlines()[-1]}"
    if ok:
        SO_CA_DUNG += 1
        print(f"OK   | {mo_ta}")
    else:
        SO_CA_SAI += 1
        print(f"FAIL | {mo_ta}" + (f"  ({ghi_chu})" if ghi_chu else ""))


def anh_gia(mau=120, h=240, w=320):
    return np.full((h, w, 3), mau, np.uint8)


# ========== NHÓM 1: hợp đồng dữ liệu của P1 ==========
def test_p1():
    print("\n--- P1: detect_people(frame) ---")
    frame = anh_gia()
    try:
        ra = detect_people(frame)
    except Exception:
        kiem_tra("detect_people chạy được với ảnh bình thường", False, traceback.format_exc(limit=1))
        return
    kiem_tra("detect_people trả về list", isinstance(ra, list))
    if not ra:
        print("     (không có ai trong ảnh giả - bỏ qua kiểm tra từng người)")
        return

    nguoi = ra[0]
    kiem_tra("mỗi người là dict", isinstance(nguoi, dict))
    for khoa in ("person_box", "chest_box", "chest_roi"):
        kiem_tra(f'có khóa "{khoa}"', khoa in nguoi)

    if "person_box" in nguoi:
        kiem_tra("person_box có đúng 4 số (x1,y1,x2,y2)", len(nguoi["person_box"]) == 4)
    if "chest_box" in nguoi:
        kiem_tra("chest_box có đúng 4 số (x1,y1,x2,y2)", len(nguoi["chest_box"]) == 4)
    if "chest_roi" in nguoi:
        roi = nguoi["chest_roi"]
        kiem_tra("chest_roi là ảnh numpy (có .shape)", hasattr(roi, "shape"))
        kiem_tra("chest_roi không rỗng (rộng & cao > 0)",
                 hasattr(roi, "shape") and roi.shape[0] > 0 and roi.shape[1] > 0)
        if hasattr(roi, "shape") and "chest_box" in nguoi:
            cx1, cy1, cx2, cy2 = nguoi["chest_box"]
            kiem_tra("kích thước chest_roi khớp với chest_box",
                     roi.shape[0] == cy2 - cy1 and roi.shape[1] == cx2 - cx1,
                     f"roi={roi.shape[:2]} nhưng box=({cy2-cy1},{cx2-cx1})")


# ========== NHÓM 2: hợp đồng dữ liệu của P2 ==========
def test_p2():
    print("\n--- P2: detect_seatbelt(chest_roi) ---")
    roi = anh_gia(h=150, w=250)
    try:
        kq = detect_seatbelt(roi)
    except Exception:
        kiem_tra("detect_seatbelt chạy được với ROI bình thường", False, traceback.format_exc(limit=1))
        return
    kiem_tra("kết quả là dict", isinstance(kq, dict))
    for khoa in ("seatbelt", "confidence", "debug_image"):
        kiem_tra(f'có khóa "{khoa}"', khoa in kq)

    if "seatbelt" in kq:
        # bool(1) == True nên phải chặn thêm kiểu int để không lọt qua nhầm 1/0
        dung_kieu = isinstance(kq["seatbelt"], bool) and not isinstance(kq["seatbelt"], int) or isinstance(kq["seatbelt"], bool)
        kiem_tra("seatbelt là True/False (kiểu bool), không phải 1/0 hay chuỗi",
                 isinstance(kq["seatbelt"], bool), f"kiểu thực tế: {type(kq['seatbelt']).__name__}")
    if "confidence" in kq:
        c = kq["confidence"]
        kiem_tra("confidence là số", isinstance(c, (int, float)))
        kiem_tra("confidence nằm trong khoảng 0..1", isinstance(c, (int, float)) and 0.0 <= c <= 1.0,
                 f"giá trị thực tế: {c}")


# ========== NHÓM 3: chạy nối tiếp P1 -> P2 (giống pipeline() của main.py) ==========
def test_ghep_noi():
    print("\n--- Ghép nối P1 -> P2 (giống pipeline trong main.py) ---")
    try:
        for nguoi in detect_people(anh_gia()):
            detect_seatbelt(nguoi["chest_roi"])
        kiem_tra("chạy hết pipeline không lỗi (có người trong ảnh)", True)
    except Exception:
        kiem_tra("chạy hết pipeline không lỗi (có người trong ảnh)", False, traceback.format_exc(limit=1))


# ========== NHÓM 4: các trường hợp biên - hay bị bỏ sót nhất ==========
def test_truong_hop_bien():
    print("\n--- Trường hợp biên (hay gây lỗi khi demo thật) ---")

    kiem_tra("P1 không lỗi với khung hình HOÀN TOÀN ĐEN (không có người)",
             _thu(lambda: detect_people(np.zeros((240, 320, 3), np.uint8))))

    kiem_tra("P1 không lỗi với khung hình rất nhỏ (60x80, camera lỗi)",
             _thu(lambda: detect_people(anh_gia(h=60, w=80))))

    kiem_tra("P2 không lỗi với ROI rất nhỏ (5x5px, do P1 cắt lố)",
             _thu(lambda: detect_seatbelt(np.zeros((5, 5, 3), np.uint8))))

    def hai_nguoi():
        frame = anh_gia()
        for nguoi in detect_people(frame) * 2:      # giả lập khung hình có 2 người
            detect_seatbelt(nguoi["chest_roi"])
    kiem_tra("Hệ thống chạy được khi có NHIỀU NGƯỜI trong khung hình", _thu(hai_nguoi))


def _thu(ham):
    try:
        ham(); return True
    except Exception:
        print("     " + traceback.format_exc(limit=1).strip().splitlines()[-1])
        return False


def main():
    print(f"Đang kiểm tra module: P1={detect_people.__module__} | P2={detect_seatbelt.__module__}")
    print(f"OpenCV {cv2.__version__}")
    test_p1()
    test_p2()
    test_ghep_noi()
    test_truong_hop_bien()

    print(f"\n=== KẾT QUẢ: {SO_CA_DUNG}/{SO_CA_DUNG + SO_CA_SAI} kiểm tra đạt ===")
    if SO_CA_SAI:
        print("CÒN LỖI HỢP ĐỒNG DỮ LIỆU - báo ngay cho người phụ trách module liên quan, đừng để tới lúc demo.")
        sys.exit(1)
    print("Hợp đồng dữ liệu giữa các module đang ổn.")


if __name__ == "__main__":
    main()
