"""
Phím tắt:
  s = lưu ảnh hiện tại vào test/seatbelt/      (đang CÓ thắt dây)
  n = lưu ảnh hiện tại vào test/no_seatbelt/   (đang KHÔNG thắt dây)
  q = thoát
"""
import time
from pathlib import Path

import cv2

from trial_p1 import detect_people   # dùng lại đúng khâu cắt vùng ngực đang có

THU_MUC_TEST = Path(__file__).resolve().parent / "test"
LAT_HINH = 1
XANH, VANG, TRANG = (0, 200, 0), (0, 220, 220), (255, 255, 255)


def luu_anh(roi, thu_muc, tien_to):
    thu_muc.mkdir(parents=True, exist_ok=True)
    ok, bo_dem = cv2.imencode(".jpg", roi)
    if not ok:
        print("Lỗi: không mã hóa được ảnh, thử lại.")
        return None

    # Bấm phím 2 lần trong cùng 1 giây thì tên file trùng nhau -> ảnh sau ĐÈ LÊN ảnh trước
    # mà không báo lỗi gì. Vì vậy nếu tên đã tồn tại, thêm số đếm (_2, _3, ...) vào sau.
    goc = f"{tien_to}_{time.strftime('%Y%m%d_%H%M%S')}"
    ten = f"{goc}.jpg"
    dem = 1
    while (thu_muc / ten).exists():
        dem += 1
        ten = f"{goc}_{dem}.jpg"

    (thu_muc / ten).write_bytes(bo_dem.tobytes())
    return ten


def dem_anh(thu_muc):
    return len(list(thu_muc.glob("*.jpg"))) if thu_muc.is_dir() else 0


def ve_chu_nen(frame, dong, y, mau=TRANG):
    cv2.putText(frame, dong, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)  # viền đen cho dễ đọc
    cv2.putText(frame, dong, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mau, 1)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không mở được camera. Thử đổi số 0 thành 1.")
        return

    thong_bao, het_han_thong_bao = "", 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if LAT_HINH is not None:
            frame = cv2.flip(frame, LAT_HINH)

        cac_nguoi = detect_people(frame)
        for n in cac_nguoi:
            x1, y1, x2, y2 = n["chest_box"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), VANG, 2)

        so_co_day = dem_anh(THU_MUC_TEST / "seatbelt")
        so_khong_day = dem_anh(THU_MUC_TEST / "no_seatbelt")
        ve_chu_nen(frame, f"seatbelt: {so_co_day}   no_seatbelt: {so_khong_day}", 25)
        ve_chu_nen(frame, "s = luu CO day | n = luu KHONG day | q = thoat", 50)
        if not cac_nguoi:
            ve_chu_nen(frame, "Khong thay nguoi - chinh lai tu the / anh sang", 75, (0, 0, 255))
        if thong_bao and time.time() < het_han_thong_bao:
            ve_chu_nen(frame, thong_bao, frame.shape[0] - 15, XANH)

        cv2.imshow("Gom anh test - Seatbelt AI", frame)
        phim = cv2.waitKey(1) & 0xFF

        if phim == ord("q"):
            break
        elif phim in (ord("s"), ord("n")) and not cac_nguoi:
            thong_bao, het_han_thong_bao = "Chua thay ROI de luu!", time.time() + 1.5
        elif phim == ord("s"):
            ten = luu_anh(cac_nguoi[0]["chest_roi"], THU_MUC_TEST / "seatbelt", "co_day")
            if ten:
                thong_bao, het_han_thong_bao = f"Da luu (CO day): {ten}", time.time() + 1.5
        elif phim == ord("n"):
            ten = luu_anh(cac_nguoi[0]["chest_roi"], THU_MUC_TEST / "no_seatbelt", "khong_day")
            if ten:
                thong_bao, het_han_thong_bao = f"Da luu (KHONG day): {ten}", time.time() + 1.5

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nTổng kết: {dem_anh(THU_MUC_TEST/'seatbelt')} ảnh seatbelt, "
          f"{dem_anh(THU_MUC_TEST/'no_seatbelt')} ảnh no_seatbelt trong {THU_MUC_TEST}")


if __name__ == "__main__":
    main()
