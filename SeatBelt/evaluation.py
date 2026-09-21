"""
Cách dùng:
  1. Chuẩn bị ảnh VÙNG NGỰC (đúng kiểu chest_roi mà P2 sẽ nhận):
        test/seatbelt/      <- ảnh CÓ thắt dây
        test/no_seatbelt/   <- ảnh KHÔNG thắt dây
  2. Chạy file này -> in bảng Đúng/Sai, Accuracy, và lưu ket_qua_danh_gia.csv
"""
import csv
from pathlib import Path

import cv2
import numpy as np

from mock_p2 import detect_seatbelt    # sau này: from seatbelt_detector import detect_seatbelt

THU_MUC_GOC = Path(__file__).resolve().parent
THU_MUC_TEST = THU_MUC_GOC / "test"
FILE_KET_QUA = THU_MUC_GOC / "ket_qua_danh_gia.csv"
DUOI_ANH = {".jpg", ".jpeg", ".png", ".bmp"}
CAC_LOP = {"seatbelt": True, "no_seatbelt": False}   # tên thư mục -> nhãn thật (có dây?)


def ten(co_day):
    return "Seatbelt" if co_day else "No Seatbelt"


def doc_anh(duong_dan):
    """Đọc ảnh an toàn với đường dẫn có dấu tiếng Việt. Trả về None nếu file hỏng."""
    du_lieu = np.fromfile(str(duong_dan), dtype=np.uint8)
    return cv2.imdecode(du_lieu, cv2.IMREAD_COLOR)


def danh_gia(detector, thu_muc_test=THU_MUC_TEST):
    """Chạy detector trên mọi ảnh test, trả về danh sách dòng kết quả."""
    cac_dong = []
    for ten_lop, nhan_that in CAC_LOP.items():
        thu_muc = thu_muc_test / ten_lop
        if not thu_muc.is_dir():
            continue
        for f in sorted(thu_muc.iterdir()):
            if f.suffix.lower() not in DUOI_ANH:
                continue
            dong = {"anh": f"{ten_lop}/{f.name}", "that": nhan_that,
                    "doan": None, "tin_cay": None, "loi": ""}
            try:
                anh = doc_anh(f)
                if anh is None:
                    raise ValueError("không đọc được ảnh")
                kq = detector(anh)                       # <-- gọi hàm của Đô
                dong["doan"] = bool(kq["seatbelt"])
                dong["tin_cay"] = float(kq["confidence"])
            except Exception as e:                       # 1 ảnh lỗi không được làm dừng cả bài test
                dong["loi"] = str(e)
            cac_dong.append(dong)
    return cac_dong


def tinh_chi_so(cac_dong):
    """Accuracy + ma trận nhầm lẫn. 'Dương tính' = VI PHẠM (không thắt dây)."""
    hop_le = [d for d in cac_dong if d["doan"] is not None]
    tp = sum(1 for d in hop_le if not d["that"] and not d["doan"])   # bắt đúng vi phạm
    fn = sum(1 for d in hop_le if not d["that"] and d["doan"])       # BỎ SÓT vi phạm
    fp = sum(1 for d in hop_le if d["that"] and not d["doan"])       # báo nhầm
    tn = sum(1 for d in hop_le if d["that"] and d["doan"])           # đúng: có dây
    tong = len(hop_le)
    return {
        "tong": tong, "loi": len(cac_dong) - tong,
        "dung": tp + tn,
        "accuracy": (tp + tn) / tong if tong else 0.0,
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "precision": tp / (tp + fp) if (tp + fp) else None,
        "recall": tp / (tp + fn) if (tp + fn) else None,
    }


def in_bang(cac_dong):
    print(f"{'STT':<4}{'Ảnh':<32}{'Nhãn thật':<14}{'Kết quả':<14}{'Tin cậy':<10}Đúng/Sai")
    print("-" * 82)
    for i, d in enumerate(cac_dong, start=1):
        if d["doan"] is None:
            print(f"{i:<4}{d['anh']:<32}{ten(d['that']):<14}{'(lỗi)':<14}{'-':<10}{d['loi']}")
            continue
        dung = "Đúng" if d["doan"] == d["that"] else "Sai"
        print(f"{i:<4}{d['anh']:<32}{ten(d['that']):<14}{ten(d['doan']):<14}"
              f"{d['tin_cay']:<10.2f}{dung}")


def in_chi_so(cs, cac_dong):
    pt = lambda x: "n/a" if x is None else f"{x:.1%}"
    print("\n=== KẾT QUẢ ===")
    print(f"Accuracy : {cs['dung']}/{cs['tong']} = {cs['accuracy']:.1%}")
    print(f"Bắt đúng vi phạm (TP) : {cs['tp']}   | BỎ SÓT vi phạm (FN) : {cs['fn']}")
    print(f"Báo nhầm (FP)         : {cs['fp']}   | Đúng là có dây (TN) : {cs['tn']}")
    print(f"Precision (báo vi phạm thì đúng bao nhiêu %) : {pt(cs['precision'])}")
    print(f"Recall    (bắt được bao nhiêu % vi phạm thật) : {pt(cs['recall'])}")
    if cs["loi"]:
        print(f"Có {cs['loi']} ảnh bị lỗi, không tính vào kết quả.")
    if cs["tong"] < 20:
        print(f"Lưu ý: mới có {cs['tong']} ảnh, quá ít để kết luận. Nên có ít nhất 20-40 ảnh.")
    sai = [d["anh"] for d in cac_dong if d["doan"] is not None and d["doan"] != d["that"]]
    if sai:
        print("\nẢnh bị đoán SAI (gửi cho Đô để chỉnh thuật toán):")
        for a in sai:
            print("  -", a)


def luu_csv(cac_dong, duong_dan=FILE_KET_QUA):
    with open(duong_dan, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["anh", "nhan_that", "ket_qua", "tin_cay", "dung_sai"])
        for d in cac_dong:
            if d["doan"] is None:
                w.writerow([d["anh"], ten(d["that"]), "loi", "", d["loi"]])
            else:
                w.writerow([d["anh"], ten(d["that"]), ten(d["doan"]), f"{d['tin_cay']:.4f}",
                            "Đúng" if d["doan"] == d["that"] else "Sai"])


def main():
    thieu = [t for t in CAC_LOP if not (THU_MUC_TEST / t).is_dir()]
    if thieu:
        for t in CAC_LOP:
            (THU_MUC_TEST / t).mkdir(parents=True, exist_ok=True)
        print(f"Đã tạo thư mục test tại: {THU_MUC_TEST}")
        print("Hãy bỏ ảnh vùng ngực vào:")
        print("  test/seatbelt/     (có thắt dây)")
        print("  test/no_seatbelt/  (không thắt dây)")
        print("rồi chạy lại file này.")
        return

    cac_dong = danh_gia(detect_seatbelt)
    if not cac_dong:
        print("Chưa có ảnh nào trong test/seatbelt và test/no_seatbelt.")
        return

    in_bang(cac_dong)
    in_chi_so(tinh_chi_so(cac_dong), cac_dong)
    luu_csv(cac_dong)
    print(f"\nĐã lưu bảng kết quả: {FILE_KET_QUA}")


if __name__ == "__main__":
    main()
