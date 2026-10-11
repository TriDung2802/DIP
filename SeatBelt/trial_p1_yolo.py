from pathlib import Path

import cv2
import numpy as np

THU_MUC_MODEL = Path(__file__).resolve().parent / "yolo_model"
KICH_THUOC_DAU_VAO = 416      # YOLOv4-tiny chuẩn: 416x416 (số càng lớn càng chính xác nhưng càng chậm)
NGUONG_TIN_CAY = 0.35         # bỏ qua box có độ tin cậy "là người" thấp hơn mức này
NGUONG_NMS = 0.4              # gộp các box chồng lấn cùng 1 người (Non-Max Suppression)
LOP_NGUOI = 0                 # trong coco.names, "person" luôn là dòng đầu tiên -> nhãn 0

# Vùng ngực = phần trăm CHIỀU CAO của person_box (đã cắt gọn theo khung hình).
# Webcam thường chỉ thấy từ đầu tới ngang ngực/bụng (ảnh "tự sướng"), nên vùng ngực
# nằm ở khoảng GIỮA của box, không phải sát đáy như ảnh chụp toàn thân đứng xa.
TI_LE_NGUC_TREN = 0.52
TI_LE_NGUC_DUOI = 1.00
TI_LE_NGUC_RONG = 1.00        # không thu hẹp bề ngang: box YOLO đã bám khá sát vai rồi


def _nap_model():
    duong_cfg = THU_MUC_MODEL / "yolov4-tiny.cfg"
    duong_weights = THU_MUC_MODEL / "yolov4-tiny.weights"
    if not duong_cfg.exists() or not duong_weights.exists():
        raise RuntimeError(
            f"Thiếu file model YOLO trong {THU_MUC_MODEL}. "
            "Cần có yolov4-tiny.cfg và yolov4-tiny.weights trong thư mục yolo_model/.")
    mang = cv2.dnn.readNetFromDarknet(str(duong_cfg), str(duong_weights))
    mang.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    mang.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    ten_lop = mang.getLayerNames()
    lop_ra = [ten_lop[i - 1] for i in mang.getUnconnectedOutLayers().flatten()]
    return mang, lop_ra


_MANG, _LOP_RA = _nap_model()      # nạp 1 lần khi import, không nạp lại mỗi khung hình (rất chậm nếu nạp lại)


def _chay_yolo(frame):
    """Trả về list (x1,y1,x2,y2,conf) các box 'person', đã lọc NMS và cắt gọn trong khung hình."""
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (KICH_THUOC_DAU_VAO, KICH_THUOC_DAU_VAO),
                                 swapRB=True, crop=False)
    _MANG.setInput(blob)
    cac_dau_ra = _MANG.forward(_LOP_RA)

    boxes, confs = [], []
    for out in cac_dau_ra:
        for d in out:
            diem = d[5:]
            lop = int(np.argmax(diem))
            conf = float(diem[lop])
            if lop == LOP_NGUOI and conf > NGUONG_TIN_CAY:
                cx, cy, bw, bh = d[0] * w, d[1] * h, d[2] * w, d[3] * h
                boxes.append([int(cx - bw / 2), int(cy - bh / 2), int(bw), int(bh)])
                confs.append(conf)
    if not boxes:
        return []

    idx = cv2.dnn.NMSBoxes(boxes, confs, NGUONG_TIN_CAY, NGUONG_NMS)
    ket_qua = []
    for i in np.array(idx).flatten():
        x, y, bw, bh = boxes[i]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w, x + bw), min(h, y + bh)
        if x2 > x1 and y2 > y1:
            ket_qua.append((x1, y1, x2, y2, confs[i]))
    return ket_qua


def detect_people(frame):
    h, w = frame.shape[:2]
    ket_qua = []
    for x1, y1, x2, y2, conf in _chay_yolo(frame):
        ph = y2 - y1
        giua_x = (x1 + x2) / 2
        nua_rong_nguc = (x2 - x1) * TI_LE_NGUC_RONG / 2

        cx1 = max(0, int(giua_x - nua_rong_nguc))
        cx2 = min(w, int(giua_x + nua_rong_nguc))
        cy1 = y1 + int(TI_LE_NGUC_TREN * ph)
        cy2 = y1 + int(TI_LE_NGUC_DUOI * ph)
        if cx2 - cx1 < 30 or cy2 - cy1 < 30:    # box quá nhỏ (người ở rất xa) -> bỏ qua
            continue

        ket_qua.append({
            "person_box": (x1, y1, x2, y2),
            "chest_box": (cx1, cy1, cx2, cy2),
            "chest_roi": frame[cy1:cy2, cx1:cx2].copy(),
            "do_tin_cay_p1": conf,   # thêm field phụ, không bắt buộc trong hợp đồng, chỉ để debug
        })
    return ket_qua
