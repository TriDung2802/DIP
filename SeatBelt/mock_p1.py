def detect_people(frame):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = int(w * 0.30), int(h * 0.10), int(w * 0.70), int(h * 0.95)
    person_box = (x1, y1, x2, y2)

    # Giả vờ "ROI" cắt vùng ngực: từ 20% đến 55% chiều cao người
    person_h = y2 - y1
    cx1, cx2 = x1, x2
    cy1 = y1 + int(person_h * 0.20)
    cy2 = y1 + int(person_h * 0.55)
    chest_box = (cx1, cy1, cx2, cy2)

    chest_roi = frame[cy1:cy2, cx1:cx2]  # cắt ảnh bằng slicing của numpy

    return [{"person_box": person_box, "chest_box": chest_box, "chest_roi": chest_roi}]
