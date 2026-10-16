import cv2
import numpy as np
import math

def calculate_angle(x1, y1, x2, y2):
    angle = math.degrees(
        math.atan2(y2 - y1, x2 - x1)
    )

    angle = abs(angle)

    if angle > 90:
        angle = 180 - angle

    return angle

def calculate_length(x1, y1, x2, y2):
    return math.sqrt(
        (x2 - x1) ** 2 +
        (y2 - y1) ** 2
    )

def line_distance(line1, line2):
    """
    Khoảng cách giữa 2 đường gần song song.
    """
    x1, y1, x2, y2 = line1
    a1, b1, a2, b2 = line2

    # Trung điểm của line 1
    mx = (x1 + x2) / 2
    my = (y1 + y2) / 2

    # Vector của line 2
    dx = a2 - a1
    dy = b2 - b1

    length = math.sqrt(dx * dx +dy * dy)

    if length == 0:
        return 9999

    distance = abs(dy * mx - dx * my + a2 * b1 - b2 * a1) / length

    return distance


def seatbelt_line_extraction(binary_img):

    height, width = binary_img.shape[:2]

    # HOUGH
    lines = cv2.HoughLinesP(
        binary_img,
        rho=1,
        theta=np.pi / 180,
        threshold=40,
        minLineLength=int(width * 0.30),
        maxLineGap=30
    )

    if lines is None:
        return []

    valid_lines = []

    # LỌC LINE

    for item in lines:

        item = np.asarray(item).reshape(-1)
        if len(item) != 4:
            continue
        x1, y1, x2, y2 = map(int, item)
        angle = calculate_angle(x1,y1,x2,y2)
        length = calculate_length(x1,y1,x2,y2)

        # Chỉ lấy đường chéo
        if angle < 25:
            continue

        if angle > 70:
            continue

        # Chỉ lấy đường đủ dài
        if length < width * 0.30:
            continue

        valid_lines.append({
            "coords": (x1,y1,x2,y2),
            "angle": angle,
            "length": length
        })

    # TÌM CẶP ĐƯỜNG
    best_pair = None
    best_score = -1

    for i in range(len(valid_lines)):

        for j in range(i + 1, len(valid_lines)):

            line1 = valid_lines[i]
            line2 = valid_lines[j]

            # Góc phải gần nhau
            angle_diff = abs(line1["angle"] - line2["angle"])

            if angle_diff > 8:
                continue

            # Khoảng cách 2 đường
            distance = line_distance(
                line1["coords"],
                line2["coords"]
            )

            # Seatbelt không quá hẹp
            if distance < width * 0.04:
                continue

            # Seatbelt không quá rộng
            if distance > width * 0.30:
                continue

            # SCORE
            length_score = (line1["length"]+line2["length"]) / (2 * width)

            length_score = min(length_score, 1.0)

            parallel_score = (1 -angle_diff / 8)

            score = (0.65 * length_score+0.35 * parallel_score)

            if score > best_score:
                best_score = score
                best_pair = (line1, line2)

    # CHỈ TRẢ VỀ 2 BIÊN TỐT NHẤT
    if best_pair is None:
        return []

    return [best_pair[0], best_pair[1]]