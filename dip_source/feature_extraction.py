import cv2
import numpy as np

def seatbelt_line_extraction(binary_img, min_line_length=20, max_line_gap=10):
    if binary_img is None:
        return []

    lines = cv2.HoughLinesP(
        binary_img, 
        rho=1, 
        theta=np.pi/180, 
        threshold=30, 
        minLineLength=min_line_length, 
        maxLineGap=max_line_gap
    )
    valid_lines = []

    if lines is not None:
        for line in lines:
            if line.ndim > 1:
                x1, y1, x2, y2 = line[0]
            else:
                x1, y1, x2, y2 = line

            angle_calc = np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1)))

            if 25 <= angle_calc <= 65:
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

                valid_lines.append({
                    "coords": (x1, y1, x2, y2),  # Đã sửa "coord" thành "coords"
                    "angle": angle_calc,
                    "length": length
                })
                
    return valid_lines