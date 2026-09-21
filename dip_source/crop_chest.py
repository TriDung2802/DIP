import cv2
import os

def crop_chest(input_path, output_path, crop_box):
    if not os.path.exists(input_path):
        print(f"Không tìm thấy ảnh gốc: {input_path}")
        return

    img = cv2.imread(input_path)
    if img is None:
        print(f"Không thể đọc ảnh: {input_path}")
        return

    h, w = img.shape[:2]
    y_min, y_max, x_min, x_max = crop_box

    ymin_p = int(y_min * h)
    ymax_p = int(y_max * h)
    xmin_p = int(x_min * w)
    xmax_p = int(x_max * w)

    roi = img[ymin_p:ymax_p, xmin_p:xmax_p]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, roi)
    print(f"Đã crop thành công: {output_path}")

if __name__ == "__main__":
    
    crop_chest("dip_3.jpg", "test/seatbelt/seatbelt_1.jpg", (0.18, 0.72, 0.48, 0.88))

   
    crop_chest("dip_4.jpg", "test/seatbelt/seatbelt_2.jpg", (0.42, 0.88, 0.32, 0.65))

   
    crop_chest("dip_1.jpg", "test/no_seatbelt/no_seatbelt_1.jpg", (0.28, 0.70, 0.18, 0.58))

   
    crop_chest("dip_2.jpg", "test/no_seatbelt/no_seatbelt_2.jpg", (0.28, 0.62, 0.22, 0.62))