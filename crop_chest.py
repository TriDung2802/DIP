import cv2
import os

def crop_chest(input_path, output_path, crop_box):
    if not os.path.exists(input_path):
        print(f"⚠️  Không tìm thấy ảnh gốc: {input_path}")
        return

    img = cv2.imread(input_path)
    if img is None:
        print(f"❌ Không thể đọc ảnh: {input_path}")
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
    print(f"✅ Đã crop thành công: {output_path}")

if __name__ == "__main__":
    # 1. dip_3.jpg (Cô gái áo xanh - Có dây an toàn):
    # Lấy từ vai xuống eo, bỏ đùi & chân váy
    crop_chest("dip_3.jpg", "test/seatbelt/seatbelt_1.jpg", (0.18, 0.72, 0.48, 0.88))

    # 2. dip_4.jpg (Bé gái áo hồng - Có dây an toàn):
    # Lấy từ cổ xuống eo, bỏ ghế tựa phía sau & chân bên dưới
    crop_chest("dip_4.jpg", "test/seatbelt/seatbelt_2.jpg", (0.42, 0.88, 0.32, 0.65))

    # 3. dip_1.jpg (2 bé áo trắng học sinh - KHÔNG dây an toàn):
    # Chỉ lấy thân trên của bé bên trái, bỏ hoàn toàn mép cổ áo & chân váy xanh
    crop_chest("dip_1.jpg", "test/no_seatbelt/no_seatbelt_1.jpg", (0.28, 0.70, 0.18, 0.58))

    # 4. dip_2.jpg (Anh nam áo trắng - KHÔNG dây an toàn):
    # Lấy vùng áo từ ngực đến eo, bỏ trần xe, kính xe & đường chéo vai tựa ghế phía sau
    crop_chest("dip_2.jpg", "test/no_seatbelt/no_seatbelt_2.jpg", (0.28, 0.62, 0.22, 0.62))