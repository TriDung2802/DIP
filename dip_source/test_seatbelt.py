import os
import cv2
from dip_and_seatbelt_detection.dip_source.seatbelt_detector import Seatbelt_Detector
from dip_and_seatbelt_detection.dip_source.image_processing import preprocessing_roi
def dataset_evaluation(test_dir="test"):

    detector = Seatbelt_Detector(min_confidence=0.65)

    categories = {
        "seatbelt": True,      
        "no_seatbelt": False   
    }

    total_images = 0
    correct_predictions = 0
    
    print("=" * 60)
    print(f"{'IMAGE NAME':<20} | {'GROUND TRUTH':<12} | {'PREDICTED':<12} | {'RESULT'}")
    print("=" * 60)

    for category, ground_truth in categories.items():
        folder_path = os.path.join(test_dir, category)
        
        if not os.path.exists(folder_path):
            print(f"Cảnh báo: Không tìm thấy thư mục {folder_path}")
            continue

        for img_name in os.listdir(folder_path):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue
                
            img_path = os.path.join(folder_path, img_name)
            chest_roi = cv2.imread(img_path)
            
            if chest_roi is None:
                print(f"Không thể đọc ảnh: {img_path}")
                continue

            result = detector.detect(chest_roi)
            predicted = result["seatbelt"]
            confidence = result["confidence"]

            is_correct = (predicted == ground_truth)
            total_images += 1
            if is_correct:
                correct_predictions += 1

            gt_str = "Seatbelt" if ground_truth else "No Seatbelt"
            pred_str = "Seatbelt" if predicted else "No Seatbelt"
            status_str = "Đúng" if is_correct else "Sai"

            print(f"{img_name:<20} | {gt_str:<12} | {pred_str:<12} | {status_str} (Conf: {confidence})")

            cv2.imshow("Debug ROI", result["debug_image"])
            cv2.waitKey(0)

    print("=" * 60)
    if total_images > 0:
        accuracy = (correct_predictions / total_images) * 100
        print(f"Tổng số ảnh test: {total_images}")
        print(f"Số dự đoán đúng : {correct_predictions}")
        print(f" Accuracy        : {accuracy:.2f}%")
    else:
        print("Không tìm thấy dữ liệu ảnh test để đánh giá.")

if __name__ == "__main__":
  
    dataset_evaluation(test_dir="test")


