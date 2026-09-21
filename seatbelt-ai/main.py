import cv2
from ultralytics import YOLO

# Load YOLO
model = YOLO("yolo11n.pt")

# Mở camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Không mở được camera!")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Không lấy được hình ảnh từ camera!")
        break

    # YOLO nhận diện
    results = model(frame, verbose=False)
    result=results[0]

    # Lấy ảnh kết quả
    annotated_frame=frame.copy()
    for box in result.boxes:
        
        class_id=int(box.cls[0])
        confidence=float(box.conf[0])

        if class_id==0:
            x1,y1,x2,y2=map(int,box.xyxy[0])
            cv2.rectangle(
                annotated_frame,
                (x1,y1),
                (x2,y2),
                (0,255,0),
                2
            )
            cv2.putText(
                annotated_frame,
                f"person {confidence:.2f}",
                (x1,y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,255,0),
                2
            )

    # Hiển thị
    cv2.imshow("AI CAMERA DETECTION", annotated_frame)

    # Nhấn Q để thoát
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()