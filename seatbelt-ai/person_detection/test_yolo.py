import cv2
import os
from camera import open_camera
from yolo_detector import detect_person
from roi import get_chest_roi

cap=open_camera()
last_chest_roi=None

save_dir = "dataset/chest_roi"
os.makedirs(save_dir,exist_ok=True)
image_count = 0

#Tìm số ảnh lớn nhất đã có
existing_files = os.listdir(save_dir)
numbers = []
for file in existing_files:
    if file.startswith("chest_") and file.endswith(".jpg"):
        number=file.replace("chest_","").replace(".jpg","")
        if number.isdigit():
            numbers.append(int(number))

if len(numbers)>0:
    image_count = max(numbers)
else:
    image_count = 0
while True:
    ret,frame=cap.read()
    if not ret:
        break
    persons = detect_person(frame)
    for person in persons:
        x1,y1,x2,y2 = person["person_box"]
        chest_roi,chest_box=get_chest_roi(frame,person["person_box"])

        #Lưu ROI
        last_chest_roi=chest_roi

        cx1,cy1,cx2,cy2=chest_box

        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.rectangle(frame, (cx1,cy1),(cx2,cy2),(255,0,0),2)
    if len(persons)==0:
        last_chest_roi = None
        cv2.putText(
            frame,
            "No person detected",
            (30,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,255),
            2
        )
    cv2.imshow("Person + Chest Roi",frame)
    key = cv2.waitKey(1) &0xFF
    if key == ord("s"):
        if last_chest_roi is not None:
            image_count +=1
            filename = f"dataset/chest_roi/chest_{image_count:03d}.jpg"
            cv2.imwrite(filename,last_chest_roi)
            print(f"Đã lưu ảnh: {filename}")
        else:
            print("Chưa tìm thấy người")
    if key == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()