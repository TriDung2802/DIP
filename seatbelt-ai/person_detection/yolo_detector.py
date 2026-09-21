from ultralytics import YOLO

model=YOLO("models/yolo11n.pt")


def detect_person(frame):
    results = model(frame,verbose=False)
    result = results[0]
    persons=[]
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float (box.conf[0])

        if class_id !=0 :
            continue
        x1,y1,x2,y2 = map(int, box.xyxy[0])
        persons.append({
            "person_box" : (x1,y1,x2,y2),
            "confidence" : confidence
        })
    return persons