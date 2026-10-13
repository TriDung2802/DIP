from .yolo_detector import detect_person
from .roi import get_chest_roi


def detect_people(frame):
    persons = detect_person(frame)
    results = []

    height, width = frame.shape[:2]

    for person in persons:
        x1, y1, x2, y2 = map(int, person["person_box"])

        x1 = max(0, min(x1, width))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height))
        y2 = max(0, min(y2, height))

        if x2 <= x1 or y2 <= y1:
            continue

        person_box = (x1, y1, x2, y2)
        person_image = frame[y1:y2, x1:x2].copy()

        chest_roi, chest_box = get_chest_roi(frame, person_box)

        results.append({
            "person_box": person_box,
            "person_image": person_image,
            "chest_roi": chest_roi,
            "chest_box": chest_box,
            "confidence": person["confidence"]
        })

    return results

