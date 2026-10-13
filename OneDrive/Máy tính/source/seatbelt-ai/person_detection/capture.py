import cv2
from pathlib import Path

from .camera import open_camera
from .person_detector import detect_people


BASE_DIR = Path(__file__).resolve().parent.parent

SEATBELT_DIR = BASE_DIR / "dataset" / "chest_roi" / "seatbelt"
NO_SEATBELT_DIR = BASE_DIR / "dataset" / "chest_roi" / "no_seatbelt"

SEATBELT_DIR.mkdir(parents=True, exist_ok=True)
NO_SEATBELT_DIR.mkdir(parents=True, exist_ok=True)

def get_next_number(folder, prefix):
    numbers = []

    for file in folder.glob("*.jpg"):
        name = file.stem

        if name.startswith(prefix):
            number = name.replace(prefix, "")

            if number.isdigit():
                numbers.append(int(number))

    if numbers:
        return max(numbers) + 1

    return 1


seatbelt_count = get_next_number(SEATBELT_DIR, "co_day_")
no_seatbelt_count = get_next_number(NO_SEATBELT_DIR, "khong_day_")

cap = open_camera()


while True:

    ret, frame = cap.read()

    if not ret:
        break

    # tìm người

    persons = detect_people(frame)

    # Vẽ kết quả

    for person in persons:

        x1, y1, x2, y2 = person["person_box"]
        confidence = person["confidence"]

        cx1, cy1, cx2, cy2 = person["chest_box"]


        # Khung người
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )


        # Khung chest ROI
        cv2.rectangle(
            frame,
            (cx1, cy1),
            (cx2, cy2),
            (255, 0, 0),
            2
        )

        # Confidence
        cv2.putText(
            frame,
            f"person {confidence:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )
    cv2.putText(
        frame,
        "S: Co day | N: Khong day | Q: Thoat",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    if len(persons) == 0:

        cv2.putText(
            frame,
            "No person detected",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )


    cv2.imshow("Collect Seatbelt Dataset", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):

        if len(persons) > 0:

            roi = persons[0]["person_box"]
            x1,y1,x2,y2 = roi
            person_image = frame[y1:y2,x1:x2]

            filename = SEATBELT_DIR / f"co_day_{seatbelt_count:03d}.jpg"

            cv2.imwrite(str(filename), person_image)

            print(f"Da luu anh CO DAY: {filename}")

            seatbelt_count += 1

        else:

            print("Chua tim thay nguoi")


    elif key == ord("n"):

        if len(persons) > 0:

            roi = persons[0]["chest_roi"]

            filename = NO_SEATBELT_DIR / f"khong_day_{no_seatbelt_count:03d}.jpg"

            ok = cv2.imwrite(str(filename), roi)

            if ok:
                print(f"Da luu anh CO DAY: {filename}")
                seatbelt_count += 1
            else:
                print("Luu anh that bai")

            print(f"Da luu anh KHONG DAY: {filename}")

            no_seatbelt_count += 1

        else:

            print("Chua tim thay nguoi")



    elif key == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()