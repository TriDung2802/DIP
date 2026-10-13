import cv2

from person_detection.camera import open_camera
from person_detection.person_detector import detect_people
from person_detection.dataset_saver import (
    save_seatbelt_image,
    save_no_seatbelt_image
)

def main():

    cap = open_camera()

    print("Camera da mo.")
    print("Nhan:")
    print("  1 -> Luu anh CO DAY")
    print("  2 -> Luu anh KHONG DAY")
    print("  q -> Thoat")

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Khong doc duoc frame.")
            break

        frame_clean = frame.copy()

        persons = detect_people(frame)


        frame_display = frame.copy()

        for person in persons:

            x1, y1, x2, y2 = person["person_box"]

            cv2.rectangle(
                frame_display,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

        cv2.imshow(
            "Capture Dataset",
            frame_display
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("1"):

            if len(persons) == 0:
                print("Khong phat hien nguoi.")
                continue

            for person in persons:

                x1, y1, x2, y2 = person["person_box"]
                person_image = frame_clean[
                    y1:y2,
                    x1:x2
                ]

                if (
                    person_image is not None
                    and person_image.size > 0
                ):
                    save_seatbelt_image(
                        person_image
                    )

        elif key == ord("2"):

            if len(persons) == 0:
                print("Khong phat hien nguoi.")
                continue

            for person in persons:

                x1, y1, x2, y2 = person["person_box"]

                person_image = frame_clean[
                    y1:y2,
                    x1:x2
                ]

                if (
                    person_image is not None
                    and person_image.size > 0
                ):
                    save_no_seatbelt_image(
                        person_image
                    )


        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
