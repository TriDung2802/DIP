import cv2
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

SEATBELT_DIR = BASE_DIR / "dataset" / "person" / "seatbelt"
NO_SEATBELT_DIR = BASE_DIR / "dataset" / "person" / "no_seatbelt"

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


seatbelt_count = get_next_number(
    SEATBELT_DIR,
    "co_day_"
)

no_seatbelt_count = get_next_number(
    NO_SEATBELT_DIR,
    "khong_day_"
)


def save_seatbelt_image(person_image):
    global seatbelt_count

    filename = (
        SEATBELT_DIR /
        f"co_day_{seatbelt_count:03d}.jpg"
    )

    ok = cv2.imwrite(
        str(filename),
        person_image
    )

    if ok:
        print(f"Da luu anh CO DAY: {filename}")
        seatbelt_count += 1
        return True

    print("Luu anh CO DAY that bai")
    return False


def save_no_seatbelt_image(person_image):
    global no_seatbelt_count

    filename = (
        NO_SEATBELT_DIR /
        f"khong_day_{no_seatbelt_count:03d}.jpg"
    )

    ok = cv2.imwrite(
        str(filename),
        person_image
    )

    if ok:
        print(f"Da luu anh KHONG DAY: {filename}")
        no_seatbelt_count += 1
        return True

    print("Luu anh KHONG DAY that bai")
    return False