
from pathlib import Path

import cv2
import numpy as np

DATASET_DIR = Path("dataset_seatbelt")
SPLIT = "train"


def draw_segmentation(image_path, label_path):
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Khong doc duoc anh: {image_path}")
        return None

    height, width = image.shape[:2]

    if not label_path.exists():
        print(f"THIEU NHAN: {label_path}")
        return image

    lines = label_path.read_text(
        encoding="utf-8"
    ).splitlines()

    polygon_count = 0

    for line in lines:
        parts = line.strip().split()

        if not parts:
            continue

        if len(parts) < 7 or (len(parts) - 1) % 2 != 0:
            print(f"Nhan polygon khong hop le: {label_path}")
            continue

        class_id = int(float(parts[0]))

        coords = np.array(
            [float(value) for value in parts[1:]],
            dtype=np.float32,
        ).reshape(-1, 2)

        coords[:, 0] *= width
        coords[:, 1] *= height

        polygon = np.round(coords).astype(np.int32)

        cv2.polylines(
            image, [polygon], True, (0, 255, 0), 2
        )

        cv2.putText(
            image,
            f"class {class_id}",
            tuple(polygon[0]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        polygon_count += 1

    cv2.putText(
        image,
        f"Image: {image_path.name}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
    )

    print(f"{image_path.name}: {polygon_count} polygon")

    return image


images_dir = DATASET_DIR / "images" / SPLIT
labels_dir = DATASET_DIR / "labels" / SPLIT

image_paths = sorted(
    list(images_dir.glob("*.jpg"))
    + list(images_dir.glob("*.jpeg"))
    + list(images_dir.glob("*.png"))
)

print(f"Tong so anh {SPLIT}: {len(image_paths)}")

for image_path in image_paths:
    label_path = labels_dir / f"{image_path.stem}.txt"

    result = draw_segmentation(image_path, label_path)

    if result is None:
        continue

    cv2.imshow("Kiem tra nhan - " + image_path.name, result)

    print("Nhan phim bat ky de xem anh tiep theo, nhan q de thoat.")

    key = cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()

    if key == ord("q"):
        break

print("Da kiem tra xong.")
