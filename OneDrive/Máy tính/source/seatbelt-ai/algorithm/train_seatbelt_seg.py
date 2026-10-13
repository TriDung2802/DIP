from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset_seatbelt"
DATA_YAML = DATASET_DIR / "data.yaml"
MODEL_OUTPUT_DIR = PROJECT_ROOT / "models"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def validate_dataset():
    required_dirs = (
        DATASET_DIR / "images" / "train",
        DATASET_DIR / "images" / "val",
        DATASET_DIR / "labels" / "train",
        DATASET_DIR / "labels" / "val"
    )

    missing_dirs = [path for path in required_dirs if not path.is_dir()]

    if missing_dirs:
        missing = "\n".join(str(path) for path in missing_dirs)
        raise FileNotFoundError(
            "YOLO-seg dataset folders are missing. Create these folders "
            "and add polygon labels before training:\n"
            f"{missing}"
        )

    for split in ("train", "val"):
        images_dir = DATASET_DIR / "images" / split
        images = [
            path for path in images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ]
        if not images:
            raise ValueError(f"No {split} images found in {images_dir}")

        labels_dir = DATASET_DIR / "labels" / split
        missing_labels = [
            image for image in images
            if not (labels_dir / f"{image.stem}.txt").is_file()
        ]
        if missing_labels:
            examples = "\n".join(str(path) for path in missing_labels[:5])
            raise ValueError(
                f"Images are missing YOLO-seg label files in {labels_dir}:\n"
                f"{examples}"
            )

        for label_path in labels_dir.glob("*.txt"):
            for line_number, line in enumerate(
                label_path.read_text(encoding="utf-8").splitlines(),
                start=1
            ):
                fields = line.split()
                if not fields:
                    continue

                if len(fields) < 7 or (len(fields) - 1) % 2:
                    raise ValueError(
                        f"Invalid polygon row at {label_path}:{line_number}"
                    )

                try:
                    class_id = int(fields[0])
                    coordinates = [float(value) for value in fields[1:]]
                except ValueError as error:
                    raise ValueError(
                        f"Invalid label value at {label_path}:{line_number}"
                    ) from error

                if class_id != 0 or any(
                    not 0 <= coordinate <= 1
                    for coordinate in coordinates
                ):
                    raise ValueError(
                        f"Expected class 0 and normalized coordinates in "
                        f"[0, 1] at {label_path}:{line_number}"
                    )


def train():
    validate_dataset()

    model = YOLO("yolo11n-seg.pt")
    model.train(
        data=str(DATA_YAML),
        epochs=100,
        imgsz=640,
        batch=4,
        project=str(MODEL_OUTPUT_DIR),
        name="seatbelt-seg-v2",
        exist_ok=True
    )


if __name__ == "__main__":
    train()
