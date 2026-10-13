from ultralytics import YOLO

# Model YOLO11 Segmentation
model = YOLO("yolo11n-seg.pt")

# Huấn luyện
model.train(
    data="dataset_seatbelt/data.yaml",
    epochs=50,
    imgsz=640,
    batch=4,
    project="runs/seatbelt",
    name="segmentation"
)