from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="configs/data.yaml",
    epochs=30,
    imgsz=640
)