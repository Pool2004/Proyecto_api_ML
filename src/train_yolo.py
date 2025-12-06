from ultralytics import YOLO

model = YOLO("yolov8n-pose.pt")  # o yolov9c-pose.pt

model.train(
    data="../dataset.yaml",
    imgsz=640,
    epochs=100,
    batch=16,
)
