from ultralytics import YOLO

model = YOLO("yolov8m.pt")  # m = medium, good balance
model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    device=0
)
