from ultralytics import YOLO

model = YOLO('yolo11l.pt')

results = model("test/001.jpg")
results[0].show()