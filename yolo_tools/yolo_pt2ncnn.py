from ultralytics import YOLO

model = YOLO("runs/detect/train26/weights/best.pt")
model.export(format='onnx',simplify=True,opset=13)


