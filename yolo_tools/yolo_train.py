from ultralytics import YOLO
import testing.test_init

model = YOLO("../yolov8n.pt")

if __name__ == '__main__':
    model.train(data="../dataset/test3_2_final/data.yaml", epochs=80, imgsz=640, cos_lr=True)
    model.val()


