from ultralytics import YOLO
import cv2

model = YOLO("../runs/detect/train4/weights/best.pt")

model.tune(data="../dataset/test3_final/data.yaml", epochs=30, iterations=30, optimizer="AdamW", plots=True, save=True, val=True)

result = model.predict(r"D:\projects\data_topdown\d2\d2_i_frames_0210.png")
result = model.predict("../data/w2_148.png")

print(len(result[0].boxes))

cv2.imshow("plot",result[0].plot())
cv2.waitKey(100000)
cv2.destroyAllWindows()

raise 1


from PIL.Image import Image
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction

detection_model = AutoDetectionModel.from_pretrained(
    model_type="yolov8",
    model_path="../runs/detect/train3/weights/best.pt",
    confidence_threshold=0.2,
    device="cuda:0"
)
result = get_sliced_prediction(
    r"D:\projects\data_topdown\d2\d2_i_frames_0210.png",
    detection_model,
    slice_height=1920,
    slice_width=1080,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2
)
print(len(result.object_prediction_list))
result.export_visuals(export_dir="../dataset/", rect_th=2, text_size=0.5)


