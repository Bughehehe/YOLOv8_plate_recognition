from ultralytics import YOLO
import cv2

DATASET_PATH = "best.pt"
VID_PATH = "sample.mp4"
YOLO_MODEL = 'yolov8n.pt'

# create model

COCO_model = YOLO(YOLO_MODEL)
licence_plate_detector = YOLO(DATASET_PATH)

# training video

vid = cv2.VideoCapture(VID_PATH)

# read frames

ret = True
while ret:
    frame_number =+ 1
    ret, frame = vid.read()
    if ret and frame_number < 10 : 
        pass

        detection = COCO_model(frame)[0]
        print(detection)