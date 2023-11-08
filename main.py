from ultralytics import YOLO
import cv2
from abewely_sort import *
import computer_vision_engineer_util as util

DATASET_PATH = "best.pt"
VID_PATH = "sample.mp4"
YOLO_MODEL = 'yolov8n.pt'


mot_tracker = Sort()

# create model

COCO_model = YOLO(YOLO_MODEL)
licence_plate_detector = YOLO(DATASET_PATH)

# training video

vid = cv2.VideoCapture(VID_PATH)

vehicles = [2, 3, 5, 7]

# read frames

frame_number = -1
ret = True
while ret:
    frame_number =+ 1
    ret, frame = vid.read()
    if ret and frame_number < 10 : 
        detection = COCO_model(frame)[0]
        detections_ = []
        
        for detection in detection.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            # print(x1, y1, x2, y2, score, class_id)
            if int(class_id) in vehicles:
                detections_.append([x1,y1,x2,y2,score])

        track_ids = mot_tracker.update(np.array(detections_))

        licence_plastes = licence_plate_detector(frame)[0]

        for licence_plate in licence_plastes.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = licence_plate
            car_x1, car_y1, car_x2, car_y2, car_id = util.get_car(licence_plate, track_ids)

            license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]

            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
            _ , license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

            # cv2.imshow("original_crop", license_plate_crop)
            # cv2.imshow("threshold", license_plate_crop_thresh)
            # cv2.waitKey(0)

            licence_plaste_text, licence_plaste_text, score = util.read_license_plate(license_plate_crop_thresh)





        

