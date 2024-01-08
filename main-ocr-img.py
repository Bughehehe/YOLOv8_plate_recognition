import os
from ultralytics import YOLO
import cv2
import easyocr
import json

def read_plates_set(path):
    with open(path, "r") as file:
        file_content = file.read()
        return json.loads(file_content)

def read_licence_plate(ocr_reader, crop):
    text_detections = ocr_reader.readtext(crop, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    max_score, max_score_id = 0, -1
    for i in range(len(text_detections)):
        if text_detections[i][2] > max_score and len(text_detections[i][1]) >= 7 and len(text_detections[i][1]) <= 8:
            max_score = text_detections[i][2]
            max_score_id = i   

    if max_score_id != -1:
        return text_detections[max_score_id][1], text_detections[max_score_id][2]
    else:
        return "None", -1

def results_analyze(file_name, results, valid_set):
    with open(file_name, "w") as results_file:
        for key in results.keys():
            valid_plates = valid_set[key].split(";")
            for plate in results[key]:
                if plate[0] in valid_plates:
                    results_file.write(f"{key};{plate[1]};{plate[0]};{plate[0]}\n")
                else:
                    results_file.write(f"{key};{plate[1]};{plate[0]};{valid_set[key]}\n")

path = "img/"
ocr_reader = easyocr.Reader(['en'], gpu=True)
licence_plate_detector = YOLO("best.pt")

plates_set = read_plates_set("plates.json")
images_names = os.listdir(path)

results = {}
for image_name in images_names:
    image = cv2.imread("img/" + image_name)
    licence_plates = licence_plate_detector(image)[0]
    image_plates = []
    for licence_plate in licence_plates.boxes.data.tolist():
        x1, y1, x2, y2, _, _ = licence_plate
        licence_plate_crop = image[int(y1):int(y2), int(x1):int(x2), :]
        licence_plate_crop_gray = cv2.cvtColor(licence_plate_crop, cv2.COLOR_BGR2GRAY)
        _, licence_plate_crop_thresh = cv2.threshold(licence_plate_crop_gray, 120, 255, cv2.THRESH_BINARY_INV)
        licence_plate_text, licence_plate_score = read_licence_plate(ocr_reader, licence_plate_crop_gray)
        image_plates.append((licence_plate_text, licence_plate_score))
    results[image_name[:-4]] = image_plates

results_analyze("results-gray.csv", results, plates_set)
