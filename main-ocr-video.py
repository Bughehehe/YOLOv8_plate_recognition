from ultralytics import YOLO
import cv2
import easyocr

def read_licence_plate(ocr_reader, crop):
    text_detections = ocr_reader.readtext(crop, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    max_score, max_score_id = 0, -1
    for i in range(len(text_detections)):
        if text_detections[i][2] > max_score and len(text_detections[i][1]) >= 2 and len(text_detections[i][1]) <= 8:
            max_score = text_detections[i][2]
            max_score_id = i   

    if max_score_id != -1:
        return text_detections[max_score_id][1], text_detections[max_score_id][2]
    else:
        return "None", -1

licence_plate_detector = YOLO("best.pt")

cap = cv2.VideoCapture('./pwr-parking.mp4')
ocr_reader = easyocr.Reader(['en'], gpu=True)

video_plates = ["EL5GY88", "DW075HS", "PKA03893", "DWR0661G", "DW839FX", "DTR38EH", "DTR85451", "DW2MS24", "DW1TK95"]

ret = True
while ret:
    ret, frame = cap.read()
    if ret:
        licence_plates = licence_plate_detector(frame)[0]
        for licence_plate in licence_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = licence_plate
            licence_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
            licence_plate_crop_gray = cv2.cvtColor(licence_plate_crop, cv2.COLOR_BGR2GRAY)
            _, licence_plate_crop_thresh = cv2.threshold(licence_plate_crop_gray, 120, 255, cv2.THRESH_BINARY_INV)
            licence_plate_text, licence_plate_score = read_licence_plate(ocr_reader, licence_plate_crop_thresh)

            if licence_plate_text in video_plates:
                video_plates.remove(licence_plate_text)

print(video_plates)