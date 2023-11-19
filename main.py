from ultralytics import YOLO
import cv2
from abewely_sort import *
import computer_vision_engineer_util as util
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score 
from keras import optimizers
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Flatten, MaxPooling2D, Dropout, Conv2D

DATASET_PATH = "best.pt"
VID_PATH = "sample.mp4"
YOLO_MODEL = 'yolov8n.pt'

def find_contours(dimensions, img) :

    # Find all contours in the image
    cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Retrieve potential dimensions
    lower_width = dimensions[0]
    upper_width = dimensions[1]
    lower_height = dimensions[2]
    upper_height = dimensions[3]
    
    # Check largest 5 or  15 contours for license plate or character respectively
    cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]
    
    ii = cv2.imread('contour.jpg')
    
    x_cntr_list = []
    target_contours = []
    img_res = []
    for cntr in cntrs :
        # detects contour in binary image and returns the coordinates of rectangle enclosing it
        intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)
        
        # checking the dimensions of the contour to filter out the characters by contour's size
        if intWidth > lower_width and intWidth < upper_width and intHeight > lower_height and intHeight < upper_height :
            x_cntr_list.append(intX) #stores the x coordinate of the character's contour, to used later for indexing the contours

            char_copy = np.zeros((44,24))
            # extracting each character using the enclosing rectangle's coordinates.
            char = img[intY:intY+intHeight, intX:intX+intWidth]
            char = cv2.resize(char, (20, 40))
            
            cv2.rectangle(ii, (intX,intY), (intWidth+intX, intY+intHeight), (50,21,200), 2)
            plt.imshow(ii, cmap='gray')

            # Make result formatted for classification: invert colors
            char = cv2.subtract(255, char)

            # Resize the image to 24x44 with black border
            char_copy[2:42, 2:22] = char
            char_copy[0:2, :] = 0
            char_copy[:, 0:2] = 0
            char_copy[42:44, :] = 0
            char_copy[:, 22:24] = 0

            img_res.append(char_copy) # List that stores the character's binary image (unsorted)
            
    # Return characters on ascending order with respect to the x-coordinate (most-left character first)
            
    # plt.show()
    # arbitrary function that stores sorted list of character indeces
    indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
    img_res_copy = []
    for idx in indices:
        img_res_copy.append(img_res[idx])# stores character images according to their index
    img_res = np.array(img_res_copy)

    return img_res

def segment_characters(image) :

    # Preprocess cropped license plate image
    img_lp = cv2.resize(image, (320, 60))
    # img_lp = image
    # plt.show()
    cv2.imwrite('contour.jpg',img_lp)
    img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)
    _, img_binary_lp = cv2.threshold(img_gray_lp, 210, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img_binary_lp = cv2.erode(img_binary_lp, (3,3))
    img_binary_lp = cv2.dilate(img_binary_lp, (3,3))
    print(img_binary_lp.shape)
    LP_WIDTH = img_binary_lp.shape[0]
    LP_HEIGHT = img_binary_lp.shape[1]

    # Make borders white
    img_binary_lp[0:3,:] = 255
    img_binary_lp[:,0:3] = 255
    img_binary_lp[LP_WIDTH-3:LP_WIDTH,:] = 255
    img_binary_lp[:,LP_HEIGHT-3:LP_HEIGHT] = 255

    # Estimations of character contours sizes of cropped license plates
    dimensions = [LP_WIDTH/25,
                       LP_WIDTH/1.3,
                       LP_HEIGHT/25,
                       LP_HEIGHT/1.3]
    plt.imshow(img_binary_lp, cmap='gray')
    cv2.imwrite('contour.jpg',img_binary_lp)

    # Get contours within cropped license plate
    char_list = find_contours(dimensions, img_binary_lp)

    return char_list

def f1score(y, y_pred):
  return f1_score(y, tf.math.argmax(y_pred, axis=1), average='micro') 

def custom_f1score(y, y_pred):
  return tf.py_function(f1score, (y, y_pred), tf.double)

def fix_dimension(img): 
  new_img = np.zeros((28,28,3))
  for i in range(3):
    new_img[:,:,i] = img
  return new_img
  
def show_results(char):
    dic = {}
    characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for i,c in enumerate(characters):
        dic[i] = c

    output = []
    for i,ch in enumerate(char): #iterating over the characters
        img_ = cv2.resize(ch, (28,28), interpolation=cv2.INTER_AREA)
        img = fix_dimension(img_)
        img = img.reshape(1,28,28,3) #preparing image for the model
        # y_ = model.predict_classes(img)[0] #predicting the class
        predict_x=model.predict(img)
        y_=np.argmax(predict_x,axis=1) 
        character = dic[int(y_)] #
        output.append(character) #storing the result in a list
        
    plate_number = ''.join(output)
    print(plate_number)
    
    return plate_number



model = Sequential()
model.add(Conv2D(16, (22,22), input_shape=(28, 28, 3), activation='relu', padding='same'))
model.add(Conv2D(32, (16,16), input_shape=(28, 28, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (8,8), input_shape=(28, 28, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (4,4), input_shape=(28, 28, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(36, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.Adam(learning_rate=0.0001), metrics=custom_f1score)
model.load_weights("CNN_Plate_Letters_Recognition_12epoch.h5")

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

            # license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
            # _ , license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

            # cv2.imwrite('plate.jpg',license_plate_crop)

            # cv2.imshow("original_crop", license_plate_crop)
            cv2.imshow("license_plate_crop", license_plate_crop)
            # cv2.waitKey(0)

            # licence_plaste_text, licence_plaste_text, score = util.read_license_plate(license_plate_crop_thresh)

            ########################### PHOTO CNN ########################################

            char = segment_characters(license_plate_crop)

            plt.figure(figsize=(10,6))
            for i,ch in enumerate(char):
                img = cv2.resize(ch, (28,28), interpolation=cv2.INTER_AREA)
                plt.subplot(3,4,i+1)
                plt.imshow(img,cmap='gray')
                plt.title(f'predicted: {show_results(char)[i]}')
                plt.axis('off')
            plt.show()

            cv2.waitKey(0)
            cv2.destroyAllWindows()
            plt.close()
