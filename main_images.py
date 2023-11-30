from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score 
from keras import optimizers
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Flatten, MaxPooling2D, Dropout, Conv2D
import os

DATASET_PATH = "best.pt"
VID_PATH = "sample_better.mp4"
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
    img_lp = cv2.resize(image, (420, 120))
    # img_lp = image
    # plt.show()
    cv2.imwrite('contour.jpg',img_lp)
    img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)
    _, img_binary_lp = cv2.threshold(img_gray_lp, 200, 240, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img_binary_lp = cv2.erode(img_binary_lp, (3,3))
    img_binary_lp = cv2.dilate(img_binary_lp, (3,3))
    # print(img_binary_lp.shape)
    LP_WIDTH = img_binary_lp.shape[0]
    LP_HEIGHT = img_binary_lp.shape[1]

    # Make borders white
    img_binary_lp[0:3,:] = 255
    img_binary_lp[:,0:3] = 255
    img_binary_lp[LP_WIDTH-3:LP_WIDTH,:] = 255
    img_binary_lp[:,LP_HEIGHT-3:LP_HEIGHT] = 255

    # Estimations of character contours sizes of cropped license plates
    dimensions = [LP_WIDTH/30,
                       LP_WIDTH/1.5,
                       LP_HEIGHT/10,
                       LP_HEIGHT/1.7]
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
  
def show_results(char, dic):
    output = []
    for i,ch in enumerate(char): #iterating over the characters
        img_ = cv2.resize(ch, (28,28), interpolation=cv2.INTER_AREA)
        img = fix_dimension(img_)
        img = img.reshape(1,28,28,3) #preparing image for the model
        predict_x=model.predict(img)
        y_=np.argmax(predict_x,axis=1) 
        character = dic[int(y_)] #
        output.append(character) #storing the result in a list
        
    plate_number = ''.join(output)
    # print(plate_number)
    
    return plate_number


dic = {}
characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
for i,c in enumerate(characters):
    dic[i] = c

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

# mot_tracker = Sort()

# create model

licence_plate_detector = YOLO(DATASET_PATH)

# read photos
main_path = os.getcwd()
os.chdir(main_path + "\\img")
photos = os.listdir(main_path + "\\img")
photos.sort(key=len)


labels = []

for photo in photos:

    img = cv2.imread(photo)

    licence_plastes = licence_plate_detector(img)[0]

    if len(licence_plastes) == 2:
       print("DWA OBRAZY")
       print(labels)
       

    
    temp_boxes = licence_plastes.boxes.data.tolist()
    for licence_plate in temp_boxes:
        x1, y1, x2, y2, score, class_id = licence_plate
        print(licence_plate)
        license_plate_crop = img[int(y1):int(y2), int(x1):int(x2), :]

        cv2.imshow("original_crop", license_plate_crop)

        ########################### PHOTO CNN ########################################

        char = segment_characters(license_plate_crop)

        predicted_label = []

        plt.figure(figsize=(10,6))
        for i,ch in enumerate(char):
            img = cv2.resize(ch, (28,28), interpolation=cv2.INTER_AREA)
            plt.subplot(3,4,i+1)
            plt.imshow(img,cmap='gray')
            temp = show_results(char, dic)[i]
            plt.title(f'predicted: {temp}')
            plt.axis('off')
            predicted_label.append(temp)
        plt.show()

        labels.append(''.join(predicted_label))

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        plt.close()
