# face recognition part II

#IMPORT
import cv2 as cv
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pickle
from keras_facenet import FaceNet



#INITIALIZE
#Simply reading the files generated from the training
facenet = FaceNet() #read FaceNet
faces_embeddings = np.load("C:\\Users\\icedl\\Dropbox\\PC\\Downloads\\Facenet_MTCNN_SVM [With Augmentation]\\venv\\faces_embeddings_done_4classes.npz") #read the npz file which contains the X and Y arrays


Y = faces_embeddings['arr_1'] #only take the Y array; arr_1 means second array which corresponds to Y
encoder = LabelEncoder()
encoder.fit(Y) #fit Y, no need to transform

haarcascade = cv.CascadeClassifier("C:\\Users\\icedl\\Dropbox\\PC\\Downloads\\Facenet_MTCNN_SVM [With Augmentation]\\venv\\haarcascade_frontalface_default.xml") #read the haarcascade frontalface file
model = pickle.load(open("C:\\Users\\icedl\\Dropbox\\PC\\Downloads\\Facenet_MTCNN_SVM [With Augmentation]\\venv\\svm_model_160x160.pkl", 'rb')) #read the svm file


cap = cv.VideoCapture(0) #initialize the camera; 0 for webcam

# WHILE LOOP
while cap.isOpened():
    _, frame = cap.read() #read the second output
    rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB) #convert the image color to RGB
    gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) #convert the RGB into gray since haarcascade requires gray images
    faces = haarcascade.detectMultiScale(gray_img, 1.3, 5) #detects faces; 1.3 is scale factor and 5 is number of neighbors; this can be changed
    for x,y,w,h in faces: #for each array there will be 4 tuple
        img = rgb_img[y:y+h, x:x+w]
        img = cv.resize(img, (160,160)) #only outputs 3 dimensional 160x160x3
        img = np.expand_dims(img,axis=0) #facenet requires 4 dimensions 1x160x160x3
        ypred = facenet.embeddings(img) #gives the image array to facenet; 512 dimensional array
        face_name = model.predict(ypred) #put the image to the model; outputs numerical value
        final_name = encoder.inverse_transform(face_name)[0] #converts numerical value to categorical value
        
        # ypred_max = np.max(ypred)
        # ypred_label = np.argmax(ypred)
        # if ypred_max > threshold:
        #     final_name = encoder.inverse_transform(face_name)[0]
        # else:
        #     final_name = "unknown"
        
        
        cv.rectangle(frame, (x,y), (x+w,y+h), (255,0,255), 5) #draw rectangle
        cv.putText(frame, str(final_name), (x,y-10), cv.FONT_HERSHEY_SIMPLEX,
                   1, (0,0,255), 2, cv.LINE_AA)

    cv.imshow("Face Recognition:", frame) #try and accept function
    if cv.waitKey(1) & ord('q') ==27:
        break

cap.release()
cv.destroyAllWindows