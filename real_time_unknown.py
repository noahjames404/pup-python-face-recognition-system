import cv2 as cv
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pickle
from keras_facenet import FaceNet
import requests


# Define a threshold value for face recognition
threshold = 0.7

# Load the saved SVM model
with open("C:\\Users\\icedl\\Dropbox\\PC\\Downloads\\Facenet_MTCNN_SVM [With Augmentation]\\venv\\svm_model_160x160.pkl",'rb') as f:
    model = pickle.load(f)

# Read the pre-trained face embedding model
facenet = FaceNet()

# Load the saved face embeddings and labels
faces_embeddings = np.load("C:\\Users\\icedl\\Dropbox\\PC\\Downloads\\Facenet_MTCNN_SVM [With Augmentation]\\venv\\faces_embeddings_done_4classes.npz")
X, Y = faces_embeddings['arr_0'], faces_embeddings['arr_1']
encoder = LabelEncoder()
encoder.fit(Y)

# Read the Haar cascade classifier for face detection
haarcascade = cv.CascadeClassifier("C:\\Users\\icedl\\Dropbox\\PC\\Downloads\\Facenet_MTCNN_SVM [With Augmentation]\\venv\\haarcascade_frontalface_default.xml")

# Open the camera and start capturing video
cap = cv.VideoCapture(0)

while cap.isOpened():
    # Read a frame from the camerac
    _, frame = cap.read()

    # Convert the image color to RGB
    rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # Convert the RGB into gray since haarcascade requires gray images
    gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Detect faces using the Haar cascade classifier
    faces = haarcascade.detectMultiScale(gray_img, 1.3, 5)

    # Process each detected face
    for x,y,w,h in faces:
        # Extract the face region from the image
        img = rgb_img[y:y+h, x:x+w]

        # Resize the face region to 160x160
        img = cv.resize(img, (160,160))

        # Convert the 3D image array to a 4D array of shape (1, 160, 160, 3)
        img = np.expand_dims(img,axis=0)

        # Embed the face using FaceNet
        embedding = facenet.embeddings(img)

        # Classify the face using the SVM model
        ypreds = model.predict_proba(embedding)
        ypreds_max = np.max(ypreds)
        ypreds_label = np.argmax(ypreds)
        if ypreds_max > threshold:
            face_name = encoder.inverse_transform([ypreds_label])[0]
             # Draw a rectangle around the face and display the name
            cv.rectangle(frame, (x,y), (x+w,y+h), (0,255, 0), 5)
            cv.putText(frame, face_name, (x,y-10), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255, 255), 2, cv.LINE_AA)
            
            
            res = requests.post(
                url="https://facedetection-1-s8245812.deta.app/sms/send",
                json={
                    "send_to": "+639706643226",
                    "body": "Hello, your son/daughter has left the school premises!",
                    "name": face_name,
                },
)
            print(res.json())
        else:
            face_name = "not registered"
        # Draw a rectangle around the face and display the name
            cv.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 5)
            cv.putText(frame, face_name, (x,y-10), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv.LINE_AA)

    # Display the result on the screen
    cv.imshow("Face Recognition:", frame)

    # Exit the loop if the 'q' key is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the resources
cap.release()
cv.destroyAllWindows()

