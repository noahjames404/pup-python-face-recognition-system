import os
import pickle
import sys
import time

import cv2 as cv
import numpy as np
import requests
import torch
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder

sys.path.append("External-Attention-pytorch")
from model.attention.CBAM import CBAMBlock

# Define a threshold value for face recognition
threshold = 0.54

# Load the saved SVM model
with open("svc_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load the saved cbam
with open("cbam.pkl", "rb") as f:
    cbam = pickle.load(f)

# Read the pre-trained face embedding model
facenet = FaceNet()

# Load the saved face embeddings and labels
faces_embeddings = np.load("faces_embeddings_done_4classes.npz")
X, Y = faces_embeddings["arr_0"], faces_embeddings["arr_1"]
encoder = LabelEncoder()
encoder.fit(Y)

# Read the Haar cascade classifier for face detection
haarcascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

save_path = "captured_img"  # Replace with your desired save folder path

if not os.path.exists(save_path):
    os.makedirs(save_path)

num_screenshots = 0

# Open the camera and start capturing video
cap = cv.VideoCapture(0)

while cap.isOpened():
    # Read a frame from the camera
    _, frame = cap.read()

    # Convert the image color to RGB
    rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # Convert the RGB into gray since haarcascade requires gray images
    gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Detect faces using the Haar cascade classifier
    faces = haarcascade.detectMultiScale(
        gray_img, scaleFactor=1.1, minNeighbors=6, minSize=(30, 30)
    )

    # Process each detected face
    for x, y, w, h in faces:
        # Extract the face region from the image
        img = rgb_img[y : y + h, x : x + w]

        # Resize the face region to 160x160
        img = cv.resize(img, (160, 160))

        # Convert the 3D image array to a 4D array of shape (1, 160, 160, 3)
        img = np.expand_dims(img, axis=0)

        # Embed the face using FaceNet
        embedding = facenet.embeddings(img)

        # cbam
        height, width = 16, 32  # Specify the desired height and width
        channels = 1  # Grayscale
        embedded_img = np.reshape(embedding, (-1, height, width, channels))

        torch_tensor = torch.from_numpy(embedded_img).float()
        with torch.no_grad():
            output = cbam(torch_tensor)

        # Classify the face using the SVM model
        test_im = output.detach().numpy()
        ypreds = model.predict_proba(test_im.reshape(test_im.shape[0], -1))
        ypreds_max = np.max(ypreds)
        ypreds_label = np.argmax(ypreds)

        if ypreds_max > threshold:
            face_name = encoder.inverse_transform([ypreds_label])[0]

            req = requests.post(
                "https://facedetection-1-s8245812.deta.app/sms/send",
                json={
                    "send_to": "+639761279041",
                    "body": "Helloooo",
                    "name": face_name,
                },
            )

            # Save a screenshot if the number of screenshots is less than 3
            if num_screenshots < 3:
                screenshot_name = "screenshot_{}_{}.png".format(face_name, num_screenshots)
                cv.imwrite(os.path.join(save_path, screenshot_name), frame)
                num_screenshots += 1

                # Delay for 1 second before capturing the next screenshot
                time.sleep(1)

            # Draw a rectangle around the face and display the name
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 5)
            cv.putText(
                frame,
                face_name,
                (x, y - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2,
                cv.LINE_AA,
            )
        else:
            face_name = "not registered"
            # Draw a rectangle around the face and display the name
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 5)
            cv.putText(
                frame,
                face_name,
                (x, y - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                cv.LINE_AA,
            )

    # Display the result on the screen
    cv.imshow("Face Recognition:", frame)

    # Exit the loop if the 'q' key is pressed
    if cv.waitKey(1) & 0xFF == ord("q"):
        break

# Release the resources
cap.release()
cv.destroyAllWindows()
