import cv2 as cv
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pickle
from keras_facenet import FaceNet
import time
import mediapipe as mp
import tensorflow as tf

# Load the saved SVM model
with open("svm_model_160x160.pkl", 'rb') as f:
    model = pickle.load(f)
    
# Load the saved face embeddings and labels
faces_embeddings = np.load("faces_embeddings_done_4classes.npz")
X, Y = faces_embeddings['arr_0'], faces_embeddings['arr_1']
encoder = LabelEncoder()
encoder.fit(Y)

# Define a threshold value for face recognition
threshold = 0.7

# Initialize the MediaPipe FaceMesh
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Read the pre-trained face embedding model
facenet = FaceNet()

# Read the Haar cascade classifier for face detection
haarcascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

# Open the camera and start capturing video
cap = cv.VideoCapture(0)

save_path = 'captured_img'  # Replace with your desired save folder path

if not os.path.exists(save_path):
    os.makedirs(save_path)

num_screenshots = 0

with mp_face_mesh.FaceMesh(max_num_faces=3, refine_landmarks=True, min_detection_confidence=0.3) as face_mesh:
    while True:

        #while cap.isOpened():
        # Read a frame from the camera
        _, frame = cap.read()

        # Convert the image color to RGB
        rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # Convert the RGB into gray since haarcascade requires gray images
        gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Detect faces using the Haar cascade classifier
        faces = haarcascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Convert the frame to RGB format for face mesh detection
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # Process each detected face
        for x, y, w, h in faces:
            
            # Detect face landmarks and draw face mesh
            results = face_mesh.process(rgb_frame)
            face_landmarks = results.multi_face_landmarks
            
            # Draw face mesh on the frame
            if face_landmarks:
                for face_landmark in face_landmarks:
                    # Draw connections
                    mp_drawing.draw_landmarks(frame, face_landmark, mp_face_mesh.FACEMESH_TESSELATION,
                                              landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 155, 0), thickness=1, circle_radius=1))
            
            # Extract the face region from the image
            img = rgb_img[y:y + h, x:x + w]
        
            # Resize the face region to 160x160
            img = cv.resize(img, (160, 160))

            # Convert the 3D image array to a 4D array of shape (1, 160, 160, 3)
            img = np.expand_dims(img, axis=0)

            # Embed the face using FaceNet
            embedding = facenet.embeddings(img)
            
            # Classify the face using the SVM model
            ypreds = model.predict_proba(embedding)
            print(ypreds)
            ypreds_max = np.max(ypreds)
            ypreds_label = np.argmax(ypreds)
            
            if ypreds_max > threshold:
                face_name = encoder.inverse_transform([ypreds_label])[0]
                # Draw a rectangle around the face and display the name
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 5)
                cv.putText(frame, face_name, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv.LINE_AA)
            else:
                face_name = "not registered"
                # Draw a rectangle around the face and display the name
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 5)
                cv.putText(frame, face_name, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
            
        # Display the result on the screen
        cv.imshow("Face Recognition:", frame)
        
        # Save a screenshot if the number of screenshots is less than 3
        if num_screenshots < 3:
            screenshot_name = 'screenshot_{}.png'.format(num_screenshots)
            cv.imwrite(os.path.join(save_path, screenshot_name), frame)
            num_screenshots += 1

            # Delay for 1 second before capturing the next screenshot
            time.sleep(1)
        
        # Exit the loop if the 'q' key is pressed
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

# Release the video capture object and close all windows
cap.release()
cv.destroyAllWindows()