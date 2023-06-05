import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp

# Create a video capture object
cap = cv2.VideoCapture(0)  # Change the index to select a different camera (if available)

# Define the colormap
cmap = cv2.COLORMAP_INFERNO

# Initialize mediapipe modules
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

with mp_face_mesh.FaceMesh() as face_mesh:
    while True:
        # Read a frame from the video capture
        ret, frame = cap.read()

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Perform your computer vision analysis or processing on the grayscale frame
        # For example, let's detect faces using Haar cascades
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Convert the frame to RGB format for face mesh detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect face landmarks and draw face mesh
        results = face_mesh.process(rgb_frame)
        face_landmarks = results.multi_face_landmarks

        # Draw face mesh on the frame
        if face_landmarks:
            for face_landmark in face_landmarks:
                # Draw connections
                mp_drawing.draw_landmarks(frame, face_landmark, mp_face_mesh.FACEMESH_TESSELATION,
                                          landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=0, circle_radius=0))

                # Draw filled polygons for each face mesh triangle
                for triangle in mp_face_mesh.FACEMESH_TESSELATION:
                    triangle_points = [(face_landmark.landmark[index].x * frame.shape[1],
                                        face_landmark.landmark[index].y * frame.shape[0]) for index in triangle]

        # Display the frame with face mesh
        cv2.imshow('Thermal Face', frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()