import os
import cv2
import numpy as np
import streamlit as st

# Set Streamlit page configuration as the first Streamlit command
st.set_page_config(
    page_title="Face Detection Test",
    page_icon=":smiley:",
    layout="wide",
)

st.title("Face Detection Test")
st.sidebar.title("Settings")

# Load Haar Cascade file for face detection
face_cascade_path = 'model/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(face_cascade_path)

if face_cascade.empty():
    st.error("Failed to load Haar Cascade file. Please check the file path.")
    st.stop()

st.write("Haar Cascade loaded successfully.")

# Use Streamlit's camera input widget to capture a photo
camera_input = st.camera_input("Take a picture", key="camera_input")

if camera_input is not None:
    # Convert the captured image to an OpenCV format
    file_bytes = np.asarray(bytearray(camera_input.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Debug: Output image dimensions
    st.write(f"Image dimensions: {gray.shape}")

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Debug: Show number of faces detected
    st.write(f"Number of faces detected: {len(faces)}")

    if len(faces) == 0:
        st.write("No faces detected. Please adjust the camera position or ensure proper lighting.")
    else:
        # Draw bounding box around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the frame with bounding boxes
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_column_width=True)
else:
    st.write("No image captured. Please use the camera to take a picture.")