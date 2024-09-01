import os
import cv2
import numpy as np
import tensorflow as tf
import streamlit as st
from tensorflow.keras.models import load_model
import time

# Set Streamlit page configuration as the first Streamlit command
st.set_page_config(
    page_title="Simulated Real-Time Facial Emotion Recognition",
    page_icon=":smiley:",
    layout="wide",
)

# Print TensorFlow and Keras versions
st.write(f"TensorFlow version: {tf.__version__}")
st.write(f"Keras version: {tf.__version__}")

# Load the pre-trained model from the 'model' directory
model_path = 'model/FER_model.h5'
if not os.path.exists(model_path):
    st.error("Model file not found. Please ensure FER_model.h5 is in the 'model' directory.")
    st.stop()
else:
    model = load_model(model_path)

# Load Haar Cascade file for face detection
face_cascade_path = 'model/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(face_cascade_path)

if face_cascade.empty():
    st.error("Failed to load Haar Cascade file. Please check the file path.")
    st.stop()

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

st.title("🎭 Simulated Real-Time Facial Emotion Recognition")
st.sidebar.title("Settings")

# Sidebar controls
confidence_threshold = st.sidebar.slider('Confidence Threshold', 0.0, 1.0, 0.5, 0.01)
st.sidebar.markdown("---")
st.sidebar.write("Adjust the confidence threshold to filter predictions.")
run = st.sidebar.checkbox('Run', value=False)  # Start/Stop the simulation

# Placeholder for the video stream
FRAME_WINDOW = st.image([])

if run:
    # Main loop to simulate real-time video feed
    while run:
        # Use Streamlit's camera input widget to capture a photo
        camera_input = st.camera_input("Take a picture", key="camera_input")

        if camera_input is not None:
            # Convert the captured image to an OpenCV format
            file_bytes = np.asarray(bytearray(camera_input.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, 1)

            # Convert frame to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the image
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            # Process each detected face
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y + h, x:x + w]
                roi_gray = cv2.resize(roi_gray, (48, 48))
                roi_gray = roi_gray.astype('float32') / 255.0  # Use float32 for better precision
                roi_gray = np.expand_dims(roi_gray, axis=0)
                roi_gray = np.expand_dims(roi_gray, axis=-1)

                # Predict the emotion
                prediction = model.predict(roi_gray)
                max_index = int(np.argmax(prediction))
                predicted_emotion = emotion_labels[max_index]
                confidence = prediction[0][max_index]

                if confidence >= confidence_threshold:
                    # Draw bounding box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Draw emotion label with bold, bigger, and green text
                    cv2.putText(
                        frame, 
                        f"{predicted_emotion} ({confidence:.2f})", 
                        (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX,  # Font type
                        1.2,  # Font size (bigger)
                        (0, 255, 0),  # Color (green)
                        2,  # Thickness (bold)
                        cv2.LINE_AA  # Line type for better anti-aliasing
                    )

            # Display the frame with bounding boxes and labels
            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_column_width=True)

            # Add a short delay to mimic frame rate
            time.sleep(0.1)  # Adjust as needed for performance
        else:
            st.write("No image captured. Please use the camera to take a picture.")
else:
    st.write("Stopped. Please check 'Run' to start again.")