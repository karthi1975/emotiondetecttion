import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load the pre-trained model and Haar Cascade file
model_path = 'model/FER_model.h5'
model = load_model(model_path, compile=False)
face_cascade_path = 'model/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(face_cascade_path)

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

st.title("Live Facial Emotion Recognition with Streamlit")

st.write("This application uses a pre-trained FER model to detect emotions in a live video stream.")

# Start/Stop button for webcam
if 'run' not in st.session_state:
    st.session_state.run = False

def toggle_webcam():
    st.session_state.run = not st.session_state.run

st.button("Start/Stop Webcam", on_click=toggle_webcam)

FRAME_WINDOW = st.image([])

def process_frame(frame):
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = roi_gray.astype('float32') / 255.0
        roi_gray = np.expand_dims(roi_gray, axis=0)
        roi_gray = np.expand_dims(roi_gray, axis=-1)

        # Predict the emotion
        prediction = model.predict(roi_gray)
        max_index = int(np.argmax(prediction))
        predicted_emotion = emotion_labels[max_index]
        confidence = prediction[0][max_index]

        if confidence >= 0.5:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{predicted_emotion} ({confidence:.2f})", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2, cv2.LINE_AA)

    return frame

# Video stream processing
cap = None

while st.session_state.run:
    if cap is None:
        cap = cv2.VideoCapture(1)  # Use the default camera

    ret, frame = cap.read()
    if not ret:
        st.write("Failed to capture frame. Please check your webcam.")
        st.session_state.run = False
        break

    # Process frame and display
    frame = process_frame(frame)
    FRAME_WINDOW.image(frame, channels='BGR')

if cap:
    cap.release()
    cap = None