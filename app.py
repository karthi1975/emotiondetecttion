import cv2
import numpy as np
import tensorflow as tf
import streamlit as st
from tensorflow.keras.models import load_model

# Set Streamlit page configuration as the first Streamlit command
st.set_page_config(
    page_title="Real-time Facial Emotion Recognition",
    page_icon=":smiley:",
    layout="wide",
)

# Print TensorFlow version
st.write(f"TensorFlow version: {tf.__version__}")
st.write(f"Keras version: {tf.__version__}")

# Load the pre-trained model from the 'model' directory
model_path = 'model/FER_model.h5'
if not st.file_uploader("Upload your model file", type=["h5"]):
    st.error("Please upload the FER_model.h5 file.")
    st.stop()

# Load Haar Cascade file
face_cascade_path = 'model/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(face_cascade_path)

if face_cascade.empty():
    st.error("Failed to load Haar Cascade file. Please check the file path.")
    st.stop()

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

st.title("🎭 Real-time Facial Emotion Recognition")
st.sidebar.title("Settings")

# Sidebar controls
confidence_threshold = st.sidebar.slider('Confidence Threshold', 0.0, 1.0, 0.5, 0.01)
st.sidebar.markdown("---")
st.sidebar.write("Adjust the confidence threshold to filter predictions.")

# Use Streamlit's camera input
camera_input = st.camera_input("Take a picture")

if camera_input is not None:
    # Convert the file to an OpenCV image
    file_bytes = np.asarray(bytearray(camera_input.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = roi_gray.astype('float') / 255.0
        roi_gray = np.expand_dims(roi_gray, axis=0)
        roi_gray = np.expand_dims(roi_gray, axis=-1)

        # Predict the emotion
        prediction = model.predict(roi_gray)
        max_index = int(np.argmax(prediction))
        predicted_emotion = emotion_labels[max_index]
        confidence = prediction[0][max_index]

        if confidence >= confidence_threshold:
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Draw emotion label with bold, bigger, and green text
            cv2.putText(
                frame, 
                f"{predicted_emotion} ({confidence:.2f})", 
                (x, y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX,  # Font type
                1.5,  # Font size (bigger)
                (0, 255, 0),  # Color (green)
                3,  # Thickness (bold)
                cv2.LINE_AA  # Line type for better anti-aliasing
            )
    
    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_column_width=True)
    st.info(f"Detected Faces: {len(faces)} | Confidence Threshold: {confidence_threshold}")