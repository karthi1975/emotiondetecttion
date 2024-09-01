import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
from tensorflow.keras.models import load_model

# Load the pre-trained model from the 'model' directory
model_path = 'model/FER_model.h5'
try:
    model = load_model(model_path)
    st.write("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Load Haar Cascade file for face detection
face_cascade_path = 'model/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(face_cascade_path)

if face_cascade.empty():
    st.error("Failed to load Haar Cascade file. Please check the file path.")
    st.stop()
else:
    st.write("Haar Cascade loaded successfully.")

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model
        self.face_cascade = face_cascade
        self.emotion_labels = emotion_labels

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        st.write("Received frame for processing.")

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Debug: Show frame dimensions
        st.write(f"Frame dimensions: {img.shape}")

        # Detect faces in the image
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # Debug: Show number of faces detected
        st.write(f"Number of faces detected: {len(faces)}")

        # Process each detected face
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi_gray = roi_gray.astype('float32') / 255.0  # Normalize pixel values
            roi_gray = np.expand_dims(roi_gray, axis=0)
            roi_gray = np.expand_dims(roi_gray, axis=-1)

            # Predict the emotion
            try:
                prediction = self.model.predict(roi_gray)
                st.write(f"Prediction: {prediction}")
            except Exception as e:
                st.error(f"Error during prediction: {e}")
                continue

            max_index = int(np.argmax(prediction))
            predicted_emotion = self.emotion_labels[max_index]
            confidence = prediction[0][max_index]

            # Debug: Show predicted emotion and confidence
            st.write(f"Predicted emotion: {predicted_emotion}, Confidence: {confidence:.2f}")

            if confidence >= 0.5:
                # Draw bounding box
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Draw emotion label with bold, bigger, and green text
                cv2.putText(
                    img, 
                    f"{predicted_emotion} ({confidence:.2f})", 
                    (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX,  # Font type
                    1.2,  # Font size (bigger)
                    (0, 255, 0),  # Color (green)
                    2,  # Thickness (bold)
                    cv2.LINE_AA  # Line type for better anti-aliasing
                )

        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.title("Real-Time Facial Emotion Recognition")

# Use WebRtcMode.SENDRECV for streaming in Streamlit Cloud
webrtc_ctx = webrtc_streamer(
    key="example",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=VideoProcessor,
    async_processing=True,
)

if not webrtc_ctx.state.playing:
    st.write("Waiting for the video feed...")