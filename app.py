import cv2
from keras.models import load_model
import numpy as np
import streamlit as st

# Load the pre-trained model (make sure to have the model file saved in your directory)
model = load_model('fer_model.h5')

# Load the face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Streamlit app configuration
st.title("Real-time Facial Emotion Recognition")
run = st.checkbox('Run')
FRAME_WINDOW = st.image([])

# Capture video feed
cap = cv2.VideoCapture(0)  # 0 is the default camera

while run:
    ret, frame = cap.read()
    if not ret:
        break

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

        # Draw bounding box and emotion label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

# Release resources
cap.release()
cv2.destroyAllWindows()
