import cv2
from keras.models import load_model
import numpy as np
import streamlit as st

# Load the pre-trained model and Haar Cascade file from the 'model' directory
model = load_model('model/FER_model.h5')
face_cascade = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')

# Check if Haar Cascade is loaded correctly
if face_cascade.empty():
    st.error("Failed to load Haar Cascade file. Please check the file path.")
    st.stop()

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Streamlit app configuration
st.set_page_config(
    page_title="Real-time Facial Emotion Recognition",
    page_icon=":smiley:",
    layout="wide",
)

st.title("🎭 Real-time Facial Emotion Recognition")
st.sidebar.title("Settings")

# Sidebar controls
run = st.sidebar.checkbox('Run')
confidence_threshold = st.sidebar.slider('Confidence Threshold', 0.0, 1.0, 0.5, 0.01)
st.sidebar.markdown("---")
st.sidebar.write("Adjust the confidence threshold to filter predictions.")

# Display area for live feed
FRAME_WINDOW = st.image([])
info_placeholder = st.empty()

# Info panel
st.sidebar.markdown("### How It Works:")
st.sidebar.write("""
This app uses a pre-trained neural network to recognize emotions from facial expressions in real-time. 
You can start or stop the video feed using the checkbox above.
""")

# Capture video feed
cap = cv2.VideoCapture(0)  # Change index if needed for your USB camera

# Streamlit app main loop
while run:
    ret, frame = cap.read()
    if not ret:
        st.error("Unable to access camera. Make sure it is connected and accessible.")
        break

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
    
    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_column_width=True)
    info_placeholder.info(f"Detected Faces: {len(faces)} | Confidence Threshold: {confidence_threshold}")

    # Exit condition to stop the video feed
    if not run:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()