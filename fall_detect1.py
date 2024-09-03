import cv2
import mediapipe as mp
import numpy as np
import os
import time

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, 
                    model_complexity=1, 
                    min_detection_confidence=0.5, 
                    min_tracking_confidence=0.5)

# Initialize MediaPipe drawing tools
mp_drawing = mp.solutions.drawing_utils

# Initialize variables for velocity check
prev_head_y = None
fall_speed_threshold = 0.01  # Threshold for speed to detect a rapid fall

def is_fall_detected(landmarks, prev_head_y):
    """
    Determine if a fall is detected based on landmarks and speed.
    """
    if landmarks:
        head_y = landmarks[mp_pose.PoseLandmark.NOSE].y
        hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP].y
        shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y
        
        # Calculate speed of head moving downwards
        speed = 0
        if prev_head_y is not None:
            speed = head_y - prev_head_y
        
        # Check if head is below hips and moving down rapidly
        if head_y > hip_y + 0.03 and speed > fall_speed_threshold:
            return True, head_y
        
        # Additional check: if shoulders are close to hips, might be a sign of lying down
        if head_y > shoulder_y and head_y > hip_y + 0.03:
            return True, head_y

    return False, None

# Initialize the camera
cap = cv2.VideoCapture(0)

fall_detected = False
fall_frame_count = 0  # Count frames where a fall is detected

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Pose
    results = pose.process(rgb_frame)

    # Extract pose landmarks
    landmarks = results.pose_landmarks

    # Check if a fall is detected
    if landmarks:
        detected, new_head_y = is_fall_detected(landmarks.landmark, prev_head_y)
        prev_head_y = new_head_y if new_head_y is not None else prev_head_y
        
        if detected:
            fall_frame_count += 1
            if fall_frame_count > 2:  # Confirm fall if detected in 3 consecutive frames
                fall_detected = True
                # Visual alert - draw bold red text
                text = 'Fall Detected!'
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.5
                font_thickness = 4
                cv2.putText(frame, text, (50, 50), font, font_scale, (0, 0, 255), font_thickness, cv2.LINE_AA)
        else:
            fall_detected = False
            fall_frame_count = 0
    else:
        prev_head_y = None

    # Draw pose landmarks on the frame
    if landmarks:
        mp_drawing.draw_landmarks(frame, landmarks, mp_pose.POSE_CONNECTIONS)

    # Display the frame
    cv2.imshow('Fall Detection with MediaPipe', frame)

    # Play a sound alert if a fall is detected
    if fall_detected:
        os.system('say "Fall detected!"')  # Voice alert (macOS specific)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()

# Close MediaPipe resources
pose.close()
