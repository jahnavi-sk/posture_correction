import cv2
import mediapipe as mp
import math
import joblib
import pandas as pd
from collections import Counter

def calculate_angle(point1, point2, point3):
    try:
        vector1 = [point1[0] - point2[0], point1[1] - point2[1]]
        vector2 = [point3[0] - point2[0], point3[1] - point2[1]]
        dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
        magnitude1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
        magnitude2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)
        angle = math.acos(dot_product / (magnitude1 * magnitude2))
        angle_degrees = math.degrees(angle)
        return angle_degrees
    except:
        return None

# Initialize MediaPipe Pose module
mpPose = mp.solutions.pose
pose = mpPose.Pose()

# Load the trained model
model = joblib.load('pose_classification_wrist_model.pkl')

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)

    if results.pose_landmarks:
        landmarks = []
        for lm in results.pose_landmarks.landmark:
            h, w, c = frame.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            landmarks.append((cx, cy))

        # Define landmark indices
        RIGHT_WRIST = 16
        LEFT_WRIST = 15
        RIGHT_INDEX = 20
        LEFT_INDEX = 19
        RIGHT_THUMB = 22
        LEFT_THUMB = 21
        RIGHT_PINKY = 18
        LEFT_PINKY = 17
        RIGHT_ELBOW = 14
        LEFT_ELBOW = 13
        RIGHT_SHOULDER = 12
        LEFT_SHOULDER = 11
        RIGHT_HIP = 24
        LEFT_HIP = 23

        # Calculate angles
        right_wrist_index_angle = calculate_angle(landmarks[RIGHT_INDEX], landmarks[RIGHT_WRIST], landmarks[RIGHT_THUMB])
        left_wrist_index_angle = calculate_angle(landmarks[LEFT_INDEX], landmarks[LEFT_WRIST], landmarks[LEFT_THUMB])
        right_wrist_thumb_angle = calculate_angle(landmarks[RIGHT_THUMB], landmarks[RIGHT_WRIST], landmarks[RIGHT_PINKY])
        left_wrist_thumb_angle = calculate_angle(landmarks[LEFT_THUMB], landmarks[LEFT_WRIST], landmarks[LEFT_PINKY])
        right_wrist_pinky_angle = calculate_angle(landmarks[RIGHT_PINKY], landmarks[RIGHT_WRIST], landmarks[RIGHT_INDEX])
        left_wrist_pinky_angle = calculate_angle(landmarks[LEFT_PINKY], landmarks[LEFT_WRIST], landmarks[LEFT_INDEX])
        right_outer_wrist_angle = calculate_angle(landmarks[RIGHT_WRIST], landmarks[RIGHT_INDEX], landmarks[RIGHT_PINKY])
        left_outer_wrist_angle = calculate_angle(landmarks[LEFT_WRIST], landmarks[LEFT_INDEX], landmarks[LEFT_PINKY])
        right_inner_wrist_1_angle = calculate_angle(landmarks[RIGHT_ELBOW], landmarks[RIGHT_WRIST], landmarks[RIGHT_INDEX])
        left_inner_wrist_1_angle = calculate_angle(landmarks[LEFT_ELBOW], landmarks[LEFT_WRIST], landmarks[LEFT_INDEX])
        right_inner_wrist_2_angle = calculate_angle(landmarks[RIGHT_ELBOW], landmarks[RIGHT_WRIST], landmarks[RIGHT_PINKY])
        left_inner_wrist_2_angle = calculate_angle(landmarks[LEFT_ELBOW], landmarks[LEFT_WRIST], landmarks[LEFT_PINKY])
        right_torso_upper_angle = calculate_angle(landmarks[RIGHT_SHOULDER], landmarks[RIGHT_HIP], landmarks[LEFT_SHOULDER])
        left_torso_upper_angle = calculate_angle(landmarks[LEFT_SHOULDER], landmarks[LEFT_HIP], landmarks[RIGHT_SHOULDER])
        right_torso_angle = calculate_angle(landmarks[RIGHT_HIP], landmarks[RIGHT_SHOULDER], landmarks[LEFT_HIP])
        left_torso_angle = calculate_angle(landmarks[LEFT_HIP], landmarks[LEFT_SHOULDER], landmarks[RIGHT_HIP])
        right_elbow_angle = calculate_angle(landmarks[RIGHT_SHOULDER], landmarks[RIGHT_ELBOW], landmarks[RIGHT_WRIST])
        left_elbow_angle = calculate_angle(landmarks[LEFT_SHOULDER], landmarks[LEFT_ELBOW], landmarks[LEFT_WRIST])

        angles = [
            right_wrist_index_angle, left_wrist_index_angle, right_wrist_thumb_angle, left_wrist_thumb_angle,
            right_wrist_pinky_angle, left_wrist_pinky_angle, right_outer_wrist_angle, left_outer_wrist_angle,
            right_inner_wrist_1_angle, left_inner_wrist_1_angle, right_inner_wrist_2_angle, left_inner_wrist_2_angle,
            right_torso_upper_angle, left_torso_upper_angle, right_torso_angle, left_torso_angle,
            right_elbow_angle, left_elbow_angle
        ]

        angles_df = pd.DataFrame([angles], columns=[
            'Right Wrist Index Angle', 'Left Wrist Index Angle', 'Right Wrist Thumb Angle', 
            'Left Wrist Thumb Angle', 'Right Wrist Pinky Angle', 'Left Wrist Pinky Angle', 
            'Right Outer Wrist Angle', 'Left Outer Wrist Angle', 'Right Inner Wrist 1 Angle', 
            'Left Inner Wrist 1 Angle', 'Right Inner Wrist 2 Angle', 'Left Inner Wrist 2 Angle', 
            'Right Torso Upper Angle', 'Left Torso Upper Angle', 'Right Torso Angle', 
            'Left Torso Angle', 'Right Elbow Angle', 'Left Elbow Angle'
        ])
        angles_df.fillna(angles_df.mean(), inplace=True)

        predictions = model.predict(angles_df)
        pose_prediction = Counter(predictions).most_common(1)[0][0]

        cv2.putText(frame, pose_prediction, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "NO POSE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Pose Classification', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pose.close()
