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
model = joblib.load('pose_classification_model.pkl')

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

        RIGHT_HIP = 24
        RIGHT_KNEE = 26
        RIGHT_ANKLE = 28
        LEFT_HIP = 23
        LEFT_KNEE = 25
        LEFT_ANKLE = 27
        RIGHT_SHOULDER = 12
        RIGHT_ELBOW = 14
        RIGHT_WRIST = 16
        LEFT_SHOULDER = 11
        LEFT_ELBOW = 13
        LEFT_WRIST = 15

        right_knee_angle = calculate_angle(landmarks[RIGHT_HIP], landmarks[RIGHT_KNEE], landmarks[RIGHT_ANKLE])
        left_knee_angle = calculate_angle(landmarks[LEFT_HIP], landmarks[LEFT_KNEE], landmarks[LEFT_ANKLE])
        right_elbow_angle = calculate_angle(landmarks[RIGHT_SHOULDER], landmarks[RIGHT_ELBOW], landmarks[RIGHT_WRIST])
        left_elbow_angle = calculate_angle(landmarks[LEFT_SHOULDER], landmarks[LEFT_ELBOW], landmarks[LEFT_WRIST])
        right_hip_angle = calculate_angle(landmarks[RIGHT_KNEE], landmarks[RIGHT_HIP], landmarks[RIGHT_SHOULDER])
        left_hip_angle = calculate_angle(landmarks[LEFT_KNEE], landmarks[LEFT_HIP], landmarks[LEFT_SHOULDER])
        left_shoulder_angle = calculate_angle(landmarks[LEFT_ELBOW], landmarks[LEFT_SHOULDER], landmarks[LEFT_HIP])
        right_shoulder_angle = calculate_angle(landmarks[RIGHT_ELBOW], landmarks[RIGHT_SHOULDER], landmarks[RIGHT_HIP])
        right_torso_angle = calculate_angle(landmarks[RIGHT_SHOULDER], landmarks[RIGHT_HIP], landmarks[LEFT_HIP])
        left_torso_angle = calculate_angle(landmarks[LEFT_SHOULDER], landmarks[LEFT_HIP], landmarks[RIGHT_HIP])
        right_side = calculate_angle(landmarks[RIGHT_SHOULDER], landmarks[RIGHT_HIP], landmarks[RIGHT_KNEE])
        left_side = calculate_angle(landmarks[LEFT_SHOULDER], landmarks[LEFT_HIP], landmarks[LEFT_KNEE])

        angles = [right_knee_angle, left_knee_angle, right_elbow_angle, left_elbow_angle, right_hip_angle, left_hip_angle, right_shoulder_angle, left_shoulder_angle, right_torso_angle, left_torso_angle, right_side, left_side]

        angles_df = pd.DataFrame([angles], columns=['Right Knee Angle', 'Left Knee Angle', 'Right Elbow Angle', 'Left Elbow Angle', 
                                                    'Right Hip Angle', 'Left Hip Angle', 'Right Shoulder Angle', 'Left Shoulder Angle', 
                                                    'Right Torso Angle', 'Left Torso Angle', 'Right Side Angle', 'Left Side Angle'])
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
