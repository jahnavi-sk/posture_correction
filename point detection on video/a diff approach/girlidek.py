import cv2
import mediapipe as mp
import math
import joblib
import pandas as pd
from collections import Counter

# Function to calculate angles
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

# Function to calculate knee-ankle vertical angle
def calculate_knee_ankle_vertical_angle(knee, ankle):
    try:
        dx = knee[0] - ankle[0]
        dy = knee[1] - ankle[1]
        angle_radians = math.atan2(abs(dx), abs(dy))
        angle_degrees = math.degrees(angle_radians)
        return angle_degrees
    except:
        return None

# Function to calculate torso angle
def calculate_torso_angle(landmark1, landmark2):
    try:
        angle = math.degrees(math.atan2(abs(landmark1[1] - landmark2[1]), abs(landmark1[0] - landmark2[0])))
        return angle
    except:
        return None

# Initialize MediaPipe Pose module
mpPose = mp.solutions.pose
pose = mpPose.Pose()

# Load the trained model
model = joblib.load('pose_classification_hip6_model.pkl')

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

        # Calculate angles
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
        right_groin_angle = calculate_angle(landmarks[LEFT_KNEE], landmarks[LEFT_HIP], landmarks[RIGHT_KNEE])
        left_groin_angle = calculate_angle(landmarks[RIGHT_KNEE], landmarks[RIGHT_HIP], landmarks[LEFT_KNEE])
        right_torsorel_angle = calculate_torso_angle(landmarks[RIGHT_SHOULDER], landmarks[RIGHT_HIP])
        left_torsorel_angle = calculate_torso_angle(landmarks[LEFT_SHOULDER], landmarks[LEFT_HIP])
        right_knee_ankle_angle = calculate_knee_ankle_vertical_angle(landmarks[RIGHT_KNEE], landmarks[RIGHT_ANKLE])
        left_knee_ankle_angle = calculate_knee_ankle_vertical_angle(landmarks[LEFT_KNEE], landmarks[LEFT_ANKLE])

        # Create a list of angles
        angles = [right_knee_angle, left_knee_angle, right_groin_angle, left_groin_angle, right_hip_angle, left_hip_angle,
                  right_shoulder_angle, left_shoulder_angle, right_torso_angle, left_torso_angle, right_side, left_side,
                  right_torsorel_angle, left_torsorel_angle, right_knee_ankle_angle, left_knee_ankle_angle]

        # Create a DataFrame with angles
        angles_df = pd.DataFrame([angles], columns=['Right Knee Angle', 'Left Knee Angle', 'Right Groin Angle', 'Left Groin Angle',
                                                    'Right Hip Angle', 'Left Hip Angle', 'Right Shoulder Angle', 'Left Shoulder Angle',
                                                    'Right Torso Angle', 'Left Torso Angle', 'Right Side Angle', 'Left Side Angle',
                                                    'Right Torso Rel Angle', 'Left Torso Rel Angle', 'Right Knee Ankle Angle', 'Left Knee Ankle Angle'])

        # Fill missing values with mean
        angles_df.fillna(angles_df.mean(), inplace=True)

        # Predict exercise using the model
        predictions = model.predict(angles_df)
        pose_prediction = Counter(predictions).most_common(1)[0][0]

        # Draw connections between landmarks (joints)
        connections = [[RIGHT_HIP, RIGHT_KNEE], [RIGHT_KNEE, RIGHT_ANKLE], [LEFT_HIP, LEFT_KNEE], [LEFT_KNEE, LEFT_ANKLE],
                       [RIGHT_SHOULDER, RIGHT_ELBOW], [RIGHT_ELBOW, RIGHT_WRIST], [LEFT_SHOULDER, LEFT_ELBOW], [LEFT_ELBOW, LEFT_WRIST],
                       [RIGHT_SHOULDER, RIGHT_HIP], [LEFT_SHOULDER, LEFT_HIP], [RIGHT_HIP, LEFT_HIP],
                       [RIGHT_SHOULDER, RIGHT_KNEE], [LEFT_SHOULDER, LEFT_KNEE], [RIGHT_KNEE, LEFT_KNEE]]

        for connection in connections:
            cv2.line(frame, landmarks[connection[0]], landmarks[connection[1]], (0, 255, 0), 3)

        # Draw landmarks on the frame
        for lm in results.pose_landmarks.landmark:
            h, w, c = frame.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 8, (255, 0, 0), cv2.FILLED)

        # Display exercise prediction
        cv2.putText(frame, f"Exercise: {pose_prediction}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    else:
        cv2.putText(frame, "NO POSE DETECTED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Exercise Classification', frame)

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
pose.close()
