import cv2
import mediapipe as mp
import math
import pandas as pd

def calculate_angle(point1, point2, point3):
    # Calculate the angle between point1, point2, and point3
    # point2 is the vertex point of the angle
    vector1 = [point1[0] - point2[0], point1[1] - point2[1]]
    vector2 = [point3[0] - point2[0], point3[1] - point2[1]]
    
    # Calculate the dot product and the magnitudes of the vectors
    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
    magnitude1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
    magnitude2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)
    
    # Calculate the angle in radians and then convert to degrees
    angle = math.acos(dot_product / (magnitude1 * magnitude2))
    angle_degrees = math.degrees(angle)
    
    return angle_degrees

# Initialize MediaPipe Pose module
mpPose = mp.solutions.pose
pose = mpPose.Pose()

# Specify the path to your input video file
video_input_path = '../media/bridge_1.mp4'
cap = cv2.VideoCapture(video_input_path)

# List to store the angles
angles_list = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the frame to RGB
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame with MediaPipe Pose
    results = pose.process(imgRGB)
    
    if results.pose_landmarks:
        # Collect landmark points
        landmarks = []
        for lm in results.pose_landmarks.landmark:
            h, w, c = frame.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            landmarks.append((cx, cy))
        
        # Define landmark indices for right and left knee and ankle
        RIGHT_HIP = 24
        RIGHT_KNEE = 26
        RIGHT_ANKLE = 28
        LEFT_HIP = 23
        LEFT_KNEE = 25
        LEFT_ANKLE = 27
        RIGHT_SHOULDER = 12
        RIGHT_ELBOW = 14
        RIGHT_WRIST = 16
        
        # Calculate angles for the right leg
        right_knee_angle = calculate_angle(landmarks[RIGHT_HIP], landmarks[RIGHT_KNEE], landmarks[RIGHT_ANKLE])
        left_knee_angle = calculate_angle(landmarks[LEFT_HIP], landmarks[LEFT_KNEE], landmarks[LEFT_ANKLE])
        right_elbow_angle = calculate_angle(landmarks[RIGHT_SHOULDER], landmarks[RIGHT_ELBOW], landmarks[RIGHT_WRIST])
        torso_angle = calculate_angle(landmarks[RIGHT_KNEE],landmarks[RIGHT_HIP], landmarks[RIGHT_SHOULDER])
        # Append angles to the list
        angles_list.append([right_knee_angle, left_knee_angle, right_elbow_angle, torso_angle])
        
       

# Release resources
cap.release()
cv2.destroyAllWindows()
pose.close()

# Create a DataFrame from the angles list
df = pd.DataFrame(angles_list, columns=['Right Knee Angle', 'Left Knee Angle', 'Right Elbow Angle','Torso Angle'])

# Write the DataFrame to an Excel file
excel_output_path = '../media/angles_output.xlsx'
df.to_excel(excel_output_path, index=False)

print(f"Angles have been saved to {excel_output_path}")
