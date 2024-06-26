import cv2
import mediapipe as mp
import math
import pandas as pd
import os

def calculate_angle(point1, point2, point3):
    try:
        # Calculate the angle between point1, point2, and point3
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
    except:
        return None

# Initialize MediaPipe Pose module
mpPose = mp.solutions.pose
pose = mpPose.Pose()

# Specify the path to your input folder containing subfolders with videos
input_folder_path = '../media/back_vid'

# List to store the angles
angles_list = []

# Traverse the directories
for subfolder_name in os.listdir(input_folder_path):
    subfolder_path = os.path.join(input_folder_path, subfolder_name)
    if os.path.isdir(subfolder_path):
        for video_number, video_file in enumerate(os.listdir(subfolder_path), start=1):
            video_path = os.path.join(subfolder_path, video_file)
            cap = cv2.VideoCapture(video_path)
            
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
                    LEFT_SHOULDER = 11
                    LEFT_ELBOW = 13
                    LEFT_WRIST = 15
                    
                    # Calculate angles for the right leg
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

                    # Append angles to the list
                    angles_list.append([subfolder_name, video_number, right_knee_angle, left_knee_angle, right_elbow_angle, left_elbow_angle, right_hip_angle, left_hip_angle, right_shoulder_angle, left_shoulder_angle, right_torso_angle, left_torso_angle, right_side, left_side])
                    
                   
            # Release resources for the current video
            cap.release()

cv2.destroyAllWindows()
pose.close()

# Create a DataFrame from the angles list
df = pd.DataFrame(angles_list, columns=['Subfolder', 'Video Number', 'Right Knee Angle', 'Left Knee Angle', 'Right Elbow Angle', 'Left Elbow Angle', 'Right Hip Angle', 'Left Hip Angle', 'Right Shoulder Angle', 'Left Shoulder Angle','Right Torso Angle', 'Left Torso Angle', 'Right Side Angle', 'Left Side angle' ])

# Write the DataFrame to an Excel file
excel_output_path = '../media/angles_output.xlsx'
df.to_excel(excel_output_path, index=False)

print(f"Angles have been saved to {excel_output_path}")
