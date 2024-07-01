import cv2
import mediapipe as mp
import math
import pandas as pd
import os

def calculate_knee_ankle_vertical_angle(knee, ankle):
    try:
        dx = knee[0] - ankle[0]
        dy = knee[1] - ankle[1]
        angle_radians = math.atan2(abs(dx), abs(dy))
        angle_degrees = math.degrees(angle_radians)
        return angle_degrees
    except:
        return None

def calculate_torso_angle(landmark1, landmark2):
    try:
        angle = math.degrees(math.atan2(abs(landmark1[1] - landmark2[1]), abs(landmark1[0] - landmark2[0])))
        return angle
    except:
        return None

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

# Specify the path to your input folder containing subfolders with videos
input_folder_path = '../media/exercise_set'

# List to store the angles
angles_list = []

# Traverse the directories
for exercise_type in os.listdir(input_folder_path):
    exercise_path = os.path.join(input_folder_path, exercise_type)
    if os.path.isdir(exercise_path):
        for subfolder_name in os.listdir(exercise_path):
            subfolder_path = os.path.join(exercise_path, subfolder_name)
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
                            LEFT_PINKY = 17
                            RIGHT_PINKY = 18
                            LEFT_INDEX = 19
                            RIGHT_INDEX = 20
                            LEFT_THUMB = 21
                            RIGHT_THUMB = 21
                            
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
                            right_wrist_index_angle = calculate_angle(landmarks[RIGHT_ELBOW], landmarks[RIGHT_WRIST], landmarks[RIGHT_INDEX])
                            right_wrist_thumb_angle = calculate_angle(landmarks[RIGHT_ELBOW], landmarks[RIGHT_WRIST], landmarks[RIGHT_THUMB])
                            right_wrist_pinky_angle = calculate_angle(landmarks[RIGHT_ELBOW], landmarks[RIGHT_WRIST], landmarks[RIGHT_PINKY])
                            right_outer_wrist_angle = calculate_angle(landmarks[RIGHT_PINKY], landmarks[RIGHT_WRIST], landmarks[RIGHT_THUMB])
                            right_inner_wrist_1_angle = calculate_angle(landmarks[RIGHT_PINKY], landmarks[RIGHT_WRIST], landmarks[RIGHT_INDEX])
                            right_inner_wrist_2_angle = calculate_angle(landmarks[RIGHT_INDEX], landmarks[RIGHT_WRIST], landmarks[RIGHT_THUMB])
                            left_wrist_index_angle = calculate_angle(landmarks[LEFT_ELBOW], landmarks[LEFT_WRIST], landmarks[LEFT_INDEX])
                            left_wrist_thumb_angle = calculate_angle(landmarks[LEFT_ELBOW], landmarks[LEFT_WRIST], landmarks[LEFT_THUMB])
                            left_wrist_pinky_angle = calculate_angle(landmarks[LEFT_ELBOW], landmarks[LEFT_WRIST], landmarks[LEFT_PINKY])
                            left_outer_wrist_angle = calculate_angle(landmarks[LEFT_PINKY], landmarks[LEFT_WRIST], landmarks[LEFT_THUMB])
                            left_inner_wrist_1_angle = calculate_angle(landmarks[LEFT_PINKY], landmarks[LEFT_WRIST], landmarks[LEFT_INDEX])
                            left_inner_wrist_2_angle = calculate_angle(landmarks[LEFT_INDEX], landmarks[LEFT_WRIST], landmarks[LEFT_THUMB])
                            right_torso_upper_angle = calculate_angle(landmarks[RIGHT_HIP], landmarks[RIGHT_SHOULDER], landmarks[LEFT_SHOULDER])
                            left_torso_upper_angle = calculate_angle(landmarks[LEFT_HIP], landmarks[LEFT_SHOULDER], landmarks[RIGHT_SHOULDER])
                            right_groin_angle = calculate_angle(landmarks[LEFT_KNEE], landmarks[LEFT_HIP], landmarks[RIGHT_KNEE])
                            left_groin_angle = calculate_angle(landmarks[RIGHT_KNEE], landmarks[RIGHT_HIP], landmarks[LEFT_KNEE])
                            right_torsorel_angle = calculate_torso_angle(landmarks[RIGHT_SHOULDER], landmarks[RIGHT_HIP])
                            left_torsorel_angle = calculate_torso_angle(landmarks[LEFT_SHOULDER], landmarks[LEFT_HIP])
                            right_knee_ankle_angle = calculate_knee_ankle_vertical_angle(landmarks[RIGHT_KNEE], landmarks[RIGHT_ANKLE])
                            left_knee_ankle_angle = calculate_knee_ankle_vertical_angle(landmarks[LEFT_KNEE], landmarks[LEFT_ANKLE])
                            
                            angles_list.append([
                                exercise_type, subfolder_name, video_number, right_knee_angle, left_knee_angle, right_groin_angle, left_groin_angle, 
                                right_hip_angle, left_hip_angle, right_shoulder_angle, left_shoulder_angle, right_torso_angle, left_torso_angle, 
                                right_side, left_side, right_torsorel_angle, left_torsorel_angle, right_knee_ankle_angle, left_knee_ankle_angle, right_elbow_angle,left_elbow_angle
                            ])
                    cap.release()

cv2.destroyAllWindows()
pose.close()

df = pd.DataFrame(angles_list, columns=[
    'Exercise Type', 'Subfolder', 'Video Number', 'Right Knee Angle', 'Left Knee Angle', 'Right Groin Angle', 'Left Groin Angle', 
    'Right Hip Angle', 'Left Hip Angle', 'Right Shoulder Angle', 'Left Shoulder Angle','Right Torso Angle', 'Left Torso Angle', 
    'Right Side Angle', 'Left Side Angle', 'Right Torso Rel Angle', 'Left Torso Rel Angle', 'Right Knee Ankle Angle', 'Left Knee Ankle Angle','Right Elbow Angle', 'Left Elbow Angle'
])

excel_output_path = '../media/exercises_dataset.xlsx'
df.to_excel(excel_output_path, index=False)

print(f"Angles have been saved to {excel_output_path}")
