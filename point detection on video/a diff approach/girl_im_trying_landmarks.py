import cv2
import mediapipe as mp
import os
import numpy as np

# Initialize MediaPipe Pose module
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Specify the path to your input folder containing subfolders with videos
input_folder_path = '../media/hip_vid'

# Dictionary to store the keypoints
data_dict = {}

# Traverse the directories
for subfolder_name in os.listdir(input_folder_path):
    subfolder_path = os.path.join(input_folder_path, subfolder_name)
    if os.path.isdir(subfolder_path):
        data_dict[subfolder_name] = []
        for video_file in os.listdir(subfolder_path):
            video_path = os.path.join(subfolder_path, video_file)
            cap = cv2.VideoCapture(video_path)
            video_keypoints = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert the frame to RGB
                imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process the frame with MediaPipe Pose
                results = pose.process(imgRGB)
                
                if results.pose_landmarks:
                    frame_keypoints = []
                    for lm in results.pose_landmarks.landmark:
                        h, w, c = frame.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        frame_keypoints.append((cx, cy))
                    video_keypoints.append(frame_keypoints)
            
            data_dict[subfolder_name].append(video_keypoints)
            cap.release()

cv2.destroyAllWindows()
pose.close()

# Save the keypoints data to a file (e.g., using numpy)
np.save('keypoints_data.npy', data_dict)
