# # import cv2
# # import mediapipe as mp

# # # Initialize MediaPipe Pose module
# # mpPose = mp.solutions.pose
# # pose = mpPose.Pose()

# # # Specify the path to your input video file
# # video_input_path = '../media/long_2.mp4'

# # # Create a VideoCapture object for reading the input video
# # cap = cv2.VideoCapture(video_input_path)

# # # Check if the video was opened successfully
# # if not cap.isOpened():
# #     print("Failed to open video.")
# # else:
# #     # Initialize empty list to store landmark points for each frame
# #     all_landmarks = []
# #     frame_count = 0  # Counter to keep track of processed frames

# #     while True:
# #         # Read a frame from the video
# #         ret, frame = cap.read()
        
# #         if not ret:
# #             break  # Break the loop if the frame was not read correctly
        
# #         # Convert the frame to RGB
# #         imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
# #         # Process the frame with MediaPipe Pose
# #         results = pose.process(imgRGB)
        
# #         if results.pose_landmarks:
# #             # Collect landmark points
# #             landmarks_for_frame = []
# #             for lm in results.pose_landmarks.landmark:
# #                 h, w, c = frame.shape
# #                 cx, cy = int(lm.x * w), int(lm.y * h)
# #                 landmarks_for_frame.append((cx, cy))
            
# #             all_landmarks.append(landmarks_for_frame)
        
# #         frame_count += 1
        
# #         # Break the loop after processing 5 frames
# #         if frame_count == 5:
# #             break

# #     # Release the VideoCapture object
# #     cap.release()
# #     cv2.destroyAllWindows()

# # # Now all_landmarks contains the landmark points for each of the first 5 frames in the video
# # print("Number of frames processed:", len(all_landmarks))
# # for i, landmarks in enumerate(all_landmarks):
# #     print(f"Landmarks for frame {i+1}:", landmarks)


# import cv2
# import mediapipe as mp
# import pandas as pd

# # Initialize MediaPipe Pose module
# mpPose = mp.solutions.pose
# pose = mpPose.Pose()

# # Specify the path to your input video file
# video_input_path = '../media/long_2.mp4'

# # Create a VideoCapture object for reading the input video
# cap = cv2.VideoCapture(video_input_path)

# # Check if the video was opened successfully
# if not cap.isOpened():
#     print("Failed to open video.")
# else:
#     # Initialize empty lists to store landmark points for each frame
#     all_frames_landmarks = []
#     frame_count = 0  # Counter to keep track of processed frames

#     while True:
#         # Read a frame from the video
#         ret, frame = cap.read()
        
#         if not ret:
#             break  # Break the loop if the frame was not read correctly
        
#         # Convert the frame to RGB
#         imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
#         # Process the frame with MediaPipe Pose
#         results = pose.process(imgRGB)
        
#         if results.pose_landmarks:
#             # Find the index of the left hip landmark
#             left_hip_index = 24
            
#             # Get the coordinates of the left hip landmark
#             left_hip_coords = (results.pose_landmarks.landmark[left_hip_index].x * frame.shape[1], 
#                                results.pose_landmarks.landmark[left_hip_index].y * frame.shape[0])
            
#             # Collect landmark points relative to the left hip
#             landmarks_for_frame_relative = []
#             for lm in results.pose_landmarks.landmark:
#                 # Calculate relative coordinates
#                 cx_rel = int((lm.x * frame.shape[1]) - left_hip_coords[0])
#                 cy_rel = int((lm.y * frame.shape[0]) - left_hip_coords[1])
#                 landmarks_for_frame_relative.append((cx_rel, cy_rel))
            
#             all_frames_landmarks.append(landmarks_for_frame_relative)
        
#         frame_count += 1
        
#     # Release the VideoCapture object
#     cap.release()
#     cv2.destroyAllWindows()

#     # Convert landmarks data to a pandas DataFrame
#     df = pd.DataFrame(all_frames_landmarks, columns=[f"Landmark_{i+1}" for i in range(len(all_frames_landmarks[0]))])
    
#     # Specify the path for the output Excel file
#     excel_output_path = '../media/Book2.xlsx'

#     # Write DataFrame to Excel
#     df.to_excel(excel_output_path, index=False)

#     print(f"Landmarks data saved to {excel_output_path}.")






'''with video number!!'''
# import os
# import cv2
# import mediapipe as mp
# import pandas as pd

# # Initialize MediaPipe Pose module
# mpPose = mp.solutions.pose
# pose = mpPose.Pose()

# # Specify the path to the folder containing your input videos
# videos_folder = '../media/long'

# # List all video files in the folder
# video_files = [os.path.join(videos_folder, file) for file in os.listdir(videos_folder) if file.endswith('.mp4')]

# # Initialize an empty DataFrame to store all landmarks
# all_data = pd.DataFrame()

# # Process each video
# for video_num, video_file in enumerate(video_files, start=1):
#     # Create a VideoCapture object for reading the input video
#     cap = cv2.VideoCapture(video_file)
    
#     # Check if the video was opened successfully
#     if not cap.isOpened():
#         print(f"Failed to open video {video_file}. Skipping...")
#         continue
    
#     # Initialize empty lists to store landmark points for each frame
#     all_frames_landmarks = []
#     frame_count = 0  # Counter to keep track of processed frames

#     while True:
#         # Read a frame from the video
#         ret, frame = cap.read()
        
#         if not ret:
#             break  # Break the loop if the frame was not read correctly
        
#         # Convert the frame to RGB
#         imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
#         # Process the frame with MediaPipe Pose
#         results = pose.process(imgRGB)
        
#         if results.pose_landmarks:
#             # Find the index of the left hip landmark
#             left_hip_index = 24
            
#             # Get the coordinates of the left hip landmark
#             left_hip_coords = (results.pose_landmarks.landmark[left_hip_index].x * frame.shape[1], 
#                                results.pose_landmarks.landmark[left_hip_index].y * frame.shape[0])
            
#             # Collect landmark points relative to the left hip
#             landmarks_for_frame_relative = []
#             for lm in results.pose_landmarks.landmark:
#                 # Calculate relative coordinates
#                 cx_rel = int((lm.x * frame.shape[1]) - left_hip_coords[0])
#                 cy_rel = int((lm.y * frame.shape[0]) - left_hip_coords[1])
#                 landmarks_for_frame_relative.append((cx_rel, cy_rel))
            
#             all_frames_landmarks.append(landmarks_for_frame_relative)
        
#         frame_count += 1
        
#     # Release the VideoCapture object
#     cap.release()
    
#     # Convert landmarks data to a pandas DataFrame
#     df_video = pd.DataFrame(all_frames_landmarks, columns=[f"Landmark_{i+1}" for i in range(len(all_frames_landmarks[0]))])
    
#     # Add a new column for Video Number
#     df_video.insert(0, 'Video Number', video_num)
    
#     # Append this video's data to the main DataFrame
#     all_data = pd.concat([all_data, df_video], ignore_index=True)

# # Specify the path for the output Excel file
# excel_output_path = '../media/Book2.xlsx'

# # Check if the Excel file already exists
# if os.path.exists(excel_output_path):
#     # Load existing data from Excel
#     existing_data = pd.read_excel(excel_output_path)
    
#     # Append new data to the existing DataFrame
#     all_data = pd.concat([existing_data, all_data], ignore_index=True)

# # Write the combined DataFrame to Excel
# all_data.to_excel(excel_output_path, index=False)

# print(f"Landmarks data saved to {excel_output_path}.")



'''without video number'''


import os
import cv2
import mediapipe as mp
import pandas as pd

# Initialize MediaPipe Pose module
mpPose = mp.solutions.pose
pose = mpPose.Pose()

# Specify the path to the folder containing your input videos
videos_folder = '../media/long'

# List all video files in the folder
video_files = [os.path.join(videos_folder, file) for file in os.listdir(videos_folder) if file.endswith('.mp4')]

# Initialize an empty DataFrame to store all landmarks
all_data = pd.DataFrame()

# Process each video
for video_num, video_file in enumerate(video_files, start=1):
    # Create a VideoCapture object for reading the input video
    cap = cv2.VideoCapture(video_file)
    
    # Check if the video was opened successfully
    if not cap.isOpened():
        print(f"Failed to open video {video_file}. Skipping...")
        continue
    
    # Initialize empty lists to store landmark points for each frame
    all_frames_landmarks = []
    frame_count = 0  # Counter to keep track of processed frames

    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        
        if not ret:
            break  # Break the loop if the frame was not read correctly
        
        # Convert the frame to RGB
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe Pose
        results = pose.process(imgRGB)
        
        if results.pose_landmarks:
            # Find the index of the left hip landmark
            left_hip_index = 24
            
            # Get the coordinates of the left hip landmark
            left_hip_coords = (results.pose_landmarks.landmark[left_hip_index].x * frame.shape[1], 
                               results.pose_landmarks.landmark[left_hip_index].y * frame.shape[0])
            
            # Collect landmark points relative to the left hip
            landmarks_for_frame_relative = []
            for lm in results.pose_landmarks.landmark:
                # Calculate relative coordinates
                cx_rel = int((lm.x * frame.shape[1]) - left_hip_coords[0])
                cy_rel = int((lm.y * frame.shape[0]) - left_hip_coords[1])
                landmarks_for_frame_relative.append((cx_rel, cy_rel))
            
            all_frames_landmarks.append(landmarks_for_frame_relative)
        
        frame_count += 1
        
    # Release the VideoCapture object
    cap.release()
    
    # Convert landmarks data to a pandas DataFrame
    df_video = pd.DataFrame(all_frames_landmarks, columns=[f"Landmark_{i+1}" for i in range(len(all_frames_landmarks[0]))])
    
    # Append this video's data to the main DataFrame
    all_data = pd.concat([all_data, df_video], ignore_index=True)

# Specify the path for the output Excel file
excel_output_path = '../media/Book3.xlsx'

# Check if the Excel file already exists
if os.path.exists(excel_output_path):
    # Load existing data from Excel
    existing_data = pd.read_excel(excel_output_path)
    
    # Append new data to the existing DataFrame
    all_data = pd.concat([existing_data, all_data], ignore_index=True)

# Write the combined DataFrame to Excel
all_data.to_excel(excel_output_path, index=False)

print(f"Landmarks data saved to {excel_output_path}.")
