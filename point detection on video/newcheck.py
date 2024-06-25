import cv2
import mediapipe as mp
import pandas as pd
from scipy.spatial.distance import euclidean

# Initialize MediaPipe Pose module
mpPose = mp.solutions.pose
pose = mpPose.Pose()

def compare_landmarks(image_landmarks, excel_landmarks):
    """
    Compare image landmarks with Excel landmarks and return matching status.
    """
    for i, (img_lmk, exc_lmk) in enumerate(zip(image_landmarks, excel_landmarks)):
        # Calculate Euclidean distance between corresponding landmarks
        dist = euclidean(img_lmk, exc_lmk)
        
        # Define a threshold distance for similarity
        threshold = 10  # Adjust this value based on your requirements
        
        # Check if the distance is within the threshold
        if dist <= threshold:
            return f"Match at Landmark {i + 1}"
    
    # If none of the landmarks matched within the threshold
    return "Doesn't Match"

# Example usage
excel_file_path = '../media/Book3.xlsx'
df = pd.read_excel(excel_file_path)

# Directly use the list of tuples without trying to map it to a string
landmarks_excel = df.iloc[1].tolist()

# Specify the path to your input image file
image_path = '../media/image.png'

# Read the image using OpenCV
image = cv2.imread(image_path)

# Convert the BGR image to RGB before processing
imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = pose.process(imgRGB)

# Check if any landmarks were detected
if results.pose_landmarks:
    # Collect landmark points relative to left hip
    landmarks_relative_to_left_hip = []
    
    # Get left hip landmark
    left_hip = results.pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_HIP]
    left_hip_x, left_hip_y = int(left_hip.x * image.shape[1]), int(left_hip.y * image.shape[0])
    
    # Iterate through all landmarks and calculate relative positions
    for lm in results.pose_landmarks.landmark:
        cx, cy = int(lm.x * image.shape[1]), int(lm.y * image.shape[0])
        relative_x, relative_y = cx - left_hip_x, cy - left_hip_y
        landmarks_relative_to_left_hip.append((relative_x, relative_y))
    
    # Compare landmarks
    result = compare_landmarks(landmarks_relative_to_left_hip, landmarks_excel)
    print(result)
else:
    print("No pose landmarks detected.")
