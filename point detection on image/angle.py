import cv2
import mediapipe as mp
import math


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

# Specify the path to your input image file
image_input_path = '../media/1.png'

# Load the image
image = cv2.imread(image_input_path)

if image is None:
    print("Failed to load image.")
else:
    # Convert the image to RGB
    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image with MediaPipe Pose
    results = pose.process(imgRGB)
    
    if results.pose_landmarks:
        # Collect landmark points
        landmarks = []
        for lm in results.pose_landmarks.landmark:
            h, w, c = image.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            landmarks.append((cx, cy))
        
        # Define landmark indices for right and left knee and ankle
        RIGHT_HIP = 24
        RIGHT_KNEE = 26
        RIGHT_ANKLE = 28
        LEFT_HIP = 23
        LEFT_KNEE = 25
        LEFT_ANKLE = 27
        
        # Calculate angles for the right leg
        right_knee_angle = calculate_angle(landmarks[RIGHT_HIP], landmarks[RIGHT_KNEE], landmarks[RIGHT_ANKLE])
        left_knee_angle = calculate_angle(landmarks[LEFT_HIP], landmarks[LEFT_KNEE], landmarks[LEFT_ANKLE])
        print()
        print(f"Angle between right hip, right knee, and right ankle: {right_knee_angle:.2f} degrees")
        print(f"Angle between left hip, left knee, and left ankle: {left_knee_angle:.2f} degrees")

# Release the Pose object
pose.close()
