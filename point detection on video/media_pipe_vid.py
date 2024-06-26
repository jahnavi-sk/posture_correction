import cv2
import mediapipe as mp

# Initialize MediaPipe Pose and drawing modules
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

# Specify the path to your input video file
video_input_path = '../media/dk_3.mp4'
# Specify the path for the output video file
video_output_path = '../media/dk_3_media.mp4'

# Create a VideoCapture object for reading the input video
cap = cv2.VideoCapture(video_input_path)

# Check if the video was opened successfully
if not cap.isOpened():
    print("Failed to open video.")
else:
    # Get the video's width, height, and FPS
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Define the codec and create a VideoWriter object for the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to choose a codec compatible with your system
    out = cv2.VideoWriter(video_output_path, fourcc, fps, (frame_width, frame_height))

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
            # Draw landmarks and connections
            mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
            
            # Optionally, draw circles around each landmark
            for id, lm in enumerate(results.pose_landmarks.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 2, (255, 0, 0), cv2.FILLED)
        
        # Write the processed frame to the output video file
        out.write(frame)
        
        # Display the processed frame
        cv2.imshow('Video', frame)
        
        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the VideoCapture and VideoWriter objects and close windows
cap.release()
out.release()
cv2.destroyAllWindows()
c