import cv2
import mediapipe as mp

# Initialize MediaPipe Pose and drawing modules
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

# Read the image
image_path = 'img1.png'  
img = cv2.imread(image_path)

if img is None:
    print("Failed to load image.")
else:
    # Convert the image to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)

    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (cx, cy), 2, (255, 0, 0), cv2.FILLED)

    # Display the result
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
