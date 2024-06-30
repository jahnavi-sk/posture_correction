import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
import joblib

# Load the trained model
model = tf.keras.models.load_model('pose_classification_lstm_model3s.h5')

# Load the LabelEncoder
label_encoder = joblib.load('label_encoder.pkl')

# Initialize MediaPipe Pose module
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Open webcam
cap = cv2.VideoCapture(0)

# List to store real-time keypoints
realtime_keypoints = []

# Initialize variables to store last predicted label and flag for prediction display
last_predicted_label = None
display_prediction = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)

    if results.pose_landmarks:
        frame_height, frame_width, _ = frame.shape
        
        # Draw landmarks on the frame
        for lm in results.pose_landmarks.landmark:
            cx, cy = int(lm.x * frame_width), int(lm.y * frame_height)
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)  # Draw a blue circle at each landmark

        frame_keypoints = []
        for lm in results.pose_landmarks.landmark:
            h, w, c = frame.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            frame_keypoints.append((cx, cy))
        realtime_keypoints.append(frame_keypoints)
        
        # Predict after accumulating a certain number of frames
        if len(realtime_keypoints) >= 1:  # Change the number as per your requirement
            # Flatten the coordinates to match the training shape
            input_data = [np.array(realtime_keypoints).reshape(-1, 33 * 2)]
            input_data = pad_sequences(input_data, maxlen=model.input_shape[1], padding='post', dtype='float32')

            predictions = model.predict(input_data)
            predicted_label = np.argmax(predictions)
            pose_prediction = label_encoder.inverse_transform([predicted_label])[0]

            # Update last predicted label and set flag to display prediction
            last_predicted_label = pose_prediction
            display_prediction = True

            # Clear the list to start collecting new frames
            realtime_keypoints = []

    else:
        if display_prediction:
            cv2.putText(frame, last_predicted_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "NO POSE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Pose Classification', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pose.close()
