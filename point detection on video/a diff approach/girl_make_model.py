import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from keras._tf_keras.keras.models import Sequential 
from keras._tf_keras.keras.layers import LSTM, Dense, Masking, Dropout
import joblib
# Load the keypoints data
data_dict = np.load('keypoints_data.npy', allow_pickle=True).item()

# Convert the data into a suitable format for training
X = []
y = []
for label, videos in data_dict.items():
    for video in videos:
        X.append(video)
        y.append(label)

# Convert labels to numerical values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Pad sequences to the same length
X = pad_sequences(X, padding='post', dtype='float32')

# Reshape input data to flatten the x and y coordinates
num_samples, num_frames, num_landmarks, num_coords = X.shape
X = X.reshape(num_samples, num_frames, num_landmarks * num_coords)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the LSTM model
input_shape = (X.shape[1], X.shape[2])

model = Sequential()
model.add(Masking(mask_value=0., input_shape=input_shape))
model.add(LSTM(64, return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(64))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(np.unique(y)), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_test, y_test))

# Print the final accuracy
train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Testing Accuracy: {test_accuracy:.4f}")

# Save the trained model
model.save('pose_classification_lstm_model3s.h5')
joblib.dump(label_encoder, 'label_encoder.pkl')
