import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load the data
df = pd.read_excel('../media/exercises_dataset.xlsx')

# Features (angles) and labels (subfolders and exercise types)
X = df[['Right Knee Angle', 'Left Knee Angle', 'Right Groin Angle', 'Left Groin Angle',
        'Right Hip Angle', 'Left Hip Angle', 'Right Shoulder Angle', 'Left Shoulder Angle',
        'Right Torso Angle', 'Left Torso Angle', 'Right Side Angle', 'Left Side Angle',
        'Right Torso Rel Angle','Left Torso Rel Angle','Right Knee Ankle Angle','Left Knee Ankle Angle']]

# Split the labels into separate columns
y_subfolder = df['Subfolder']
y_exercise_type = df['Exercise Type']

# Handle missing values by replacing them with the column mean
X.fillna(X.mean(), inplace=True)

# Split the data into training and test sets
X_train, X_test, y_subfolder_train, y_subfolder_test, y_exercise_type_train, y_exercise_type_test = train_test_split(
    X, y_subfolder, y_exercise_type, test_size=0.2, random_state=42
)

# Initialize and train the models
model_subfolder = RandomForestClassifier(n_estimators=100, random_state=42)
model_exercise_type = RandomForestClassifier(n_estimators=100, random_state=42)

model_subfolder.fit(X_train, y_subfolder_train)
model_exercise_type.fit(X_train, y_exercise_type_train)

# Evaluate the models
y_subfolder_pred = model_subfolder.predict(X_test)
y_exercise_type_pred = model_exercise_type.predict(X_test)

print(f'Accuracy for Subfolder: {accuracy_score(y_subfolder_test, y_subfolder_pred)}')
print(f'Accuracy for Exercise Type: {accuracy_score(y_exercise_type_test, y_exercise_type_pred)}')

# Save the trained models using joblib
joblib.dump(model_subfolder, 'pose_classification_subfolder_model.pkl')
joblib.dump(model_exercise_type, 'pose_classification_exercise_type_model.pkl')
