import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the data
df = pd.read_excel('../media/angles_output_hip6.xlsx')

# Features (angles) and labels (poses)
X = df[['Right Knee Angle', 'Left Knee Angle', 'Right Groin Angle', 'Left Groin Angle',
        'Right Hip Angle', 'Left Hip Angle', 'Right Shoulder Angle', 'Left Shoulder Angle',
        'Right Torso Angle', 'Left Torso Angle', 'Right Side Angle', 'Left Side Angle','Right Torso Rel Angle','Left Torso Rel Angle','Right Knee Ankle Angle','Left Knee Ankle Angle']]

# X = df[['Right Wrist Index Angle', 'Left Wrist Index Angle', 'Right Wrist Thumb Angle', 
#         'Left Wrist Thumb Angle', 'Right Wrist Pinky Angle', 'Left Wrist Pinky Angle', 
#         'Right Outer Wrist Angle', 'Left Outer Wrist Angle', 'Right Inner Wrist 1 Angle', 
#         'Left Inner Wrist 1 Angle', 'Right Inner Wrist 2 Angle', 'Left Inner Wrist 2 Angle', 
#         'Right Torso Upper Angle', 'Left Torso Upper Angle', 'Right Torso Angle', 
#         'Left Torso Angle', 'Right Elbow Angle', 'Left Elbow Angle']]


y = df['Subfolder']

# Handle missing values by replacing them with the column mean
X.fillna(X.mean(), inplace=True)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

# Save the trained model using joblib
import joblib
joblib.dump(model, 'pose_classification_hip6_model.pkl')
