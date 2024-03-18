import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('Crop_recommendation.csv')

# Display dataset information
print(df.info())

# Extract features and labels
all_columns = df.columns[:-1]
label_encoder = LabelEncoder()
X = df[all_columns]
y = label_encoder.fit_transform(df["label"])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.2, random_state=0)

# SVM model
svm_pipeline = make_pipeline(StandardScaler(), SVC(kernel='rbf', random_state=0))
svm_pipeline.fit(X_train, y_train)

# Save the SVM model
pickle.dump(svm_pipeline, open("svm_pipeline.pkl", "wb"))

# Take input from the user
user_input = []

for column in all_columns:
    value = float(input(f"Enter value for {column}: "))
    user_input.append(value)

# Convert user input to a NumPy array
user_input_array = np.array(user_input).reshape(1, -1)

# Standardize the user input
user_input_scaled = svm_pipeline.named_steps['standardscaler'].transform(user_input_array)

# Predict with the SVM model
svm_prediction = svm_pipeline.predict(user_input_scaled)[0]

# Map the prediction to crop name
predicted_crop_svm = label_encoder.inverse_transform([svm_prediction])[0]

# Display the recommended crop
print(f"Recommended Crop (SVM): {predicted_crop_svm}")

# Evaluate accuracy on the test set
test_predictions = svm_pipeline.predict(X_test)
test_accuracy = accuracy_score(y_test, test_predictions)
print(f"SVM Model Accuracy on Test Set: {test_accuracy * 100}%")
