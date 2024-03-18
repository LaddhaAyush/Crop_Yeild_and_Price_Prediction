import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
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

# Ridge Regression model
ridge_pipeline = make_pipeline(StandardScaler(), Ridge(alpha=1.0))  # You can adjust the alpha parameter for regularization
ridge_pipeline.fit(X_train, y_train)

# Save the Ridge Regression model
pickle.dump(ridge_pipeline, open("ridge_pipeline.pkl", "wb"))

# Take input from the user
user_input = []

for column in all_columns:
    value = float(input(f"Enter value for {column}: "))
    user_input.append(value)

# Convert user input to a NumPy array
user_input_array = np.array(user_input).reshape(1, -1)

# Standardize the user input
user_input_scaled = ridge_pipeline.named_steps['standardscaler'].transform(user_input_array)

# Predict with the Ridge Regression model
ridge_prediction = ridge_pipeline.predict(user_input_scaled)[0]

# Round the prediction to the nearest integer (assuming it represents a class)
predicted_crop_ridge = round(ridge_prediction)

# Map the prediction to crop name
predicted_crop_ridge = label_encoder.inverse_transform([predicted_crop_ridge])[0]

# Display the recommended crop
print(f"Recommended Crop (Ridge Regression): {predicted_crop_ridge}")

# Evaluate accuracy on the test set (Note: Ridge Regression is not typically evaluated using accuracy)
test_predictions_ridge = ridge_pipeline.predict(X_test)
test_accuracy_ridge = accuracy_score(y_test, np.round(test_predictions_ridge))
print(f"Ridge Regression Model Accuracy on Test Set: {test_accuracy_ridge * 100}%")
