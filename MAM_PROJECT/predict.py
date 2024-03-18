import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, KFold, cross_val_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost
from xgboost import XGBClassifier

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

# K-Nearest Neighbors model
k_value = 3
knn_pipeline = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k_value))
knn_pipeline.fit(X_train, y_train)

# RandomForest model
rf_pipeline = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=18))
rf_pipeline.fit(X_train, y_train)

# XGBoost model
xgb_pipeline = make_pipeline(StandardScaler(), XGBClassifier(random_state=18))
xgb_pipeline.fit(X_train, y_train)

# Save models
pickle.dump(knn_pipeline, open("knn_pipeline.pkl", "wb"))
pickle.dump(rf_pipeline, open("rf_pipeline.pkl", "wb"))
pickle.dump(xgb_pipeline, open("xgb_pipeline.pkl", "wb"))

# Take input from the user
user_input = []

for column in all_columns:
    value = float(input(f"Enter value for {column}: "))
    user_input.append(value)

# Convert user input to a NumPy array
user_input_array = np.array(user_input).reshape(1, -1)

# Standardize the user input
user_input_scaled = knn_pipeline.named_steps['standardscaler'].transform(user_input_array)

# Predict with each model
knn_prediction = knn_pipeline.predict(user_input_scaled)[0]
rf_prediction = rf_pipeline.predict(user_input_array)[0]
xgb_prediction = xgb_pipeline.predict(user_input_array)[0]

# Map the predictions to crop names
predicted_crop_knn = label_encoder.inverse_transform([knn_prediction])[0]
predicted_crop_rf = label_encoder.inverse_transform([rf_prediction])[0]
predicted_crop_xgb = label_encoder.inverse_transform([xgb_prediction])[0]

# Display the recommended crops
print(f"Recommended Crop (k-Nearest Neighbors): {predicted_crop_knn}")
print(f"Recommended Crop (RandomForest): {predicted_crop_rf}")
print(f"Recommended Crop (XGBoost): {predicted_crop_xgb}")
