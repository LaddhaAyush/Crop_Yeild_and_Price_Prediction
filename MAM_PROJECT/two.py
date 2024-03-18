import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_val_score, KFold

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, confusion_matrix
import pickle

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('Crop_recommendation.csv')
print(df.info())
all_columns = df.columns[:-1]

label_encoder = LabelEncoder()
X = df[all_columns]
y = label_encoder.fit_transform(df["label"])
print(X.shape, y.shape)
X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.2, random_state=0)
print(f"Train Data: {X_train.shape}, {y_train.shape}")
print(f"Test Data: {X_test.shape}, {y_test.shape}")

# KNN
error_rate = []
k_value = 4
knn_pipeline = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k_value))
knn_pipeline.fit(X_train, y_train)
predictions_knn = knn_pipeline.predict(X_test)
accuracy_knn = accuracy_score(y_test, predictions_knn)
print(f"Accuracy of k-Nearest Neighbors at k = {k_value}: {accuracy_knn * 100}%")

# RandomForest
rf_pipeline = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=18))
rf_pipeline.fit(X_train, y_train)
predictions_rf = rf_pipeline.predict(X_test)
accuracy_rf = accuracy_score(y_test, predictions_rf)
print(f"Accuracy of RandomForest: {accuracy_rf * 100}%")

# XGBoost
xgb_pipeline = make_pipeline(StandardScaler(), XGBClassifier(random_state=18))
xgb_pipeline.fit(X_train, y_train)
predictions_xgb = xgb_pipeline.predict(X_test)
accuracy_xgb = accuracy_score(y_test, predictions_xgb)
print(f"Accuracy of XGBoost: {accuracy_xgb * 100}%")

# K-Fold Cross-Validation
num_folds = 10

# K-Fold for k-Nearest Neighbors
knn_kfold = KFold(n_splits=num_folds, shuffle=True, random_state=0)
knn_scores = cross_val_score(knn_pipeline, X.values, y, cv=knn_kfold)
print(f"k-Nearest Neighbors Cross-Validation Accuracy: {np.mean(knn_scores) * 100}%")

# K-Fold for RandomForest
rf_kfold = KFold(n_splits=num_folds, shuffle=True, random_state=0)
rf_scores = cross_val_score(rf_pipeline, X.values, y, cv=rf_kfold)
print(f"RandomForest Cross-Validation Accuracy: {np.mean(rf_scores) * 100}%")

# K-Fold for XGBoost
xgb_kfold = KFold(n_splits=num_folds, shuffle=True, random_state=0)
xgb_scores = cross_val_score(xgb_pipeline, X.values, y, cv=xgb_kfold)
print(f"XGBoost Cross-Validation Accuracy: {np.mean(xgb_scores) * 100}%")

# Save Models
pickle.dump(knn_pipeline, open("knn_pipeline.pkl", "wb"))
pickle.dump(rf_pipeline, open("rf_pipeline.pkl", "wb"))
pickle.dump(xgb_pipeline, open("xgb_pipeline.pkl", "wb"))
print("Saved All Models")
