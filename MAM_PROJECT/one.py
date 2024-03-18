import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

from sklearn.neighbors import  KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost
from xgboost import XGBClassifier

from sklearn.metrics import confusion_matrix
import pickle

import warnings
warnings.filterwarnings("ignore")
df = pd.read_csv('Crop_recommendation.csv')
print(df.info())
all_columns = df.columns[:-1]


# for column in all_columns:
#     plt.figure(figsize=(19,7))
#     sns.barplot(x = "label", y = column, data = df)
#     plt.xticks(rotation=90)
#     plt.title(f"{column} vs Crop Type")
#     plt.show()
label_dict = {}



label_encoder = LabelEncoder()
X = df[all_columns]
y = label_encoder.fit_transform(df["label"])
print(X.shape, y.shape)
X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size = 0.2, random_state = 0)
print(f"Train Data: {X_train.shape}, {y_train.shape}")
print(f"Train Data: {X_test.shape}, {y_test.shape}")

from sklearn.metrics import accuracy_score, confusion_matrix

error_rate = []
k_value = 4

knn_pipeline = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k_value))
knn_pipeline.fit(X_train, y_train)
predictions = knn_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy at k = {k_value} is {accuracy}")

error_rate = np.mean(predictions != y_test)
print("Error Rate at k =", k_value, "is", error_rate)


rf_pipeline = make_pipeline(StandardScaler(), RandomForestClassifier(random_state = 18))
rf_pipeline.fit(X_train, y_train)
print()
# Accuray On Whole Data
predictions_rf = rf_pipeline.predict(X.values)
accuracy_rf = accuracy_score(y, predictions_rf)
print(f"Accuracy of RF: {accuracy_rf*100}%")



xgb_pipeline = make_pipeline(StandardScaler(), XGBClassifier(random_state = 18))
xgb_pipeline.fit(X_train, y_train)
predictions_xgb = xgb_pipeline.predict(X.values)
accuracy_xgb = accuracy_score(y, predictions_xgb)
print(f"Accuracy of XGBoost: {accuracy_xgb*100}%")


pickle.dump(knn_pipeline, open("knn_pipeline.pkl", "wb"))
pickle.dump(rf_pipeline, open("rf_pipeline.pkl", "wb"))
pickle.dump(xgb_pipeline, open("xgb_pipeline.pkl", "wb"))
pickle.dump(label_dict, open("label_dictionary.pkl", "wb"))
print("Saved All Models")