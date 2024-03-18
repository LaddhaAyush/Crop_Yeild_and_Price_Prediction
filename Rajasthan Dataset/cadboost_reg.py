import warnings

import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor

print("Successfully Imported!")

# Load data
df_crop_production = pd.read_csv("crop_production_data.csv")
df_soil_analysis = pd.read_csv("soil_analysis_data.csv")
df_water_usage = pd.read_csv("water_usage_data.csv")

# Merge data
merge_soil_crop_production = df_crop_production.merge(df_soil_analysis, on='District')
merge_water_soil_crop_production = merge_soil_crop_production.merge(df_water_usage, on=['District', 'Crop'])
df_agro = merge_water_soil_crop_production.copy()

# Drop unnecessary columns
df_agro = df_agro.drop(columns=['Production (metric tons)', 'Water Consumption (liters/hectare)'], axis=1)

# Convert categorical columns to category type
cat_cols = ['District', 'Crop', 'Season', 'Soil Type', 'Irrigation Method']
df_agro[cat_cols] = df_agro[cat_cols].astype("category")

# Define features and target
X = df_agro.drop('Yield (quintals)', axis=1)
y = df_agro['Yield (quintals)']

# Split data
X_train, X_, y_train, y_ = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_, y_, test_size=0.7, random_state=42)

# Model training
catboost_model = CatBoostRegressor(random_state=42, cat_features=cat_cols)
catboost_model.fit(X_train, y_train, verbose=False)

# Take user input
user_input = {}

for column in X.columns:
    value = input(f"Enter value for {column}: ")
    user_input[column] = [value]

# Create DataFrame from user input
user_df = pd.DataFrame(user_input)

# Convert user input to appropriate data types
user_df[cat_cols] = user_df[cat_cols].astype("category")
user_df[X.select_dtypes(include=['float64', 'int64']).columns] = user_df[
    X.select_dtypes(include=['float64', 'int64']).columns].astype(float)

# Make predictions
y_user_pred = catboost_model.predict(user_df)

# Display predicted yield
print(f"Predicted Yield (quintals): {y_user_pred[0]}")
