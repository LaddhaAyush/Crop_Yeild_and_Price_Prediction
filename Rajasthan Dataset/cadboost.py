import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score

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

# Define features and target
X = df_agro.drop('Yield (quintals)', axis=1)
y = df_agro['Yield (quintals)']

# Split data
X_train, X_, y_train, y_ = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_, y_, test_size=0.7, random_state=42)

# Specify categorical columns explicitly for CatBoost
cat_cols = ['District', 'Crop', 'Season', 'Soil Type', 'Irrigation Method']

# Model training
catboost_model = CatBoostRegressor(random_state=42, cat_features=cat_cols)
catboost_model.fit(X_train, y_train, verbose=False)

# Make predictions
y_train_pred = catboost_model.predict(X_train)
y_val_pred = catboost_model.predict(X_val)
y_test_pred = catboost_model.predict(X_test)

# Evaluate training set
mse_train = mean_squared_error(y_train, y_train_pred)
rmse_train = np.sqrt(mse_train)
r2_train = r2_score(y_train, y_train_pred)

print("Training Set Evaluation:")
print(f"Mean Square Error: {mse_train:.4f}")
print(f"Root Mean Square Error: {rmse_train:.4f}")
print(f"R2 Score: {r2_train:.4f}")

# Evaluate validation set
mse_val = mean_squared_error(y_val, y_val_pred)
rmse_val = np.sqrt(mse_val)
r2_val = r2_score(y_val, y_val_pred)

print("\nValidation Set Evaluation:")
print(f"Mean Square Error: {mse_val:.4f}")
print(f"Root Mean Square Error: {rmse_val:.4f}")
print(f"R2 Score: {r2_val:.4f}")

# Evaluate test set
mse_test = mean_squared_error(y_test, y_test_pred)
rmse_test = np.sqrt(mse_test)
r2_test = r2_score(y_test, y_test_pred)

print("\nTest Set Evaluation:")
print(f"Mean Square Error: {mse_test:.4f}")
print(f"Root Mean Square Error: {rmse_test:.4f}")
print(f"R2 Score: {r2_test:.4f}")

# Tune model
params = {'iterations': 300, 'depth': 10, 'learning_rate': 0.1, 'random_state': 42}
tuned_catboost_model = CatBoostRegressor(**params, cat_features=cat_cols)
tuned_catboost_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False, early_stopping_rounds=10)

# Make predictions with tuned model
y_test_tuned_pred = tuned_catboost_model.predict(X_test)

# Evaluate tuned model on test set
mse_test_tuned = mean_squared_error(y_test, y_test_tuned_pred)
rmse_test_tuned = np.sqrt(mse_test_tuned)
r2_test_tuned = r2_score(y_test, y_test_tuned_pred)

print("\nTuned Model Evaluation:")
print(f"Mean Square Error: {mse_test_tuned:.4f}")
print(f"Root Mean Square Error: {rmse_test_tuned:.4f}")
print(f"R2 Score: {r2_test_tuned:.4f}")

# Plot predictions vs actual for tuned model
plt.figure(figsize=(8, 6))
sns.regplot(x=y_test, y=y_test_tuned_pred, line_kws={'color': 'red'})
plt.xlabel("Actual Yield (Quintals)")
plt.ylabel("Predicted Yield (Quintals)")
plt.title(f"Tuned Model Predictions vs Actual (R2 Score: {r2_test_tuned:.4f})")
plt.show()
