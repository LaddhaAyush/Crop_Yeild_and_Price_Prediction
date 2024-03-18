import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectFromModel

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
print("Successfully Imported!")


df_crop_production = pd.read_csv("crop_production_data.csv")
df_soil_analysis = pd.read_csv("soil_analysis_data.csv")
df_water_usage = pd.read_csv("water_usage_data.csv")

merge_soil_crop_production = df_crop_production.merge(df_soil_analysis, on = ['District'])
merge_water_soil_crop_production = merge_soil_crop_production.merge(df_water_usage, on = ['District', 'Crop'])
# print(merge_water_soil_crop_production.head())
df_agro = merge_water_soil_crop_production.copy()
print(df_agro)
print(df_agro.describe().T)
df_agro = df_agro.drop(columns = ['Production (metric tons)', 'Water Consumption (liters/hectare)'], axis = 1)
print(df_agro.info())
print(df_agro.duplicated().sum())
num_cols = ['Area (hectares)', 'pH Level', 'Organic Matter (%)', 'Nitrogen Content (kg/ha)', 'Phosphorus Content (kg/ha)',
       'Potassium Content (kg/ha)', 'Water Availability (liters/hectare)']
cat_cols = ['District', 'Crop', 'Season', 'Soil Type', 'Irrigation Method']
label = 'Yield (quintals)'
print(df_agro[cat_cols].describe().T)


df_agro[cat_cols] = df_agro[cat_cols].astype("category")


X_train, X_, y_train, y_ = train_test_split(df_agro.drop(label, axis = 1), df_agro[label].copy(), test_size = 0.3, random_state = 42)
X_val, X_test, y_val, y_test = train_test_split(X_, y_, test_size = 0.7, random_state = 42)


xgb_model = XGBRegressor(tree_method = "hist",random_state = 42, enable_categorical = True)
xgb_model.fit(X_train, y_train)




y_train_pred = xgb_model.predict(X_train)
y_val_pred = xgb_model.predict(X_val)
y_test_pred = xgb_model.predict(X_test)


# Train set evaluation
mse_train = mean_squared_error(y_train, y_train_pred)
rmse_train = np.sqrt(mse_train)
r2_train = r2_score(y_train, y_train_pred)

print(f"Mean Square Error: {mse_train:.4f}")
print(f"Root Mean Square Error: {rmse_train:.4f}")
print(f"R2 Score: {r2_train:.4f}")

# Cross validation set evaluation
mse_val = mean_squared_error(y_val, y_val_pred)
rmse_val = np.sqrt(mse_val)
r2_val = r2_score(y_val, y_val_pred)

print(f"Mean Square Error: {mse_val:.4f}")
print(f"Root Mean Square Error: {rmse_val:.4f}")
print(f"R2 Score: {r2_val:.4f}")
print()

# test set evaluation
mse_test = mean_squared_error(y_test, y_test_pred)
rmse_test = np.sqrt(mse_test)
r2_test = r2_score(y_test, y_test_pred)
print()
print(f"Mean Square Error: {mse_test:.4f}")
print(f"Root Mean Square Error: {rmse_test:.4f}")
print(f"R2 Score: {r2_test:.4f}")
print()


params = {'n_estimators': 300,
               'max_depth': 10,
               'learning_rate': 0.1,
               'subsample': 1,
               'colsample_bytree': 1}
tuned_selection_model = XGBRegressor(**params, random_state = 42, enable_categorical = True, tree_method = "hist")
tuned_selection_model.fit(X_train, y_train, eval_metric = 'rmse', eval_set = [(X_train, y_train), (X_val, y_val)], verbose = False, early_stopping_rounds=10)
y_test_tuned_pred = tuned_selection_model.predict(X_test)
# test set evaluation
mse_test = mean_squared_error(y_test, y_test_tuned_pred)
rmse_test = np.sqrt(mse_test)
r2_test = r2_score(y_test, y_test_tuned_pred)

print(f"Mean Square Error: {mse_test:.4f}")
print(f"Root Mean Square Error: {rmse_test:.4f}")
print(f"R2 Score: {r2_test:.4f}")

plt.figure(figsize = (5,4))
sns.regplot(x = y_test, y = y_test_tuned_pred, line_kws = {'color':'red'})
plt.xlabel("Yield (Quintals)")
plt.ylabel("Predictions")
plt.title(f"R2 Score: {r2_test:.4f}")
plt.show()
