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
df_agro.to_csv("modified_agro_data.csv", index=False)


