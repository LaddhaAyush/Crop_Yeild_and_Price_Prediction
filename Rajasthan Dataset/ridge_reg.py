import warnings

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

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

# Define features and target
X = df_agro.drop('Yield (quintals)', axis=1)
y = df_agro['Yield (quintals)']

# Split data
X_train, X_, y_train, y_ = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_, y_, test_size=0.7, random_state=42)

# Preprocessing
numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
categorical_features = X.select_dtypes(include=['category']).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Append ridge regression to preprocessing steps
ridge_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('ridge', Ridge())])

# Model training
ridge_pipeline.fit(X_train, y_train)

# Take user input
user_input = {}

for column in X.columns:
    value = input(f"Enter value for {column}: ")
    user_input[column] = [value]

# Create DataFrame from user input
user_df = pd.DataFrame(user_input)

# Make predictions
y_user_pred = ridge_pipeline.predict(user_df)

# Display predicted yield
print(f"Predicted Yield (quintals): {y_user_pred[0]}")
