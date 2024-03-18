import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

print("Successfully Imported!")

df_crop_production = pd.read_csv("crop_production_data.csv")
df_soil_analysis = pd.read_csv("soil_analysis_data.csv")
df_water_usage = pd.read_csv("water_usage_data.csv")

merge_soil_crop_production = df_crop_production.merge(df_soil_analysis, on=['District'])
merge_water_soil_crop_production = merge_soil_crop_production.merge(df_water_usage, on=['District', 'Crop'])
# print(merge_water_soil_crop_production.head())
df_agro = merge_water_soil_crop_production.copy()
# print(df_agro)
# print(df_agro.describe().T)
df_agro = df_agro.drop(columns=['Production (metric tons)', 'Water Consumption (liters/hectare)'], axis=1)
# print(df_agro.info())
# print(df_agro.duplicated().sum())
num_cols = ['Area (hectares)', 'pH Level', 'Organic Matter (%)', 'Nitrogen Content (kg/ha)',
            'Phosphorus Content (kg/ha)',
            'Potassium Content (kg/ha)', 'Water Availability (liters/hectare)']
cat_cols = ['District', 'Crop', 'Season', 'Soil Type', 'Irrigation Method']
label = 'Yield (quintals)'
# print(df_agro[cat_cols].describe().T)


plt.figure(figsize=(10, 14))
for i, col in enumerate(num_cols):
    plt.subplot(4, 2, i + 1)
    sns.histplot(df_agro, x=col, kde=True, color='red', alpha=0.2, bins=20)
    plt.xlabel(col)
    plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 10))
sns.heatmap(df_agro[num_cols + [label]].corr(), cmap='Blues', square=True, annot=True, linewidths=0.5)

plt.figure(figsize=(12, 7))
ax = sns.countplot(df_agro, y='District', palette='tab20', order=pd.Series(df_agro['District']).value_counts().index)
ax.bar_label(ax.containers[0])
plt.show()

plt.figure(figsize=(12, 4))
sns.boxplot(df_agro, y=label, x='District', palette='tab20')
plt.xticks(rotation=15)
plt.show()

plt.figure(figsize=(10, 15))
ax = sns.countplot(df_agro, y='Crop', palette='tab20', order=pd.Series(df_agro['Crop']).value_counts().index)
ax.bar_label(ax.containers[0])
plt.show()

plt.figure(figsize=(12, 4))
sns.boxplot(df_agro, y=label, x='Crop', palette='tab20')
plt.xticks(rotation=35)
plt.show()

plt.figure(figsize=(10, 4))
sns.boxplot(df_agro, y=label, x='Soil Type', palette='tab20')
plt.xticks(rotation=15)
plt.show()

plt.figure(figsize=(6, 4))
sns.boxplot(df_agro, y=label, x='Irrigation Method', palette='tab20')
plt.xticks(rotation=15)
plt.show()
