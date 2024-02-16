import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset using pd.read_csv()
df = pd.read_csv("D:\\VIT\\SEM 4\\EDAI\\crop_yield.csv")

# Get unique crops in the dataset
unique_crops = df['Crop'].unique()

# Create an empty dictionary to store average yields for each crop
average_yields = {}

# Calculate average yield for each crop from 1997 to 2019
for crop in unique_crops:
    if crop != 'Coconut':  # Exclude Coconut
        crop_data = df[(df['Crop'] == crop) & (df['Crop_Year'] >= 1997) & (df['Crop_Year'] <= 2019)]
        average_yield = crop_data['Yield'].mean()
        average_yields[crop] = average_yield

# Plot average yields in a pie chart
plt.figure(figsize=(12, 10))
plt.pie(average_yields.values(), labels=average_yields.keys(), autopct='%1.1f%%', startangle=140, wedgeprops=dict(width=0.4))
plt.title('Average Yield of Crops (excluding Coconut) from 1997 to 2019')
plt.show()
