import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# read csv files
vehicle_data = pd.read_csv("vehicle.csv")
person_data = pd.read_csv("person.csv")
flt_vehicle = pd.read_csv("filtered_vehicle.csv")
accident_data = pd.read_csv("accident.csv")

# columns to keep
person_columns = ["ACCIDENT_NO", "SEX", "AGE_GROUP", "INJ_LEVEL", "ROAD_USER_TYPE_DESC"]
accident_columns = ["ACCIDENT_NO", "ACCIDENT_TIME", "ACCIDENT_TYPE", "ROAD_GEOMETRY_DESC", "SEVERITY", "SPEED_ZONE"]
vehicle_columns = ["ACCIDENT_NO", "ROAD_SURFACE_TYPE_DESC", "VEHICLE_BODY_STYLE"]

# take out columns to keep
new_person_data = person_data[person_columns]
new_vehicle_data = vehicle_data[vehicle_columns]
new_accident_data = accident_data[accident_columns]

combined_1 = pd.merge(new_person_data, new_vehicle_data, on='ACCIDENT_NO', how='outer')
combined_data = pd.merge(combined_1, new_accident_data, on='ACCIDENT_NO', how='outer')

# I acknowledge the use of ChatGPT [https://chat.openai.com/] to help explain how to 
# apply one hot encoding and remove duplicate rows in the data.

# I entered the following prompt: How to remove duplicates in data and apply one hot 
# encoding to a specific column in Python.

# I used the output to clarify how the data can be cleaned and apply one hot encoding 
# in Python.

# one hot encoding
combined_data = combined_data.drop_duplicates()
combined_data = pd.get_dummies(combined_data, columns=['SEX'])
combined_data['SEX_F'] = combined_data['SEX_F'].astype(int)
combined_data['SEX_M'] = combined_data['SEX_M'].astype(int)
combined_data['SEX_U'] = combined_data['SEX_U'].astype(int)

# group age
age_group_to_label_num = {
    '0-4' : 0,       #Child
    '5-12': 0,       #Child
    '13-15': 1,      # Teen
    '16-17': 1,      # Teen
    '18-21': 2,      # Young Adult
    '22-25': 2,
    '26-29': 3,      # Adult
    '30-39': 3,
    '40-49': 4,      # Middle Age
    '50-59': 4,
    '60-64': 5,      # Senior
    '65-69': 5,
    '70+': 5
}
combined_data['AGE_GROUP'] = combined_data['AGE_GROUP'].map(age_group_to_label_num)
age = combined_data['AGE_GROUP']
mode_age = age.mode()[0]
age = age.fillna(mode_age,inplace=True)
combined_data['AGE_GROUP'] = combined_data['AGE_GROUP'].astype(int)

# I acknowledge the use of ChatGPT [https://chat.openai.com/] to explain methods to 
# remove large numbers of unique values in a column in the data.

# I entered the following prompt: How can a large number of unique values in a column 
# be fitted into a heatmap in Python.

# I used the output to combine unique data values into 'other' while keeping the top 3
# for vehicle body style and road geometry and create a heatmap using the cleaned data 

combined_data_clean = pd.get_dummies(combined_data, columns = ["ROAD_USER_TYPE_DESC"])


top_vehicle_types = combined_data_clean['VEHICLE_BODY_STYLE'].value_counts().nlargest(3).index
combined_data_clean['VEHICLE_BODY_GROUP'] = combined_data_clean['VEHICLE_BODY_STYLE'].apply(
    lambda x: x if x in top_vehicle_types else 'Other'
)

top_road_types = combined_data_clean['ROAD_GEOMETRY_DESC'].value_counts().nlargest(3).index
combined_data_clean['ROAD_GEOMETRY_GROUP'] = combined_data_clean['ROAD_GEOMETRY_DESC'].apply(
    lambda x: x if x in top_road_types else 'Other'
)

grouped = combined_data_clean.groupby(['ROAD_GEOMETRY_GROUP', 'VEHICLE_BODY_GROUP'])['INJ_LEVEL'].mean().unstack()

# Plot
plt.figure(figsize=(12, 8))
sns.heatmap(grouped, annot=True, cmap='YlOrRd')
plt.title("Injury Level by Road Geometry and Vehicle Body Style")
plt.ylabel("Road Geometry")
plt.xlabel("Vehicle Body Style")
plt.tight_layout()
plt.show()