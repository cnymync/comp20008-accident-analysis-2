import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

# read in CSV files
person_data = pd.read_csv("person.csv")
flt_vehicle = pd.read_csv("filtered_vehicle.csv")
accident_data = pd.read_csv("accident.csv")

# merge data files
merged_df = accident_data.merge(person_data, on="ACCIDENT_NO", how="left")
merged_df = merged_df.merge(flt_vehicle, on="ACCIDENT_NO", how="left")

#group chosen columns from different csv files to chosen column names for later usage 
person_columns = ["ACCIDENT_NO", "SEX", "AGE_GROUP", "HELMET_BELT_WORN","ROAD_USER_TYPE_DESC","INJ_LEVEL"]
accident_columns = ["ACCIDENT_NO", "ACCIDENT_TIME", "DCA_DESC","ACCIDENT_TYPE_DESC", "ROAD_GEOMETRY_DESC", "SEVERITY", "SPEED_ZONE"]
vehicle_columns = ["ACCIDENT_NO","LEVEL_OF_DAMAGE","VEHICLE_MODEL","VEHICLE_TYPE_DESC","VEHICLE_MAKE","SEATING_CAPACITY", "VEHICLE_BODY_STYLE"]

# name the new data
new_person_data = person_data[person_columns]
new_vehicle_data = flt_vehicle[vehicle_columns]
new_accident_data = accident_data[accident_columns]

combined_1 = pd.merge(new_person_data, new_vehicle_data, on='ACCIDENT_NO', how='outer')
combined_data = pd.merge(combined_1, new_accident_data, on='ACCIDENT_NO', how='outer')

# group different agegroup to specific number for better data visualization  
age_group_to_label_num = {
    '0-4' : 0,       
    '5-12': 0,       #Child
    '13-15': 1,     
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

# replace unknown value in age column with the most occured age in the column
mode_age = age.mode()[0]
age = age.fillna(mode_age,inplace=True)

# combined the data and give out the final CSV
combined_data['AGE_GROUP'] = combined_data['AGE_GROUP'].astype(int)
combined_data.to_csv("finale.csv", index=False)

grouped = combined_data.groupby(['SEX', 'AGE_GROUP'])['INJ_LEVEL'].mean().unstack()

# Plot
plt.figure(figsize=(12, 8))
sns.heatmap(grouped, annot=True, cmap='YlOrRd')
plt.title("Injury Level by SEX and Age Group")
plt.ylabel("SEX")
plt.xlabel("Age group")
plt.tight_layout()
plt.show()