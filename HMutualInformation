import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

# read in csv files 
accident_df = pd.read_csv("accident.csv")
person_df = pd.read_csv("person.csv")
filtered_vehicle_df = pd.read_csv("filtered_vehicle.csv")

# chose columns as the feature to be compared later 
features = ["ACCIDENT_TIME", "ACCIDENT_TYPE_DESC", "DCA_DESC", "ROAD_GEOMETRY_DESC", "SEVERITY","SPEED_ZONE", "VEHICLE_YEAR_MANUF","ROAD_SURFACE_TYPE_DESC","VEHICLE_BODY_STYLE",
             "VEHICLE_MAKE","VEHICLE_MODEL","VEHICLE_TYPE_DESC","NO_OF_CYLINDERS","SEATING_CAPACITY","FINAL_DIRECTION","SEX","AGE_GROUP","HELMET_BELT_WORN","ROAD_USER_TYPE_DESC",
           "LEVEL_OF_DAMAGE","CAUGHT_FIRE","DAY_WEEK_DESC"]

# compare features with target
target = "INJ_LEVEL_DESC"


# I acknowledge the use of ChatGPT [https://chat.openai.com/] to help explain how to 
# apply one hot encoding and remove duplicate rows in the data.

# I entered the following prompt: How to generate a mutual information table with two chosen array

# I used the output to clarify how the MI table can be constructed in python

df = merged_df[features + [target]].dropna()

encoders = {}
for col in features + [target]:
    if df[col].dtype == 'object' or df[col].dtype.name == 'category':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

X = df[features]
y = df[target]
mi_scores = mutual_info_classif(X, y, discrete_features=True)

mi_df = pd.DataFrame({
    "Feature": features,
    "Mutual Information": mi_scores
}).sort_values(by="Mutual Information", ascending=False)

print(mi_df.head(16))
