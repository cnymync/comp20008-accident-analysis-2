import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif 
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold

df = pd.read_csv("finale.csv")

# Drop redundant columns and rows with missing values
ddf = df.drop(columns=['ACCIDENT_NO','VEHICLE_MODEL','VEHICLE_BODY_STYLE',
                       'VEHICLE_MAKE'])
dddf = ddf.dropna(axis=0)

# Encoding categorical columns using Label Encoder 
# Creating a copy to preserve original dataset 
dddf_copy = dddf.copy()

le = LabelEncoder()
dddf_copy['ACCIDENT_TYPE_DESC'] = le.fit_transform(dddf_copy['ACCIDENT_TYPE_DESC'])
dddf_copy['DCA_DESC'] = le.fit_transform(dddf_copy['DCA_DESC'])
dddf_copy['VEHICLE_TYPE_DESC'] = le.fit_transform(dddf_copy['VEHICLE_TYPE_DESC'])
dddf_copy['ROAD_USER_TYPE_DESC'] = le.fit_transform(dddf_copy['ROAD_USER_TYPE_DESC'])
dddf_copy['ROAD_GEOMETRY_DESC'] = le.fit_transform(dddf_copy['ROAD_GEOMETRY_DESC'])
dddf_copy['SEX'] = le.fit_transform(dddf_copy['SEX'])

# Function transforms time into integer form for KNN
def tag(time_remove):
    if ':' in time_remove:
        time = int(time_remove.replace(':',''))
    else:
        time = int(time_remove)
    return time

# Applys time transformation 
dddf_copy['ACCIDENT_TIME'] = dddf_copy['ACCIDENT_TIME'].apply(tag)

# Splitting data into features and target
X = dddf_copy.drop(["INJ_LEVEL"], axis=1)
y = dddf_copy["INJ_LEVEL"]

# Normalizing feature values 
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Computing Mutual Information again but with normalized data
mi_scores = mutual_info_classif(X_scaled_df, y)
mi_df = pd.DataFrame({"Feature": X_scaled_df.columns, "MI Score": mi_scores})
mi_df = mi_df.sort_values(by="MI Score", ascending=False)

# Identify top 15 most informative features 
top_features = mi_df["Feature"].head(15).tolist()
X_top = X[top_features]

# Initializing KNN classifier and using Stratified KFold cross-validation
# to score and predictions
knn = KNeighborsClassifier()
skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=88)
scores = cross_val_score(knn, X_top, y, cv=skf, scoring='accuracy')
y_pred = cross_val_predict(knn, X_top, y, cv=skf)

# Compute and plot confusion matrix
cm = confusion_matrix(y,y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Label Encoded KNN Confusion Matrix")
plt.xlabel('Predicted')
plt.ylabel('True')

# Save confusion matrix as a PNG file
#plt.savefig('/Users/user/Desktop/CMKNNOneHot.png', format='png')

# Generating classification report
report = classification_report(y, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df = report_df.round(2)

# Plotting classification report as a table
fig, ax = plt.subplots(figsize=(8, 4))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=report_df.values,
                colLabels=report_df.columns,
                rowLabels=report_df.index,
                loc='center',
                cellLoc='center',
                colLoc='center',
                bbox=[0, 0, 1, 1])

# Save the table as a PNG file
#plt.savefig('/Users/user/Desktop/classification_report_KNN_OneHot.png', format='png', bbox_inches='tight')

#I ackowledge the use of ChatGPT [https://chat.openai.com/] to help me explain and apply certain
#class and method documentations from pandas, sklearn, seaborn, and matplotlib
#I entered the following prompt: "What forms of cross validation is the most useful for normalized numeric datasets
#and how can it be used in conjunction with knn classifiers?"
#I used the output to identify possible types of cross validation that could be used and applied for knn classifiers