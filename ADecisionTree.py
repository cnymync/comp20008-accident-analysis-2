import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Load the data
df = pd.read_csv('finale.csv')  


# Step 2: Define the columns I'm using
columns = ['AGE_GROUP', 'HELMET_BELT_WORN', 'LEVEL_OF_DAMAGE', 'SEATING_CAPACITY', 'ACCIDENT_TIME',
           'SEVERITY', 'SPEED_ZONE', 'SEX',
           'ROAD_GEOMETRY_DESC',
           'ROAD_USER_TYPE_DESC',
           'ACCIDENT_TYPE_DESC','SEVERITY','INJ_LEVEL']

df = df[columns].dropna(axis=0)

# Function transforms time into integer form for KNN
def tag(time_remove):
    if ':' in time_remove:
        time = int(time_remove.replace(':',''))
    else:
        time = int(time_remove)
    return time

# Applys time transformation 
df['ACCIDENT_TIME'] = df['ACCIDENT_TIME'].apply(tag)


le = LabelEncoder()
df['SEX'] = le.fit_transform(dddf_copy['SEX'])
df['ROAD_GEOMETRY_DESC'] = le.fit_transform(dddf_copy['ROAD_GEOMETRY_DESC'])
df['ROAD_USER_TYPE_DESC'] = le.fit_transform(dddf_copy['ROAD_USER_TYPE_DESC'])
df['ACCIDENT_TYPE_DESC'] = le.fit_transform(dddf_copy['ACCIDENT_TYPE_DESC'])

# Step 3: Split into features and target
X = df.drop('INJ_LEVEL', axis=1)
y = df['INJ_LEVEL']

# Step 4: Initialize K-Fold
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=88)

# To store scores and predictions
fold = 1
accuracies = []
all_y_true = []
all_y_pred = []

for train_index, test_index in kf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model = DecisionTreeClassifier(random_state=88)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    print(f"Fold {fold} Accuracy: {acc:.4f}")
    fold += 1

    # Collect all predictions for confusion matrix
    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred)

# Step 5: Average Accuracy
print(f"\nAverage Accuracy across folds: {np.mean(accuracies):.4f}")

# Step 6: Confusion Matrix
cm = confusion_matrix(all_y_true, all_y_pred, labels=np.unique(y))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
disp.plot(xticks_rotation='vertical', cmap='Blues')
plt.title("Confusion Matrix (All Folds Combined)")
plt.tight_layout()
plt.show()

#creating my accuracy table
from sklearn.metrics import classification_report

report = classification_report(all_y_true, all_y_pred, output_dict=True)

# Convert the report to a pandas DataFrame
report_df = pd.DataFrame(report).transpose()
report_df = report_df.round(2)
# Plot the table
fig, ax = plt.subplots(figsize=(8, 4))  # You can adjust the size
ax.axis('tight')
ax.axis('off')

# Create the table
table = ax.table(cellText=report_df.values,
                colLabels=report_df.columns,
                rowLabels=report_df.index,
                loc='center',
                cellLoc='center',
                colLoc='center',
                bbox=[0, 0, 1, 1])