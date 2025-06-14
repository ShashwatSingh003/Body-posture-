import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ✅ Step 1: Load Features
features = []
with open('UCI HAR Dataset/features.txt') as f:
    features = [line.split()[1] for line in f.readlines()]

print('No of Features: {}'.format(len(features)))

# ✅ Step 2: Make feature names unique
seen = set()
uniq_features = []
for x in features:
    if x not in seen:
        uniq_features.append(x)
        seen.add(x)
    elif x + 'n' not in seen:
        uniq_features.append(x + 'n')
        seen.add(x + 'n')
    else:
        uniq_features.append(x + 'nn')
        seen.add(x + 'nn')

# ✅ Step 3: Load Training Data
X_train = pd.read_csv(
    'UCI HAR Dataset/train/X_train.txt',
    sep='\s+',
    header=None,
    names=uniq_features
)

subject_train = pd.read_csv(
    'UCI HAR Dataset/train/subject_train.txt',
    header=None
)[0]

y_train = pd.read_csv(
    'UCI HAR Dataset/train/y_train.txt',
    header=None
)[0]

activity_labels = {
    1: 'WALKING',
    2: 'WALKING_UPSTAIRS',
    3: 'WALKING_DOWNSTAIRS',
    4: 'SITTING',
    5: 'STANDING',
    6: 'LAYING'
}
y_train_labels = y_train.map(activity_labels)

X_train['subject'] = subject_train
X_train['Activity'] = y_train
X_train['ActivityName'] = y_train_labels
train = X_train

# ✅ Step 4: Load Testing Data
X_test = pd.read_csv(
    'UCI HAR Dataset/test/X_test.txt',
    sep='\s+',
    header=None,
    names=uniq_features
)

subject_test = pd.read_csv(
    'UCI HAR Dataset/test/subject_test.txt',
    header=None
)[0]

y_test = pd.read_csv(
    'UCI HAR Dataset/test/y_test.txt',
    header=None
)[0]

y_test_labels = y_test.map(activity_labels)

X_test['subject'] = subject_test
X_test['Activity'] = y_test
X_test['ActivityName'] = y_test_labels
test = X_test

# ✅ Step 5: Display Samples & Shape
print(train.sample())
print(test.sample())
print("Train shape:", train.shape)
print("Test shape :", test.shape)

# ✅ Step 6: Check for duplicates and nulls
print('No of duplicates in train: {}'.format(sum(train.duplicated())))
print('No of duplicates in test : {}'.format(sum(test.duplicated())))

print('We have {} NaN/Null values in train'.format(train.isnull().values.sum()))
print('We have {} NaN/Null values in test'.format(test.isnull().values.sum()))

# ✅ Step 7: Visualization
sns.set_style('whitegrid')
plt.rcParams['font.family'] = 'Dejavu Sans'

plt.figure(figsize=(16,8))
plt.title('Data provided by each user', fontsize=20)
sns.countplot(x='subject', hue='ActivityName', data=train)
plt.show()

plt.title('No of Datapoints per Activity', fontsize=15)
sns.countplot(y=train.ActivityName)
plt.xticks(rotation=90)
plt.show()

# ✅ Step 8: Clean Column Names
columns = train.columns
columns = columns.str.replace(r'[()]', '', regex=True)
columns = columns.str.replace(r'[-]', '', regex=True)
columns = columns.str.replace(r'[,]', '', regex=True)

train.columns = columns
test.columns = columns

# ✅ Step 9: Save Preprocessed Data
os.makedirs('data', exist_ok=True)
train.to_csv('data/train.csv', index=False)
test.to_csv('data/test.csv', index=False)

print("✅ Data saved to 'data/train.csv' and 'data/test.csv'")