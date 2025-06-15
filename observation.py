import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/train.csv')

corr_matrix = df.drop(columns=['Activity', 'subject'], errors='ignore').select_dtypes(include=[np.number]).corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr_matrix, cmap='coolwarm', annot=False)
plt.show()

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

threshold = 0.9
to_drop = [column for column in upper.columns if any(upper[column].abs() > threshold)]
df_reduced = df.drop(columns=to_drop)
df_reduced = df_reduced.drop(columns=['Activity', 'subject'], errors='ignore')

print(f"Dropped {len(to_drop)} features: {to_drop}")

plt.figure(figsize=(12,10))
sns.heatmap(df_reduced.select_dtypes(include=np.number), cmap='coolwarm', annot=False)
plt.show()

df_reduced.shape[1]

activity_dummies = pd.get_dummies(df_reduced['ActivityName'])
print("Available activity columns:", activity_dummies.columns.tolist())
correlations = {}
for activity in activity_dummies.columns:
  correlations[activity] = df_reduced.drop('ActivityName', axis=1).corrwith(activity_dummies[activity])
  top_corr = correlations[activity].abs().sort_values(ascending=False).head(20)
  plt.figure(figsize=(10, 4))
  top_corr.plot(kind='bar')
  plt.title(f"Top 20 Features Correlated with {activity}")
  plt.xticks(rotation=90)
  plt.tight_layout()
  plt.show()

activity_means = df_reduced.groupby('ActivityName').mean()
activity_variances = activity_means.var().sort_values(ascending=False)

print("Top 30 features to distinguish activites are:")

print(activity_variances.head(30)) 