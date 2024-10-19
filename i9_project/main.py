import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, RocCurveDisplay
import numpy as np
import matplotlib.pyplot as plt

# Load the Cleveland dataset
url = "C:/Users/parth/Downloads/heart_disease/processed.cleveland.data"
column_names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
df = pd.read_csv(url, names=column_names)

# Replace '?' with NaN
df.replace('?', np.nan, inplace=True)

# Convert columns to numeric
df['ca'] = pd.to_numeric(df['ca'])
df['thal'] = pd.to_numeric(df['thal'])

# Fill missing values with the mean
df.fillna(df.mean(), inplace=True)

# Encode categorical variables
df = pd.get_dummies(df, drop_first=True)

# Normalize numerical features
numeric_features = ['age', 'trestbps', 'chol', 'thalach']
scaler = StandardScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# Split the data
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Filter the dataset to binary classification
df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

# Split, train, and evaluate with binary classification
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Feature importance for specific parameters
importance = model.coef_[0]
parameters_of_interest = ['age', 'trestbps', 'chol', 'thalach']
indices = [X.columns.get_loc(param) for param in parameters_of_interest]

# Plot feature importance for specific parameters
plt.bar([parameters_of_interest[i] for i in range(len(indices))], [importance[idx] for idx in indices])
plt.xlabel('Features')
plt.ylabel('Importance Score')
plt.title('Feature Importance for Selected Parameters')
plt.show()
