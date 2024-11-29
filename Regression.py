import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
data = pd.read_csv("/Users/ritikagarwal/Downloads/raw_merged_heart_dataset.csv")

# Replace invalid entries ('?') with NaN
data.replace('?', np.nan, inplace=True)

# Convert numeric columns from object type to float
numeric_columns = ['trestbps', 'chol', 'fbs', 'restecg', 'thalachh', 'exang', 'slope', 'ca', 'thal']
for col in numeric_columns:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

# Fill NaN values with column medians
data.fillna(data.median(numeric_only=True), inplace=True)

# Display dataset information and summary
print("Dataset Info After Cleaning:")
print(data.info())

print("\nDataset Description After Cleaning:")
print(data.describe())

# Compute and display correlation matrix
print("\nCorrelation Matrix:")
print(data.corr().to_string())

# Convert categorical column 'Location' to dummy variables if present
if 'Location' in data.columns:
    data = pd.get_dummies(data, columns=['Location'], dtype='int64')

# Drop 'ID' column if it exists
if 'ID' in data.columns:
    data = data.drop(columns=['ID'])

# Define features (X) and target (Y)
if 'chol' in data.columns and 'target' in data.columns:
    X = data[['chol']]
    Y = data['target']
else:
    raise ValueError("Columns 'chol' or 'target' are missing in the dataset.")

print("\nFeature (X) Head:")
print(X.head())

print("\nTarget (Y) Head:")
print(Y.head())

# Scatter plot of chol vs target
plt.figure(figsize=(8, 6))
sns.scatterplot(x='chol', y='target', data=data)
plt.title('Scatter Plot of chol vs target')
plt.xlabel('chol (Cholesterol)')
plt.ylabel('target (Heart Disease Presence)')
plt.show()

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=42)

# Logistic Regression
logistic_model = LogisticRegression()
logistic_model.fit(X_train, Y_train)
Y_pred_logistic = logistic_model.predict(X_test)

# Calculate Logistic Regression Accuracy
logistic_accuracy = accuracy_score(Y_test, Y_pred_logistic)
print(f"\nLogistic Regression Accuracy: {logistic_accuracy:.2f}")

# Confusion Matrix and Classification Report
print("\nConfusion Matrix (Logistic Regression):")
print(confusion_matrix(Y_test, Y_pred_logistic))

print("\nClassification Report (Logistic Regression):")
print(classification_report(Y_test, Y_pred_logistic))

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, Y_train)

# Predict and calculate R² score
r2_score = linear_model.score(X_test, Y_test)
print(f"\nR² Score (Linear Regression): {r2_score:.4f}")

# Plot Linear Regression Line
plt.figure(figsize=(8, 6))
plt.scatter(X, Y, color='blue', label='Data points')
plt.plot(X_test, linear_model.predict(X_test), color='red', label='Regression Line')
plt.title('Regression Line for chol vs target')
plt.xlabel('chol (Cholesterol)')
plt.ylabel('target (Heart Disease Presence)')
plt.legend()
plt.show()
