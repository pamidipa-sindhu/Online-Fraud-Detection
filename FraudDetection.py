#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Step 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Step 2: Load the Dataset
import pandas as pd

# Load the dataset from Google Drive
data = pd.read_csv('/content/drive/MyDrive/Onlinefraud.csv')

# Display the first few rows of the dataset to ensure it loaded correctly
print(data.head())


# 

# In[ ]:


# Step 1: Mount Google Drive and Load Dataset
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd

# Load the dataset
#Kaggle : https://www.kaggle.com/datasets/jainilcoder/online-payment-fraud-detection/data
data = pd.read_csv('/content/drive/MyDrive/Onlinefraud.csv')
print(data.head())

# Step 2: Inspect the Dataset
# Check the shape of the dataset
print("Dataset shape:", data.shape)

# Get an overview of the dataset
print(data.info())

# Check for any missing values
print("Missing values in each column:\n", data.isnull().sum())

# Statistical summary of the dataset
print(data.describe())

# Step 3: Data Cleansing and Processing
# Perform one-hot encoding on the 'type' column
data_encoded = pd.get_dummies(data, columns=['type'], drop_first=True)

# Drop the columns with string values that cannot be converted to numeric
data_encoded = data_encoded.drop(columns=['nameOrig', 'nameDest'])

# Scale the numerical features if needed
from sklearn.preprocessing import StandardScaler

# Select numerical columns for scaling
numerical_features = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']

# Initialize the scaler
scaler = StandardScaler()

# Fit and transform the numerical features
data_encoded[numerical_features] = scaler.fit_transform(data_encoded[numerical_features])

print(data_encoded.head())  # Check the scaled features

# Step 4: Splitting the Dataset
from sklearn.model_selection import train_test_split

# Define features and target
X = data_encoded.drop('isFraud', axis=1)
y = data_encoded['isFraud']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the shapes of the resulting datasets
print("Training set shape:", X_train.shape, y_train.shape)
print("Testing set shape:", X_test.shape, y_test.shape)

# Step 5: Modeling with Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

# Initialize the Logistic Regression model
lr_model = LogisticRegression(solver='liblinear', random_state=42)

# Fit the model on the training data
lr_model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred_lr = lr_model.predict(X_test)

# Evaluate the model's performance
print("Logistic Regression Accuracy: ", accuracy_score(y_test, y_pred_lr))
print("Classification Report:\n", classification_report(y_test, y_pred_lr))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))
print("ROC AUC Score: ", roc_auc_score(y_test, y_pred_lr))

# Step 6: Modeling with Random Forest
from sklearn.ensemble import RandomForestClassifier

# Initialize the Random Forest model
rf_model = RandomForestClassifier(random_state=42)

# Fit the model on the training data
rf_model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred_rf = rf_model.predict(X_test)

# Evaluate the model's performance
print("Random Forest Accuracy: ", accuracy_score(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("ROC AUC Score: ", roc_auc_score(y_test, y_pred_rf))

# Step 7: Visualizing the Results
import seaborn as sns
import matplotlib.pyplot as plt

# Create confusion matrices for both models
matrix_lr = confusion_matrix(y_test, y_pred_lr)
matrix_rf = confusion_matrix(y_test, y_pred_rf)

# Create a figure with two subplots side by side
fig, ax = plt.subplots(1, 2, figsize=(16, 6))

# Plot Logistic Regression Confusion Matrix
sns.heatmap(matrix_lr, annot=True, fmt='d', cmap='coolwarm', cbar=False,
            xticklabels=['Actual Negative', 'Actual Positive'],
            yticklabels=['Predict Negative', 'Predict Positive'], ax=ax[0])
ax[0].set_xlabel('Actual')
ax[0].set_ylabel('Predicted')
ax[0].set_title('Confusion Matrix - Logistic Regression')

# Plot Random Forest Confusion Matrix
sns.heatmap(matrix_rf, annot=True, fmt='d', cmap='magma', cbar=False,
            xticklabels=['Actual Negative', 'Actual Positive'],
            yticklabels=['Predict Negative', 'Predict Positive'], ax=ax[1])
ax[1].set_xlabel('Actual')
ax[1].set_ylabel('Predicted')
ax[1].set_title('Confusion Matrix - Random Forest')

# Show the plots
plt.tight_layout()
plt.show()


# For logistic regression

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Assuming you have your data in X and y
# X: Features
# y: Labels (0 for non-fraud, 1 for fraud)

# Step 1: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 2: Train the Logistic Regression model
log_reg_model = LogisticRegression()
log_reg_model.fit(X_train, y_train)

# Step 3: Generate predicted probabilities
y_pred_proba = log_reg_model.predict_proba(X_test)[:, 1]  # Get the probabilities for the positive class (fraud)

# Calculate the ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Plotting the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


# For Random forest

# In[ ]:


# Assuming you have already trained your random forest model as `rf_model`
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]  # Get the probabilities for the positive class (fraud)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='green', lw=2, label=f'ROC Curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

