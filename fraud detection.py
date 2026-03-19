# Fraud Detection Project

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
df = pd.read_csv("transactions.csv")

df['Fraud']= df['Fraud'].map({0:'Normal', 1:'Fraud'})
print("First rows")
print(df.head())

print("\nDataset Shape")
print(df.shape)

print("\nDataset Info")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())

# Target distribution
print("\nFraud Distribution")
print(df['Fraud'].value_counts())

# Plot distribution
df['Fraud'].value_counts().plot(kind='bar')
plt.title("Fraud vs Normal Transactions")
plt.xlabel("Transaction Type")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.show()

# Features and target
X = df[['Amount','Transaction_Time']]
y = df['Fraud']

print("\nFeature Sample")
print(X.head())

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

print("\nTraining Data Size:", X_train.shape)
print("Testing Data Size:", X_test.shape)

# Train model
model = LogisticRegression()
model.fit(X_train,y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test,y_pred)
print("\nModel Accuracy:",accuracy)

# Confusion matrix
cm = confusion_matrix(y_test,y_pred)
print("\nConfusion Matrix")
print(cm)

# Scatter plot
colors={'Normal':'blue','Fraud':'red'}
plt.scatter(df['Amount'], df['Transaction_Time'], c=df['Fraud'].map(colors))
plt.xlabel("Amount")
plt.ylabel("Transaction Time")
plt.title("Fraud Transactions Visualization")
plt.show()

print("\nFraud Detection Completed")
