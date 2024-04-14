import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Reads preprocessed dataset from csv
df = pd.read_csv("./preprocessed_dataset/preprocessed_dataset.csv.zip", compression="zip")

# List of features
features = df.columns.tolist()
features.remove("Target")  # Remove the target variable from features

# Categorical features
categorical_features = ['Color', 'Source', 'Month', 'Day', 'Time of Day']
num_features = [col for col in features if col not in categorical_features]

# Splitting the dataset into features and target variable
X = df[features]
y = df["Target"]

# One-hot encoding for categorical features
encoder = OneHotEncoder(drop='first')
X_encoded = encoder.fit_transform(X[categorical_features])

# Getting the column names for the one-hot encoded features
encoded_columns = []
for i, col in enumerate(categorical_features):
    unique_values = df[col].unique()[1:]  # Remove the first category
    for value in unique_values:
        encoded_columns.append(f"{col}_{value}")

# Creating a DataFrame with the one-hot encoded features
X_encoded_df = pd.DataFrame(X_encoded.toarray(), columns=encoded_columns)

# Concatenating encoded categorical features with numerical features
X_processed = pd.concat([X_encoded_df, X[num_features]], axis=1)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Creating the AdaBoost model with default base estimator (DecisionTreeClassifier)
adaboost_model = AdaBoostClassifier(n_estimators=50, learning_rate=1)

# Training the AdaBoost model
adaboost_model.fit(X_train, y_train)

# Predicting on the testing set
y_pred = adaboost_model.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))
