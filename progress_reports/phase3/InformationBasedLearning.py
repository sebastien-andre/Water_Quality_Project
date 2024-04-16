# Random Forest Classifier

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, precision_score, recall_score, f1_score, cohen_kappa_score, log_loss
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

# Creating the Random Forest model
random_forest_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)

# Training the Random Forest model
random_forest_model.fit(X_train, y_train)

# Predicting on the testing set
y_pred = random_forest_model.predict(X_test)

# Get probability estimates for the positive class
y_prob = random_forest_model.predict_proba(X_test)[:, 1]

# Visualizing distribution of target variables
class_distribution = df['Target'].value_counts()
print(class_distribution)

class_distribution.plot(kind='bar')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.title('Class Distribution')
plt.show()

# Create a confusion matrix to visualize results
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={"color": 'black'})
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Random Forest Classifier Confusion Matrix')
plt.show()

# Generate ROC curve data
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# Plot ROC curve
plt.figure()
auc = roc_auc_score(y_test, y_prob)
plt.plot(fpr, tpr, label=f'ROC Curve (area = {auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # Add a diagonal line for reference
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Random Forest Classifier')
plt.legend(loc="lower right")
plt.show()

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
kappa = cohen_kappa_score(y_test, y_pred)
logloss = log_loss(y_test, y_prob)

# Print metrics
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')
print(f'AUC: {auc:.2f}')
print(f'Cohen\'s Kappa: {kappa:.2f}')
print(f'Log Loss: {logloss:.2f}')