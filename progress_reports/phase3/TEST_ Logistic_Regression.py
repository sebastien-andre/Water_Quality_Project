import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score, log_loss
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("./preprocessed_dataset/preprocessed_dataset.csv.zip", compression='zip')

# List of categorical features
categorical_features = ['Color', 'Source', 'Month', 'Day', 'Time of Day']
numerical_features = df.columns.difference(['Target'] + categorical_features).tolist()

# Create a transformer for categorical features using OneHotEncoder
categorical_transformer = Pipeline(steps=[
    ('encoder', OneHotEncoder(drop='first'))
])

# Create a transformer for scaling numerical features
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Create a column transformer that applies the transformation to certain columns
preprocessor = ColumnTransformer(transformers=[
    ('cat', categorical_transformer, categorical_features),
    ('num', numerical_transformer, numerical_features)],
    remainder='passthrough'
)

# Create a pipeline that applies the preprocessor and then fits the logistic regression model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=5000, class_weight='balanced'))
])

# Split the dataset into features and the target
X = df.drop('Target', axis=1)
y = df['Target']

# Split the dataset into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)  # 70% training, 30% temp
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)  # Split temp into 50% validation and 50% test

# Train the logistic regression model
model.fit(X_train, y_train)

# Validate and Test the model
y_val_pred = model.predict(X_val)
y_val_prob = model.predict_proba(X_val)[:, 1]
y_test_pred = model.predict(X_test)
y_test_prob = model.predict_proba(X_test)[:, 1]

# Plotting all figures in a 2x2 grid
plt.figure(figsize=(12, 10))

# Validation Confusion Matrix
plt.subplot(2, 2, 1)  # First plot in a 2x2 grid
cm_val = confusion_matrix(y_val, y_val_pred)
sns.heatmap(cm_val, annot=True, fmt='d', cmap='Greens')
plt.title('Validation Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# Test Confusion Matrix
plt.subplot(2, 2, 2)  # Second plot in a 2x2 grid
cm_test = confusion_matrix(y_test, y_test_pred)
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues')
plt.title('Test Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# Validation ROC Curve
plt.subplot(2, 2, 3)  # Third plot in a 2x2 grid
fpr_val, tpr_val, _ = roc_curve(y_val, y_val_prob)
plt.plot(fpr_val, tpr_val, label=f'AUC (Validation) = {roc_auc_score(y_val, y_val_prob):.2f}', color='green')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve (Validation)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")

# Test ROC Curve
auc_test = roc_auc_score(y_test, y_test_prob)
plt.subplot(2, 2, 4)  # Fourth plot in a 2x2 grid
fpr_test, tpr_test, _ = roc_curve(y_test, y_test_prob)
plt.plot(fpr_test, tpr_test, label=f'AUC (Test) = {auc_test:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve (Test)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")

plt.savefig('progress_reports/phase3/evaluations/Logistic_Regression.png', format='png', dpi=600)

plt.tight_layout()
plt.show()


# # Print metrics for validation and test sets
# print("Validation Metrics:")
# print(f'Accuracy: {accuracy_score(y_val, y_val_pred):.2f}')
# print(f'Precision: {precision_score(y_val, y_val_pred):.2f}')
# print(f'Recall: {recall_score(y_val, y_val_pred):.2f}')
# print(f'F1 Score: {f1_score(y_val, y_val_pred):.2f}')
# print(f'AUC: {roc_auc_score(y_val, y_val_prob):.2f}')
# print(f'Cohen\'s Kappa: {cohen_kappa_score(y_val, y_val_pred):.2f}')
# print(f'Log Loss: {log_loss(y_val, y_val_prob):.2f}')

# print("\n\nTest Metrics:")
# print(f'Accuracy: {accuracy_score(y_test, y_test_pred):.2f}')
# print(f'Precision: {precision_score(y_test, y_test_pred):.2f}')
# print(f'Recall: {recall_score(y_test, y_test_pred):.2f}')
# print(f'F1 Score: {f1_score(y_test, y_test_pred):.2f}')
# print(f'AUC: {roc_auc_score(y_test, y_test_prob):.2f}')
# print(f'Cohen\'s Kappa: {cohen_kappa_score(y_test, y_test_pred):.2f}')
# print(f'Log Loss: {log_loss(y_test, y_test_prob):.2f}')


# Calculate metrics for validation set
validation_metrics = {
    "Accuracy": accuracy_score(y_val, y_val_pred),
    "Precision": precision_score(y_val, y_val_pred),
    "Recall": recall_score(y_val, y_val_pred),
    "F1 Score": f1_score(y_val, y_val_pred),
    "AUC": roc_auc_score(y_val, y_val_prob),
    "Cohen's Kappa": cohen_kappa_score(y_val, y_val_pred),
    "Log Loss": log_loss(y_val, y_val_prob)
}

# Calculate metrics for test set
test_metrics = {
    "Accuracy": accuracy_score(y_test, y_test_pred),
    "Precision": precision_score(y_test, y_test_pred),
    "Recall": recall_score(y_test, y_test_pred),
    "F1 Score": f1_score(y_test, y_test_pred),
    "AUC": roc_auc_score(y_test, y_test_prob),
    "Cohen's Kappa": cohen_kappa_score(y_test, y_test_pred),
    "Log Loss": log_loss(y_test, y_test_prob)
}

# Convert metrics to DataFrame
metrics_df = pd.DataFrame([validation_metrics, test_metrics], index=['Validation', 'Test'])

# Save metrics to CSV
metrics_df.to_csv('progress_reports/phase3/evaluations/Logistic_Regression.csv')

# Print metrics
print(metrics_df.round(2))