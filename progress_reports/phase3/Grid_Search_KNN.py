import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import f_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score, log_loss
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
# df = pd.read_csv("./preprocessed_dataset/preprocessed_dataset.csv.zip", compression='zip')

# # List of features
# features = df.columns.tolist()
# features.remove("Target")  # Remove the target variable from features

# # Categorical features
# categorical_features = ['Color', 'Source', 'Month', 'Day', 'Time of Day']
# num_features = [col for col in features if col not in categorical_features]

# # One-hot encoding for categorical features
# categorical_transformer = Pipeline(steps=[
#     ('encoder', OneHotEncoder(drop='first'))
# ])

# # Scaling for numerical features
# numerical_transformer = Pipeline(steps=[
#     ('scaler', StandardScaler())
# ])

# # Create a column transformer that applies the transformation to certain columns
# preprocessor = ColumnTransformer(transformers=[
#     ('cat', categorical_transformer, categorical_features),
#     ('num', numerical_transformer, num_features)],
#     remainder='passthrough'
# )

# # Drop the 'Day' and 'Time of Day' columns from the DataFrame
# df = df.drop(['Day', 'Time of Day'], axis=1)

# # Define features for different types of encoding
# ordinal_features = ['Color', 'Month']
# one_hot_features = ['Source']
# numerical_features = df.columns.difference(['Target'] + ordinal_features + one_hot_features).tolist()

# # Create transformers for each type of preprocessing
# ordinal_transformer = Pipeline(steps=[
#     ('ordinal', OrdinalEncoder())  # Default strategy is 'ordinal'
# ])

# categorical_transformer = Pipeline(steps=[
#     ('onehot', OneHotEncoder(drop='first'))
# ])

# numerical_transformer = Pipeline(steps=[
#     ('scaler', StandardScaler())
# ])

# # Create a column transformer that applies the transformation to certain columns
# preprocessor = ColumnTransformer(transformers=[
#     ('ordinal', ordinal_transformer, ordinal_features),
#     ('onehot', categorical_transformer, one_hot_features),
#     ('num', numerical_transformer, numerical_features)],
#     remainder='passthrough'
# )

# Load the dataset
df1 = pd.read_csv("./preprocessed_dataset/preprocessed_dataset.csv.zip", compression='zip')
df = df1.sample(frac=0.05)

# Drop the 'Day' and 'Time of Day' columns from the DataFrame
df = df.drop(['Day', 'Time of Day', 'Index'], axis=1)

# FEATURE SELECTION:
df = df.drop(['Zinc', 'Month', 'Air Temperature', 'Water Temperature', 'Source', 'Conductivity'], axis=1)
ordinal_features = ['Color']
numerical_features = df.columns.difference(['Target'] + ordinal_features).tolist()

# Create transformers for each type of preprocessing
ordinal_transformer = Pipeline(steps=[
    ('ordinal', OrdinalEncoder())  # Default strategy is 'ordinal'
])


numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(transformers=[
    ('ordinal', ordinal_transformer, ordinal_features),
    ('num', numerical_transformer, numerical_features)],
    remainder='passthrough'
)

# Create a pipeline that applies the preprocessor and then fits the KNN model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', KNeighborsClassifier())
])

# Define the hyperparameters grid
param_grid = {
    'classifier__n_neighbors': [5, 10, 15, 20, 25],
    'classifier__weights': ['uniform', 'distance'],  # Weighting of neighbors
    'classifier__p': [1, 2, 3]  # Range for power parameter for Minkowski distance
}


# Split the dataset into features and the target
X = df.drop('Target', axis=1)
y = df['Target']

# Split the dataset into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)  # 70% training, 30% temp
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)  # Split temp into 50% validation and 50% test

# Perform grid search with cross-validation on the validation set, optimizing for F1 score
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1', verbose=1, n_jobs=-1)

grid_search.fit(X_val, y_val)

# Get the best model and its hyperparameters
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# Predict on the validation set using the best model
y_val_pred = best_model.predict(X_val)
y_val_prob = best_model.predict_proba(X_val)[:, 1]

# Predict on the test set using the best model
y_test_pred = best_model.predict(X_test)
y_test_prob = best_model.predict_proba(X_test)[:, 1]

# Calculate evaluation metrics for the test set
accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)
kappa = cohen_kappa_score(y_test, y_test_pred)
logloss = log_loss(y_test, y_test_prob)
auc = roc_auc_score(y_test, y_test_prob)

# Print the best hyperparameters and evaluation metrics
print("Best Hyperparameters:", best_params)
print("Evaluation Metrics:")
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')
print(f'Cohen\'s Kappa: {kappa:.2f}')
print(f'Log Loss: {logloss:.2f}')
print(f'AUC: {auc:.2f}')

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
plt.subplot(2, 2, 4)  # Fourth plot in a 2x2 grid
fpr_test, tpr_test, _ = roc_curve(y_test, y_test_prob)
plt.plot(fpr_test, tpr_test, label=f'AUC (Test) = {roc_auc_score(y_test, y_test_prob):.2f}', color='blue')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve (Test)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")

plt.tight_layout()
plt.savefig('progress_reports/phase3/evaluations/Grid_Search_KNN.png', format='png', dpi=600)

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
metrics_df.to_csv('progress_reports/phase3/evaluations/Grid_Search_KNN.csv')

# Print metrics
print(metrics_df.round(2))


def get_feature_names(column_transformer):
    """Get feature names from a ColumnTransformer, including transformations and pipelines."""
    feature_names = []
    
    # Loop through each transformer in the column transformer
    for name, transformer, columns in column_transformer.transformers_:
        if transformer == 'passthrough':
            # Append the feature names directly
            feature_names.extend(columns)
        elif hasattr(transformer, 'get_feature_names_out'):
            # If a transformer has a 'get_feature_names_out', use it
            feature_names.extend(transformer.get_feature_names_out(columns))
        elif hasattr(transformer, 'named_steps'):
            last_step = transformer.named_steps[list(transformer.named_steps.keys())[-1]]
            if hasattr(last_step, 'get_feature_names_out'):
                feature_names.extend(last_step.get_feature_names_out(columns))
            elif hasattr(last_step, 'get_feature_names'):
                feature_names.extend(last_step.get_feature_names(columns))
            else:
                # If no method is available, use the original column names
                feature_names.extend(columns)
        else:
            # Just add the column names directly if no transformer method is applicable
            feature_names.extend(columns)

    return feature_names

import numpy as np

X_train_transformed = preprocessor.fit_transform(X_train)

# Now perform ANOVA F-test
f_values, p_values = f_classif(X_train_transformed, y_train)

# Get the feature names after transformation, just like before
transformed_feature_names = get_feature_names(preprocessor)

# Plotting the F-values
plt.figure(figsize=(12, 8))
ordering = np.argsort(f_values)[::-1]  # sort features by F-value
plt.title('ANOVA F-test Feature Importances for KNN')
plt.barh(np.array(transformed_feature_names)[ordering], f_values[ordering], color='blue')
plt.xlabel('Features')
plt.ylabel('F-value')
plt.tight_layout()
plt.show()
plt.savefig('progress_reports/phase3/evaluations/Feature_Selection_Grid_Search_KNN.png', format='png', dpi=600)


