# K Nearest Neighbors Model

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, cohen_kappa_score, log_loss

# Load the dataset
df = pd.read_csv("./preprocessed_dataset/preprocessed_dataset.csv.zip", compression='zip')

# List of categorical features
categorical_features = ['Color', 'Source', 'Month', 'Day', 'Time of Day']
num_features = [col for col in df.columns if col not in categorical_features + ['Target']]

# Create a transformer for categorical features using Ordinal Encoder
categorical_transformer = Pipeline(steps=[
    ('encoder', OrdinalEncoder())
])

# Create a transformer for numerical features
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Create a column transformer that applies the transformation to certain columns
preprocessor = ColumnTransformer(transformers=[
    ('cat', categorical_transformer, categorical_features),
    ('num', numerical_transformer, num_features)
], remainder='passthrough')

# Create a pipeline that applies the preprocessor and then fits the KNN model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', KNeighborsClassifier(n_neighbors=10, weights='distance'))
])

# Split the dataset into features and the target
X = df.drop('Target', axis=1)
y = df['Target']

# Split the dataset into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the KNN model
model.fit(X_train, y_train)

# Get probability estimates for the positive class
y_prob = model.predict_proba(X_test)[:, 1]

# Predict on the testing set
y_pred = model.predict(X_test)

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
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', annot_kws={"color": 'black'})
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('KNN Classifier Confusion Matrix')
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
plt.title('ROC Curve for KNN Classifier')
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