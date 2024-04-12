import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('/Users/ehagye/Documents/Data_Science_Project/Updated_WQP/Water_Quality_Project/preprocessed_dataset/preprocessed_dataset.csv')

# Remove the 'Lead' column from the dataset
df.drop('Lead', axis=1, inplace=True)

# List of categorical features
categorical_features = ['Color', 'Source', 'Month', 'Day', 'Time of Day']

# Create a transformer for categorical features using Ordinal Encoder
categorical_transformer = Pipeline(steps=[
    ('encoder', OrdinalEncoder())
])

# Create a column transformer that applies the transformation to certain columns
preprocessor = ColumnTransformer(transformers=[
    ('cat', categorical_transformer, categorical_features)],
    remainder='passthrough'
)

# Create a pipeline that applies the preprocessor and then fits the logistic regression model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Split the dataset into features and the target
X = df.drop('Target', axis=1)
y = df['Target']

# Split the dataset into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the logistic regression model
model.fit(X_train, y_train)

# Predict on the testing set
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Logistic Regression Accuracy: {accuracy:.2f}')

# Create a confusion matrix to visualize results
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Logistic Regression Confusion Matrix')
plt.show()

# Show the ROC curve of the model
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC Curve (area = {roc_auc_score(y_test, y_prob):.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()
