import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# Reads preprocessed dataset from csv
df = pd.read_csv("./preprocessed_dataset/preprocessed_dataset.csv.zip", compression="zip")

