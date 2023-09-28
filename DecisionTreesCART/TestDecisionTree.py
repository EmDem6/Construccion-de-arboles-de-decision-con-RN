from sklearn.model_selection import train_test_split, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from sklearn.datasets import  load_breast_cancer
import time
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

# dataset = load_breast_cancer()
#
# X = dataset.data
# y = dataset.target

url = '../Datasets/CancerData/Cancer_Data.csv'
data_df = pd.read_csv(url, skiprows=1,header=None)
X = data_df.values[:,2:].astype('float') # Exclude the last column as the target variable
y = data_df.values[:,1]
label_encoder = LabelEncoder()
label_encoder.fit(y)
y = label_encoder.transform(y)


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#
# # Create an instance of DecisionTreeClassifier
# model = DecisionTreeClassifier(max_depth=3)
#
# t = time.time()
# model.fit(X_train, y_train)
# print("Tiempo: ", time.time() - t)
#
# # Make predictions on the test data
# y_pred = model.predict(X_test)
#
# # Calculate evaluation metrics
# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred, average='weighted')
# recall = recall_score(y_test, y_pred, average='weighted')
# f1 = f1_score(y_test, y_pred, average='weighted')
#
# print("Accuracy:", accuracy)
# print("Precision:", precision)
# print("Recall:", recall)
# print("F1-score:", f1)
#
# print(time.time() - t)


t = time.time()

k = 10  # Number of folds
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

models = []

kf = KFold(n_splits=k, shuffle=True)

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Initialize your model
    model = DecisionTreeClassifier(max_depth=12)

    # Fit the model on the training data
    model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Append the metrics to the respective lists
    accuracy_scores.append(accuracy)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)

    #We also store every model.
    models.append(model)

avg_accuracy = sum(accuracy_scores) / k
avg_precision = sum(precision_scores) / k
avg_recall = sum(recall_scores) / k
avg_f1 = sum(f1_scores) / k


print("Average Accuracy:", avg_accuracy)
print("Average Precision:", avg_precision)
print("Average Recall:", avg_recall)
print("Average F1-score:", avg_f1)

print(time.time() - t)

