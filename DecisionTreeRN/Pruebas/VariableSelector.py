import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.datasets import load_breast_cancer

from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_iris

import time

from sklearn.model_selection import KFold


import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from DecisionTreeRN import TreeNetClassifier

from sklearn.preprocessing import LabelEncoder

# Load the dataset
dataset = load_wine()

# Split the dataset into features (X) and labels (y)
X = dataset.data
y = dataset.target


url = '../../Datasets/CancerData/Cancer_Data.csv'
data_df = pd.read_csv(url, skiprows=1,header=None)
X = data_df.values[:,2:].astype('float') # Exclude the last column as the target variable
y = data_df.values[:,1]
label_encoder = LabelEncoder()
label_encoder.fit(y)
y = label_encoder.transform(y)


# url = '../../Datasets/LetterRecognition/letter-recognition.data'
#
# data_df = pd.read_csv(url, header=None)
#
# X = data_df.values[:,1:-1].astype('float')  # Exclude the last column as the target variable
# y = data_df.values[:,0]
# #
# from sklearn.preprocessing import LabelEncoder
# label_encoder = LabelEncoder()
#
# # Fit the LabelEncoder on the target column
# label_encoder.fit(y)
#
# y = label_encoder.transform(y)

# url = '../../Datasets/ImageSegmentation/segmentation.test'
#
# data_df = pd.read_csv(url, skiprows=4, header=None)
#
# X = data_df.values[:,1:-1].astype('float')  # Exclude the last column as the target variable
# y = data_df.values[:,0]
#
# from sklearn.preprocessing import LabelEncoder
# label_encoder = LabelEncoder()
#
# # Fit the LabelEncoder on the target column
# label_encoder.fit(y)
#
# # Transform the target column labels
# y = label_encoder.transform(y)

# url = '../../Datasets/GlassIdentification/glass.data'
#
# data_df = pd.read_csv(url, header=None)
#
# X = data_df.values[:, :-1]  # Exclude the last column as the target variable
# y = data_df.values[:, -1]
# #
# from sklearn.preprocessing import LabelEncoder
# label_encoder = LabelEncoder()
#
# # Fit the LabelEncoder on the target column
# label_encoder.fit(y)
#
# # Transform the target column labels
# y = label_encoder.transform(y)

# data_df = pd.read_csv('../../Datasets/abalone19/abalone19/abalone19.data', skiprows=1, header=None)
#
# X = data_df.values[:, :-1] # Exclude the last column as the target variable
# X[:,0] = np.where(X[:,0] == 'M', 0, 1)
# X = X.astype('float')
# y = data_df.values[:, -1]   # Use the last column as the target variable
# y = np.where(y == 'negative', 0, 1)

# data_df = pd.read_csv('../../Datasets/vehicle0/vehicle0/vehicle0.data', skiprows=1, header=None)
#
# X = data_df.values[:, :-1].astype('int')  # Exclude the last column as the target variable
# y = data_df.values[:, -1]   # Use the last column as the target variable
# y = np.where(y == 'negative', 0, 1)

# data_df = pd.read_csv('../../Datasets/Diabetes/diabetes.csv', skiprows=1, header=None)
#
# data_df = data_df.apply(pd.to_numeric, errors='coerce')
#
# X = data_df.values[:, :-1]  # Exclude the last column as the target variable
# y = data_df.values[:, -1]   # Use the last column as the target variable
# y.astype('int')



# # Read the .data file as a CSV
# data_df = pd.read_csv('../../Datasets/CarEvaluation/car.data', header=None)
#
# # Define the mapping for each categorical feature
# buying_mapping = {'v-high': 4, 'high': 3, 'med': 2, 'low': 1}
# maint_mapping = {'v-high': 4, 'high': 3, 'med': 2, 'low': 1}
# doors_mapping = {'2': 2, '3': 3, '4': 4, '5-more': 5}
# persons_mapping = {'2': 2, '4': 4, 'more': 6}
# lug_boot_mapping = {'small': 1, 'med': 2, 'big': 3}
# safety_mapping = {'low': 1, 'med': 2, 'high': 3}
# class_mapping = {'unacc': 0, 'acc': 1, 'good': 2, 'v-good': 3}
#
# # Apply the mappings to convert the categorical values to numerical labels
# data_df[0] = data_df[0].map(buying_mapping)
# data_df[1] = data_df[1].map(maint_mapping)
# data_df[2] = data_df[2].map(doors_mapping)
# data_df[3] = data_df[3].map(persons_mapping)
# data_df[4] = data_df[4].map(lug_boot_mapping)
# data_df[5] = data_df[5].map(safety_mapping)
# data_df[6] = data_df[6].map(class_mapping)
#
# data_df.dropna(inplace=True)
#
# # Split the dataset into features (X) and labels (y)
# X = data_df.values[:, :-1]
# y = data_df.values[:, -1].astype('int')


# data_df = pd.read_csv('../../Datasets/Diabetes/diabetes.csv', skiprows=1, header=None)
#
# data_df = data_df.apply(pd.to_numeric, errors='coerce')
#
# X = data_df.values[:, :-1]  # Exclude the last column as the target variable
# y = data_df.values[:, -1]   # Use the last column as the target variable
# y.astype('int')


# # Read the .data file to extract the actual data instances and labels
# data_df = pd.read_csv('../../Datasets/habermanSurvival/haberman.data', header=None)
#
# # Split the dataset into features (X) and labels (y)
# X = data_df.values[:,0:-1]
# y = data_df.values[:,-1]
#
# y = np.where(y == 1, 0, 1)

# Perform feature selection using mutual information
selector = SelectKBest(score_func=mutual_info_classif, k='all')  # Select top 10 features
X_selected = selector.fit_transform(X, y)

# Get the indices of the selected features
selected_indices = selector.get_support(indices=True)

# Get the names of the selected features
selected_features = X[selected_indices]

# Print the selected feature names
print("Selected Features:")
for feature in selected_features:
    print(feature)

X = X_selected

num_features = X.shape[1]
num_classes = np.unique(y).shape[0]

#Entrenamiento por nivel
epoch = 30000

#Cuantos niveles maximos
depth = 12

#Condición de parada. Si la diferencia entre 2 iteraciones es menor a este valor
#deja de entrenar
stop_condition = 10**(-7)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = TreeNetClassifier.TreeNetClassifier(num_features=num_features,
                                                num_classes=num_classes,
                                                stop_condition=stop_condition,
                                                depth=depth,
                                                epoch=epoch)


t = time.time()
model.fit(X_train, y_train)
print("Tiempo: ", time.time() - t)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

print(time.time() - t)




# t = time.time()
#
# k = 10  # Number of folds
# accuracy_scores = []
# precision_scores = []
# recall_scores = []
# f1_scores = []
#
# models = []
#
# kf = KFold(n_splits=k, shuffle=True)
#
# for train_index, test_index in kf.split(X):
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]
#
#     # Initialize your model
#     model = TreeNetClassifier.TreeNetClassifier(num_features=num_features,
#                                                 num_classes=num_classes,
#                                                 stop_condition=stop_condition,
#                                                 depth=depth,
#                                                 epoch=epoch)
#
#     # Fit the model on the training data
#     model.fit(X_train, y_train)
#
#     # Make predictions on the test data
#     y_pred = model.predict(X_test)
#
#     # Calculate evaluation metrics
#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred, average='weighted')
#     recall = recall_score(y_test, y_pred, average='weighted')
#     f1 = f1_score(y_test, y_pred, average='weighted')
#
#     # Append the metrics to the respective lists
#     accuracy_scores.append(accuracy)
#     precision_scores.append(precision)
#     recall_scores.append(recall)
#     f1_scores.append(f1)
#
#     #We also store every model.
#     models.append(model)
#
# avg_accuracy = sum(accuracy_scores) / k
# avg_precision = sum(precision_scores) / k
# avg_recall = sum(recall_scores) / k
# avg_f1 = sum(f1_scores) / k
#
#
# print("Average Accuracy:", avg_accuracy)
# print("Average Precision:", avg_precision)
# print("Average Recall:", avg_recall)
# print("Average F1-score:", avg_f1)
#
# print(time.time() - t)
#
# print(url)