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
import torch.nn.functional as F

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from DecisionTreeRN import TreeNetClassifier

from sklearn.preprocessing import LabelEncoder

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):

        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x

datasets = []



data_df = pd.read_csv('../Datasets/Diabetes/diabetes.csv', skiprows=1, header=None)

data_df = data_df.apply(pd.to_numeric, errors='coerce')

X = data_df.values[:, :-1]  # Exclude the last column as the target variable
y = data_df.values[:, -1]   # Use the last column as the target variable
y.astype('int')

datasets.append({'X': X, 'y':y, 'name': 'Diabetes'})


# Read the .data file to extract the actual data instances and labels
data_df = pd.read_csv('../Datasets/habermanSurvival/haberman.data', header=None)

# Split the dataset into features (X) and labels (y)
X = data_df.values[:,0:-1]
y = data_df.values[:,-1]

y = np.where(y == 1, 0, 1)

datasets.append({'X': X, 'y':y, 'name': 'Haberman Survival'})

# # Read the .data file as a CSV
# data_df = pd.read_csv('../Datasets/CarEvaluation/car.data', header=None)
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
#
# datasets.append({'X': X, 'y':y, 'name': 'CarEvaluation'})
#
# data_df = pd.read_csv('../Datasets/vehicle0/vehicle0/vehicle0.data', skiprows=1, header=None)
#
# X = data_df.values[:, :-1].astype('int')  # Exclude the last column as the target variable
# y = data_df.values[:, -1]   # Use the last column as the target variable
# y = np.where(y == 'negative', 0, 1)
#
# datasets.append({'X': X, 'y':y, 'name': 'vehicle'})
#
# url = '../Datasets/GlassIdentification/glass.data'
#
# data_df = pd.read_csv(url, header=None)
#
# X = data_df.values[:, :-1]  # Exclude the last column as the target variable
# y = data_df.values[:, -1]
#
# from sklearn.preprocessing import LabelEncoder
# label_encoder = LabelEncoder()
#
# # Fit the LabelEncoder on the target column
# label_encoder.fit(y)
#
# # Transform the target column labels
# y = label_encoder.transform(y)
#
# datasets.append({'X': X, 'y':y, 'name': 'glass'})
#
# url = '../Datasets/ImageSegmentation/segmentation.test'
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
#
# datasets.append({'X': X, 'y':y, 'name': 'Image Segmentation'})
#
# url = '../Datasets/CancerData/Cancer_Data.csv'
# data_df = pd.read_csv(url, skiprows=1,header=None)
# X = data_df.values[:,2:].astype('float') # Exclude the last column as the target variable
# y = data_df.values[:,1]
# label_encoder = LabelEncoder()
# label_encoder.fit(y)
# y = label_encoder.transform(y)
#
# datasets.append({'X': X, 'y':y, 'name': 'Cancer Data'})





class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):

        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x




for d in datasets:

    print('DATASET: ', d['name'])

    X = d['X']
    y = d['y']

    # Set hyperparameters
    input_size = X.shape[1]
    hidden_size = 64
    num_classes = np.unique(y).size
    num_epochs = 100000
    learning_rate = 0.01
    k = 10  # Number of folds


    # Initialize lists to store evaluation metrics for each fold
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []



    # Perform k-fold cross-validation
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    for fold, (train_indices, test_indices) in enumerate(kf.split(X)):
        print(f"Fold {fold + 1}/{k}")

        t = time.time()

        # Split the data into training and testing sets for this fold
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        # Convert data to PyTorch tensors
        X_train = torch.Tensor(X_train)
        y_train = torch.LongTensor(y_train)
        X_test = torch.Tensor(X_test)
        y_test = torch.LongTensor(y_test)

        # Initialize the model
        model = Net(input_size, hidden_size, num_classes)

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

        # Training loop
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()

            # Forward pass
            outputs = model(X_train)
            loss = criterion(outputs, y_train)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        # Evaluation
        model.eval()
        with torch.no_grad():
            # Calculate predictions
            y_pred = model(X_test).argmax(dim=1)

            # Calculate evaluation metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')

            # Store evaluation metrics for this fold
            accuracy_scores.append(accuracy)
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)

            # Print the average evaluation metrics
            print("Accuracy:", accuracy)
            print("Precision:", precision)
            print("Recall:", recall)
            print("F1 Score:", f1)



        print(time.time() - t)

    # Calculate average evaluation metrics across all folds
    avg_accuracy = np.mean(accuracy_scores)
    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)
    avg_f1 = np.mean(f1_scores)

    # Print the average evaluation metrics
    print("Average Accuracy:", avg_accuracy)
    print("Average Precision:", avg_precision)
    print("Average Recall:", avg_recall)
    print("Average F1 Score:", avg_f1)