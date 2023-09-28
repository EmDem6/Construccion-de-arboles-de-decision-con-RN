import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Assuming you have your model ready
from DecisionTreeRN import TreeNetClassifier

dataset = load_iris()
X = dataset.data
y = dataset.target


num_features = X.shape[1]
num_classes = np.unique(y).shape[0]

#Entrenamiento por nivel
epoch = 30000

#Cuantos niveles maximos
depth = 12

#Condición de parada. Si la diferencia entre 2 iteraciones es menor a este valor
#deja de entrenar
stop_condition = 10**(-7)


k = 10  # Number of folds
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

models = []

kf = KFold(n_splits=k, shuffle=False)

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Initialize your model
    model = TreeNetClassifier.TreeNetClassifier(num_features=num_features,
                                                num_classes=num_classes,
                                                stop_condition=stop_condition,
                                                depth=depth,
                                                epoch=epoch)

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

    models.append(model)

avg_accuracy = sum(accuracy_scores) / k
avg_precision = sum(precision_scores) / k
avg_recall = sum(recall_scores) / k
avg_f1 = sum(f1_scores) / k


print("Average Accuracy:", avg_accuracy)
print("Average Precision:", avg_precision)
print("Average Recall:", avg_recall)
print("Average F1-score:", avg_f1)

