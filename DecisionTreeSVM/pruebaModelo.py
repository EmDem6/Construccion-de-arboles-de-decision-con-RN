# importar las bibliotecas necesarias
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from DecisionTreeSVM import TreeSVMClassifier

import time



# cargar el conjunto de datos iris
iris = load_iris()
# iris = load_wine()
# iris = load_breast_cancer()

n_features = 4
# n_features = 13
# n_features = 30

n_classes = 3
# n_classes = 2

depth = 3

# separar los datos en características y etiquetas
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = TreeSVMClassifier.TreeSVMClassifier(num_features=n_features,
                                                num_classes=n_classes,
                                                depth=depth)


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