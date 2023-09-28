from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from DecisionTreeSVM import TreeSVMClassifier  # Import the TreeSVMClassifier

import matplotlib.pyplot as plt

# Load the dataset
dataset = load_iris()

# Split the dataset into features (X) and labels (y)
X = dataset.data[:,2:4]
y = dataset.target

num_features = X.shape[1]
num_classes = np.unique(y).shape[0]

# Maximum depth of the tree
depth = 12

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = TreeSVMClassifier.TreeSVMClassifier(num_features=num_features,
                                            num_classes=num_classes,
                                            depth=depth)

model.fit(X_train, y_train)

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

# Plot the data points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k')

# Create a meshgrid of points
x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
grid = np.c_[xx.ravel(), yy.ravel()]



# Reshape the grid predictions
grid_predicted = model.predict(grid)
grid_predicted = np.array(grid_predicted).reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, grid_predicted, alpha=0.8, cmap='viridis')

# Set the labels and title
plt.xlabel('Variable 3')
plt.ylabel('Variable 4')
plt.title('Fronteras de decisión aprendidas')

# Show the plot
plt.show()
