from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris.data[:,2:4], iris.target, test_size=0.2)

# Instantiate the SVM
svm = LinearSVC()

# Train the SVM
svm.fit(X_train[:,1:2], y_train)

# Make predictions
predicted = svm.predict(X_test[:,1:2])

# Plot the data points
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis', edgecolor='k')

# Create a meshgrid of points
x_min, x_max = X_test[:, 0].min() - 0.1, X_test[:, 0].max() + 0.1
y_min, y_max = X_test[:, 1].min() - 0.1, X_test[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
grid = np.c_[xx.ravel(), yy.ravel()]

# Make predictions on the grid points
grid_predicted = svm.predict(grid[:,1:2].reshape(-1, 1))

# Reshape the grid predictions
grid_predicted = grid_predicted.reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, grid_predicted, alpha=0.8, cmap='viridis')

# Set the labels and title
plt.xlabel('Variable 3')
plt.ylabel('Variable 4')
plt.title('Fronteras de decisión aprendidas')

# Show the plot
plt.show()

print("Accuracy: ", accuracy_score(predicted,y_test))

# Create a series of uniformly distributed points.
X = np.linspace(y_min, y_max, 4000).reshape(4000,1)

# Calculate when the prediction changes. That would be the cutpoint.
y_pred = svm.predict(X)
c = y_pred[0]
Cutpoints = []
Valores = [c]
for i, prediccion in enumerate(y_pred):
    if prediccion != c:
        c = prediccion
        Cutpoints.append(X[i].item())
        Valores.append(c)

print("Cutpoints: ", Cutpoints)

# Create a series of uniformly distributed points.
X = np.linspace(y_min, y_max, 4000).reshape(4000,1)

# Calculate the prediction of the SVM for each point.
y_pred = svm.predict(X)

# Create the plot.
plt.figure(figsize=(10, 6))
plt.plot(X, y_pred)
plt.xlabel('Variable utilizada para predecir')
plt.ylabel('Predicción de la SVM')
plt.title('Predicción de la SVM en función de la variable de entrada')
plt.show()
