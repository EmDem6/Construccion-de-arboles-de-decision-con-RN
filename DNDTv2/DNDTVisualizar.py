import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from DNDTv2 import DNDTModel
import numpy as np

# Load the dataset
dataset = load_iris()

# Select only two features
X = dataset.data[:, 2:4]
y = dataset.target

num_features = X.shape[1]
num_classes = np.unique(y).size

# Train the model
model = DNDTModel.DNDTClassifier(1, num_classes)
model.fit(X[:,0:1], y, 30000)

# Create a grid of points
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Predict the class for each point in the grid
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Create a scatter plot
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.8, cmap='viridis')
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap='viridis')
plt.title('DNDT Classifier - Iris Dataset')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Save the plot
plt.savefig('DNDT_Iris.png')

plt.show()

# Crear una serie de puntos uniformemente distribuidos.
X = np.linspace(x_min, x_max, 4000).reshape(4000,1)

# Calcular la predicción de la red neuronal para cada punto.
y_pred = model.predict(X)

# Crear el gráfico.
plt.figure(figsize=(10, 6))
plt.plot(X, y_pred)
plt.xlabel('Variable utilizada para predecir')
plt.ylabel('Predicción de la red neuronal')
plt.title('Predicción de la red neuronal en función de la variable de entrada')
plt.show()
