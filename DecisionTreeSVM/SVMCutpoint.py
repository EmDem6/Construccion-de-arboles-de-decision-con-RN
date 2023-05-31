from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np

import matplotlib.pyplot as plt

# cargar el conjunto de datos iris
iris = load_iris()
# iris = load_wine()
# iris = load_breast_cancer()

n_features = 1
# n_features = 13
# n_features = 30

n_classes = 3
# n_classes = 2


# separar los datos en características y etiquetas
X = iris.data
y = iris.target

# dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# crear un modelo SVM con kernel lineal
model = SVC(kernel='linear')

# ajustar el modelo usando los datos de entrenamiento
model.fit(X_train[:,2:3], y_train)

# hacer predicciones en los datos de prueba
y_pred = model.predict(X_test[:,2:3])

# evaluar la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print("Precisión del modelo SVM: {:.2f}%".format(accuracy*100))

# Plot the data points
plt.scatter(X_test[:, 2], X_test[:, 3], c=y_test, cmap='viridis')

# Create a meshgrid of points
x_min, x_max = X_test[:, 2].min() - 0.1, X_test[:, 2].max() + 0.1
y_min, y_max = X_test[:, 3].min() - 0.1, X_test[:, 3].max() + 0.1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
grid = np.c_[xx.ravel(), yy.ravel()]

grid_pred = model.predict(grid[:,0:1])

grid_predicted = grid_pred.reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, grid_predicted, alpha=0.5, cmap='viridis')

# Set the labels and title
plt.xlabel('Feature 3')
plt.ylabel('Feature 4')
plt.title('Learned Decision Boundary')

# Show the plot
plt.show()

##Para calcular los puntos de corte: Algoritmo montecarlo

#Crear una serie de puntos uniformemente distribuidos.
X = np.linspace(x_min, x_max, 4000).reshape(4000,1)
#Calcular en que momento cambia la predicción. Ese sería el cutpoint.
y_pred = model.predict(X)
c = y_pred[0]
Cutpoints = []
Valores = [c]
for i, prediccion in enumerate(y_pred):
    if prediccion != c:
        c = prediccion
        Cutpoints.append(X[i].item())
        Valores.append(c)
print("Cutpoints: ", Cutpoints)
