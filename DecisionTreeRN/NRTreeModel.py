from sklearn.base import BaseEstimator, ClassifierMixin

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

class TreeNetClassifier(BaseEstimator, ClassifierMixin):
    
    def __init__(self, num_features, num_classes, epoch, depth, stop_condition):
        # Initialize your model with any required parameters
        self.num_features = num_features
        self.num_classes = num_classes
        self.epoch = epoch
        self.depth = depth
        self.stop_condition = stop_condition
        self.root = None

        # Initialize any other variables or attributes needed by your model

    def fit(self, X, y):
        # Implement the training logic for your model
        # X: array-like, shape (n_samples, n_features)
        # y: array-like, shape (n_samples,)
        # This method should update the model's internal state based on the training data

        self.root = self.tree_net(X, y, self.epoch, self.num_classes, self.num_features, len(y), self.stop_condition, self.depth)

        return self

    def tree_net(self, X, y, epoch, nc, nv, l, sc, depth):

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.fc = nn.Linear(1, nc)

            def forward(self, x):
                x = self.fc(x)
                return x

        if depth == 0:
            return np.bincount(y).argmax()

        mejorLoss = float("Inf")
        mejorNet = Net()
        mejorVar = 0

        tree = {"Variable": 0, "Loss": float("Inf"), "data": X, "target": y}

        for i in range(nv):
            Xaux = X[:, i:i + 1]

            net = Net()
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(net.parameters())

            lossAux = float("Inf")
            for _ in range(epoch * int(l / len(y))):
                optimizer.zero_grad()
                outputs = net(torch.tensor(Xaux, dtype=torch.float32))
                loss = criterion(outputs, torch.tensor(y, dtype=torch.long))
                loss.backward()
                optimizer.step()

                if abs(loss.item() - lossAux) < sc:
                    break
                lossAux = loss.item()

                if _ % 100 == 0:
                    print(f"Epoch {_}: Loss={loss.item():.4f}")

            if loss.item() < mejorLoss:
                tree["Variable"], tree["Loss"] = i, loss.item()
                mejorVar = i
                mejorNet = net
                mejorLoss = loss.item()

        x_values = np.linspace(X.min() - 1, X.max() + 1, 2000).reshape(-1, 1)

        y_pred = mejorNet(torch.tensor(x_values, dtype=torch.float32)).argmax(axis=1).numpy()

        c = y_pred[0]
        Cutpoints = []
        Valores = [c]
        for i, prediccion in enumerate(y_pred):
            if prediccion != c:
                c = prediccion
                Cutpoints.append(x_values[i].item())
                Valores.append(c)

        tree["Cutpoints"] = Cutpoints
        tree["Valores"] = Valores

        pobx = {}
        poby = {}
        for i in enumerate(X[:, mejorVar:mejorVar + 1]):
            pre = mejorNet(torch.tensor(i[1], dtype=torch.float32)).argmax().item()
            if pre not in poby:
                pobx[pre], poby[pre] = [], []
            pobx[pre].append(X[i[0]])
            poby[pre].append(y[i[0]])

        num_hijos = len(poby)
        tree["Hijos"] = {}
        for key, value in poby.items():
            if len(np.unique(np.array(value))) == 1 or num_hijos == 1:
                tree["Hijos"][key] = np.bincount(value).argmax()
            else:
                tree["Hijos"][key] = self.tree_net(
                    np.array(pobx[key]).reshape((len(pobx[key]), nv)), np.array(value), epoch, nc, nv, l, sc, depth - 1)

        return tree

    def predict(self, X):
        # Implement the prediction logic for your model
        # X: array-like, shape (n_samples, n_features)
        # This method should return the predicted labels for the input samples

        predictions = []
        for sample in X:
            label = self.traverse_tree(sample.reshape(1,self.num_features), self.root)
            predictions.append(label)

        return predictions

    def traverse_tree(self, X, node):
        if isinstance(node, np.int64):
            return node
        else:
            var = node["Variable"]
            xaux = X[:, var:var + 1]
            xaux = xaux[(0, 0)]

            ind = self.find_index_to_insert(node['Cutpoints'], xaux)

            pre = node["Valores"][ind]

            return self.traverse_tree(X, node["Hijos"][pre])

    def find_index_to_insert(self, arr, new_int):
        left = 0
        right = len(arr) - 1

        while left <= right:
            mid = (left + right) // 2

            if new_int < arr[mid]:
                right = mid - 1
            elif new_int > arr[mid]:
                left = mid + 1
            else:
                return mid
        return left


from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generar datos de ejemplo
X, y = make_classification(n_samples=10000, n_features=5, n_informative=3, n_classes=2, random_state=42)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear una instancia del modelo
model = TreeNetClassifier(num_features=5, num_classes=3, stop_condition=0.0001, epoch=30000, depth=12)

# Ajustar el modelo a los datos de entrenamiento
model.fit(X_train, y_train)

# Hacer predicciones sobre los datos de prueba
predictions = model.predict(X_test)#TODO: DEjar esto así. Pero luego cambiar el predictor.

# Calcular la precisión
accuracy = accuracy_score(y_test, predictions)
print("Precisión:", accuracy)


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# Assuming you have your data stored in 'X' and target variable in 'y'

# Create the k-fold cross-validation object
k = 5  # Number of folds
kf = KFold(n_splits=k, shuffle=True)


# Perform k-fold cross-validation
scores = cross_val_score(model, X, y, cv=kf)

# Print the accuracy scores for each fold
for i, score in enumerate(scores):
    print(f"Fold {i+1} Accuracy: {score}")

# Print the mean accuracy and standard deviation
print(f"Mean Accuracy: {scores.mean()}")
print(f"Standard Deviation: {scores.std()}")

