import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array

class TreeNetClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, epoch, nc, nv, l, sc, depth=3):
        self.epoch = epoch
        self.nc = nc
        self.nv = nv
        self.l = l
        self.sc = sc
        self.depth = depth
        self.tree = None

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.tree = self.treeNet(X, y, self.epoch, self.nc, self.nv, self.l, self.sc, self.depth)

    def predict(self, X):
        X = check_array(X)
        predictions = []
        for x in X:
            prediction = self.traverseTree(self.tree, x)
            predictions.append(prediction)
        return np.array(predictions)

    def treeNet(self, X, y, epoch, nc, nv, l, sc, depth):
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
            Xaux = X[:, i:i+1]

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

            if loss.item() < mejorLoss:
                tree["Variable"], tree["Loss"] = i, loss.item()
                mejorVar = i
                mejorNet = net
                mejorLoss = loss.item()

        pobx = {}
        poby = {}
        for i, x in enumerate(X[:, mejorVar:mejorVar+1]):
            pre = mejorNet(torch.tensor(i[1], dtype=torch.float32)).argmax().item()
            if pre not in poby:
                pobx[pre], poby[pre] = [], []
            pobx[pre].append(X[i])
            poby[pre].append(y[i])

        num_hijos = len(poby)
        tree["Hijos"] = {}
        for key, value in poby.items():
            if len(np.unique(np.array(value))) == 1 or num_hijos == 1:
                tree["Hijos"][key] = np.bincount(value).argmax()
            else:
                tree["Hijos"][key] = self.treeNet(
                    np.array(pobx[key]).reshape((len(pobx[key]), nv)), np.array(value), epoch, nc, nv, l, sc, depth-1)

        return tree

    def traverseTree(self, tree, x):
        if isinstance(tree, int):
            return tree

        variable = tree["Variable"]
        cutpoints = tree["Cutpoints"]
        valores = tree["Valores"]
        hijos = tree["Hijos"]

        value = x[variable]
        child_index = 0
        for i, cutpoint in enumerate(cutpoints):
            if value > cutpoint:
                child_index = i + 1

        return self.traverseTree(hijos[valores[child_index]], x)



from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generar datos de ejemplo
X, y = make_classification(n_samples=1000, n_features=5, n_informative=3, n_classes=2, random_state=42)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear una instancia del modelo
model = TreeNetClassifier(epoch=10000, nc=2, nv=5, l=1000, sc=0.0001, depth=3)

# Ajustar el modelo a los datos de entrenamiento
model.fit(X_train, y_train)

# Hacer predicciones sobre los datos de prueba
predictions = model.predict(X_test)

# Calcular la precisión
accuracy = accuracy_score(y_test, predictions)
print("Precisión:", accuracy)
