import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
import torch
import torch.nn as nn
import torch.optim as optim



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(1, 5)

    def forward(self, x):
        x = self.fc(x)
        return x

class TreeNetClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, epoch, nc, nv, l, sc, depth=3):
        self.epoch = epoch
        self.nc = nc
        self.nv = nv
        self.l = l
        self.sc = sc
        self.depth = depth
        self.model = None

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        X = X.astype(np.float32)
        y = y.astype(np.int64)

        mejorLoss = float("Inf")
        mejorNet = Net()
        mejorVar = 0

        for i in range(self.nv):
            Xaux = X[:, i:i+1]

            net = Net()

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(net.parameters())

            lossAux = float("Inf")
            for epoch in range(self.epoch * int(self.l / len(y))):
                optimizer.zero_grad()
                outputs = net(torch.tensor(Xaux, dtype=torch.float32))
                loss = criterion(outputs, torch.tensor(y, dtype=torch.long))
                loss.backward()
                optimizer.step()

                if abs(loss.item() - lossAux) < self.sc:
                    break
                lossAux = loss.item()

                if epoch % 100 == 0:
                    print(f"Epoch {epoch}: Loss={loss.item():.4f}")

            if loss.item() < mejorLoss:
                mejorVar = i
                mejorNet = net
                mejorLoss = loss.item()

        self.model = mejorNet

    def predict(self, X):
        X = check_array(X)
        X = X.astype(np.float32)

        with torch.no_grad():
            outputs = self.model(torch.tensor(X, dtype=torch.float32))
            predictions = outputs.argmax(axis=1).numpy()

        return predictions


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
