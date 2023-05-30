import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
import numpy as np

#Cargamos Dataset:

dataset = load_iris()

X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.2)

#Definimos parametros:
#Cuantas variables predictoras
num_features = dataset.data.shape[1]

#Cuantas clases diferentes
num_classes = np.unique(dataset.target).shape[0]

#Entrenamiento por nivel
epoch = 3000

#Cuantos niveles maximos
depth = 12

#Condición de parada. Si la diferencia entre 2 iteraciones es menor a este valor
#deja de entrenar
stop_condition = 10**(-7)



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, num_classes)

    def forward(self, x):
        return self.fc1(x)

def treeNet(X, y, epoch, nc, nv, l, sc, depth = 3):

    if depth == 0:
        return np.bincount(y).argmax()

    mejorLoss = float("Inf")
    mejorNet = Net()
    mejorVar = 0

    tree = {"Variable": 0, "Loss": float("Inf"), "Net": mejorNet,
            "data": X, "target":y}

    for i in range(nv):

        print("Variable: ", i, "Depth:", depth)

        Xaux = X[:,i:i+1]

        # Instantiate the neural network
        net = Net()

        # Define the loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters())

        lossAux = float("Inf")
        for epoch in range(epoch * int(l/len(y))):
            optimizer.zero_grad()
            outputs = net(torch.tensor(Xaux, dtype=torch.float32))
            loss = criterion(outputs, torch.tensor(y, dtype=torch.long))
            loss.backward()
            optimizer.step()

            if abs(loss.item() - lossAux) < sc:
                break
            lossAux = loss.item()
            # Print the loss every 10 epochs
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss={loss.item():.4f}")
        if loss.item() < mejorLoss:
            tree["Variable"], tree["Loss"], tree["Net"] = i, loss.item(), net
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
    for i in enumerate(X[:,mejorVar:mejorVar+1]):
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
            tree["Hijos"][key] = (
                treeNet(np.array(pobx[key]).reshape((len(pobx[key]), nv)), np.array(value), epoch, nc, nv, l,
                        depth-1))
    return tree


tree = treeNet(X_train,y_train, epoch, num_classes, num_features, len(y_train),stop_condition, depth)