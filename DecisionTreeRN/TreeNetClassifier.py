
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim


class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.fc = nn.Linear(1, num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x
class TreeNetClassifier():

    def __init__(self, num_features, num_classes, epoch, depth, stop_condition):
        # Initialize your model with any required parameters
        self.num_features = num_features
        self.num_classes = num_classes
        self.epoch = epoch
        self.depth = depth
        self.stop_condition = stop_condition
        self.root = None

        self.n_nodos = None
        self.n_hojas = None
        self.profundidad = None
        self.longitudMediaRamas = None


    def fit(self, X, y):

        self.root = self.tree_net(X, y, self.epoch, self.num_classes, self.num_features, len(y), self.stop_condition,
                                  self.depth)

        self.n_nodos = self.num_nodos(self.root)
        self.n_hojas = self.num_hojas(self.root)
        self.profundidad = self.branch_length(self.root)
        self.longitudMediaRamas = self.averageBranchesLength(self.root)

        return self

    def tree_net(self, X, y, epoch, num_classes, num_variables, longitud_inicial, stop_condition, depth):

        if depth == 0:
            return np.bincount(y).argmax()

        mejorLoss = float("Inf")
        mejorNet = Net(num_classes)
        mejorVar = 0

        tree = {"Variable": 0, "Loss": float("Inf"), "data": X, "target": y}

        for i in range(num_variables):
            Xaux = X[:, i:i + 1]

            net = Net(num_classes)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(net.parameters())

            lossAux = float("Inf")
            for _ in range(epoch * int(longitud_inicial / len(y))):
                optimizer.zero_grad()
                outputs = net(torch.tensor(Xaux, dtype=torch.float32))
                loss = criterion(outputs, torch.tensor(y, dtype=torch.long))
                loss.backward()
                optimizer.step()

                if abs(loss.item() - lossAux) < stop_condition:
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
                    np.array(pobx[key]).reshape((len(pobx[key]), num_variables)), np.array(value), epoch, num_classes, num_variables, longitud_inicial, stop_condition, depth - 1)

        return tree

    def predict(self, X):

        predictions = []
        for sample in X:
            label = self.traverse_tree(sample.reshape(1, self.num_features), self.root)
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

    def num_nodos(self, nodo):
        res = 1
        hijos = nodo['Hijos']
        if len(hijos) <= 1:
            return 0
        for key, hijo in hijos.items():
            if not isinstance(hijo, np.int64):
                res += self.num_nodos(hijo)
        return res

    def num_hojas(self, nodo):

        if isinstance(nodo, np.int64):
            return 1
        hijos = nodo['Hijos']
        if len(hijos) <= 1:
            return 1

        res = 0
        for key, hijo in hijos.items():
            res += self.num_hojas(hijo)

        return res

    def es_hoja(self, nodo):
        if isinstance(nodo, np.int64):
            return True
        hijos = nodo['Hijos']
        if len(hijos) <= 1:
            return True
        return False

    def branch_length(self, nodo):

        if self.es_hoja(nodo):
            return 1
        return 1 + max(self.branch_length(hijo) for key, hijo in nodo['Hijos'].items())

    def averageBranchesLength(self, root):
        lengths = []
        queue = [(root, 1)]

        while queue:
            node, depth = queue.pop(0)

            if self.es_hoja(node):
                lengths.append(depth)
            else:
                queue.extend([(child, depth + 1) for key, child in node['Hijos'].items()])

        return sum(lengths) / len(lengths)