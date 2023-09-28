from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import numpy as np


class TreeSVMClassifier():

    def __init__(self, num_features, num_classes, depth):
        # Initialize your model with any required parameters
        self.num_features = num_features
        self.num_classes = num_classes
        self.depth = depth
        self.root = None

        self.n_nodos = None
        self.n_hojas = None
        self.profundidad = None
        self.longitudMediaRamas = None

        # Initialize any other variables or attributes needed by your model

    def fit(self, X, y):
        # Implement the training logic for your model
        # X: array-like, shape (n_samples, n_features)
        # y: array-like, shape (n_samples,)
        # This method should update the model's internal state based on the training data

        self.root = self.tree_SVM(X, y, self.num_classes, self.num_features, self.depth)


        self.n_nodos = self.num_nodos(self.root)
        self.n_hojas = self.num_hojas(self.root)
        self.profundidad = self.branch_length(self.root)
        self.longitudMediaRamas = self.averageBranchesLength(self.root)

        return self

    def tree_SVM(self, X, y, num_classes, num_variables, depth=3):

        if depth == 0:
            return np.bincount(y).argmax()

        mejorV = 0
        mejorAcc = 0
        mejorModelo = SVC(kernel='linear')

        for variable in range(num_variables):

            model = SVC(kernel='linear')
            model.fit(X[:, variable: variable + 1], y)
            y_pred = model.predict(X[:, variable: variable + 1])

            accuracy = accuracy_score(y, y_pred)
            print("Precisión del modelo SVM: {:.2f}%".format(accuracy * 100))

            if accuracy > mejorAcc:
                mejorV = variable
                mejorAcc = accuracy
                mejorModelo = model

        tree = {
            "Variable": mejorV, "Accuracy": mejorAcc,
            "data": X, "target": y
        }

        x_values = np.linspace(X.min() - 1, X.max() + 1, 2000).reshape(-1, 1)

        y_pred = mejorModelo.predict(x_values)

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

        y_pred = mejorModelo.predict(X[:, mejorV: mejorV + 1])

        pobx = {}
        poby = {}

        for i in enumerate(y_pred):
            pre = i[1]
            if pre not in poby:
                pobx[pre], poby[pre] = [], []
            pobx[pre].append(X[i[0]])
            poby[pre].append(y[i[0]])

        tree["pobx"] = pobx
        tree["poby"] = poby

        num_hijos = len(poby)
        tree["Hijos"] = {}
        for key, value in poby.items():
            if len(np.unique(np.array(value))) == 1 or num_hijos == 1:
                tree["Hijos"][key] = np.bincount(value).argmax()
            else:
                tree["Hijos"][key] = self.tree_SVM(np.array(pobx[key]).reshape((len(pobx[key]), num_variables)), np.array(value), num_classes, num_variables,
                                             depth - 1)

        return tree

    def predict(self, X):
        # Implement the prediction logic for your model
        # X: array-like, shape (n_samples, n_features)
        # This method should return the predicted labels for the input samples

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

            try:
                return self.traverse_tree(X, node["Hijos"][pre])
            except:
                return pre

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