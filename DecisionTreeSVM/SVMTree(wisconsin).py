# importar las bibliotecas necesarias
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# cargar el conjunto de datos iris
iris = load_breast_cancer()
# iris = load_wine()
# iris = load_breast_cancer()

n_features = 30
# n_features = 13
# n_features = 30

n_classes = 2
# n_classes = 2

# separar los datos en características y etiquetas
X = iris.data
y = iris.target

# dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

def treeSVM(X, y, nc, nv, depth = 3):

    if depth == 0:
        return np.bincount(y).argmax()

    mejorV = 0
    mejorAcc = 0
    mejorModelo = SVC(kernel='linear')

    for variable in range(nv):

        model = SVC(kernel='linear')
        model.fit(X[:,variable: variable + 1], y)
        y_pred = model.predict(X[:,variable: variable + 1])

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

    y_pred = mejorModelo.predict(X[:,mejorV: mejorV + 1])

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
            tree["Hijos"][key] = treeSVM(np.array(pobx[key]).reshape((len(pobx[key]), nv)), np.array(value), nc, nv,
                                         depth - 1)

    return tree

def find_index_to_insert(arr, new_int):
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
def predictor(X, tree):

    if isinstance(tree, np.int64):
        return tree
    else:
        var = tree["Variable"]
        xaux = X[:,var:var + 1]
        xaux = xaux[(0,0)]

        ind = find_index_to_insert(tree['Cutpoints'], xaux)

        pre = tree["Valores"][ind]

        return predictor(X, tree["Hijos"][pre])

tree = treeSVM(X_train, y_train, n_classes, n_features, 12)

cont = 0
results = []
for i in range(len(y_test)):
    # print(predictor(X_test[i].reshape(1,13),tree))
    if predictor(X_test[i].reshape(1, n_features), tree) == y_test[i]:
        cont += 1
    results.append(predictor(X_test[i].reshape(1,n_features),tree))

print(cont/len(y_test))

results_array = np.array(results)

y_pred = results_array
y_true = y_test

conM = confusion_matrix(y_true, y_pred)

# tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average=None)
recall = recall_score(y_true, y_pred, average=None)
f1 = f1_score(y_true, y_pred, average=None)

print("matriz de confusión: ", conM)
print("Accuracy: ",accuracy)
print("precision: ", precision)
print("recall: ",recall)
print("f1-score: ",f1)