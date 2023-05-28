import numpy as np
import torch
from functools import reduce
import matplotlib.pyplot as plt
from sklearn import datasets
import pickle


np.random.seed(1943)
torch.manual_seed(1943)

def torch_kron_prod(a, b):
    res = torch.einsum('ij,ik->ijk', [a, b])
    res = torch.reshape(res, [-1, np.prod(res.shape[1:])])
    return res

def torch_bin(x, cut_points, temperature=0.1):
    # x is a N-by-1 matrix (column vector)
    # cut_points is a D-dim vector (D is the number of cut-points)
    # this function produces a N-by-(D+1) matrix, each row has only one element being one and the rest are all zeros
    D = cut_points.shape[0]
    W = torch.reshape(torch.linspace(1.0, D + 1.0, D + 1), [1, -1])
    cut_points, _ = torch.sort(cut_points)  # make sure cut_points is monotonically increasing
    b = torch.cumsum(torch.cat([torch.zeros([1]), -cut_points], 0),0)
    h = torch.matmul(x, W) + b
    # res = torch.exp(h-torch.max(h))
    # res = res/torch.sum(res, dim=-1, keepdim=True)
    m = torch.nn.Softmax(dim=D)
    res = m(h/temperature)

    #res = torch.nn.functional.softmax(h/temperature, D)

    return res

def nn_decision_tree(x, cut_points_list, leaf_score, temperature=0.1):
    # cut_points_list contains the cut_points for each dimension of feature
    leaf = reduce(torch_kron_prod,
                  map(lambda z: torch_bin(x[:, z[0]:z[0] + 1], z[1], temperature), enumerate(cut_points_list)))
    return torch.matmul(leaf, leaf_score)

def DNDTv2(X, y, nf, nc, train, profundidad, tamano):

    if profundidad == 0:
        return 0

    num_cut = [1]
    num_leaf = np.prod(np.array(num_cut) + 1)
    num_class = nc

    scores = []

    BV = 0
    BVL = float('Inf')

    for var in range(nf):

        x = X[:, var:var + 1]

        cut_points_list = [torch.rand([i], requires_grad=True) for i in num_cut]
        leaf_score = torch.rand([num_leaf, num_class], requires_grad=True)
        loss_function = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(cut_points_list + [leaf_score], lr=0.01)

        aux = 1

        for i in range(train * int(tamano/X.shape[0])):
            optimizer.zero_grad()
            y_pred = nn_decision_tree(x, cut_points_list, leaf_score, temperature=0.1)
            loss = loss_function(y_pred, y)
            loss.backward()
            optimizer.step()
            if i % 200 == 0:
                print(loss.detach().numpy())
                if abs(aux - loss.detach().numpy()) < 10**(-5):
                    break
                aux = loss.detach().numpy()

        print("TOTAL: ", loss.detach().numpy())

        if BVL > loss.detach().numpy():
            BV = var
            BVL = loss.detach().numpy()

        scores += [{'Variable': var, 'cut_points_list': cut_points_list,
                  'leaf_score': leaf_score, 'loss': loss.detach().numpy(),
                  'yHijos': [], 'hijos': [], 'X': X, 'y': y}]

    x = X[:, BV:BV + 1]

    cut_points_list = scores[BV]['cut_points_list']
    leaf_score = scores[BV]['leaf_score']

    y_pred = nn_decision_tree(x, cut_points_list, leaf_score, temperature=0.1)
    y_pred = np.argmax(y_pred.detach().numpy(), axis=1)

    # Aquí creamos los cutpoints reales que calcula.
    # Create a range of values for the feature
    x_values = np.linspace(X.min() - 1, X.max() + 1, 2000).reshape(-1, 1)
    y_pred2 = nn_decision_tree(torch.from_numpy(x_values.astype(np.float32)), cut_points_list, leaf_score,
                               temperature=0.1)
    y_pred2 = np.argmax(y_pred2.detach().numpy(), axis=1)

    c = y_pred2[0]
    MisCutpoints = []
    for i, prediccion in enumerate(y_pred2):

        if prediccion != c:
            c = prediccion
            MisCutpoints.append(x_values[i])

    scores[BV]['MisCutpoints'] = MisCutpoints

    pobx = [[] for i in range(3)]
    poby = [[] for i in range(3)]

    for i, element in enumerate(y_pred):
        pobx[element].append(X[i].numpy())
        poby[element].append(y[i].numpy())

    for i in range(nc):
        if len(pobx[i]) > 0:
            scores[BV]['yHijos'].append(i)
            scores[BV]['hijos'].append(
                DNDTv2(torch.tensor(np.array(pobx[i])),
                       torch.from_numpy(np.array(poby[i]).astype(np.compat.long)).type(torch.LongTensor), nf, nc, train,
                       profundidad - 1, tamano)
            )

    return scores[BV]

# Load the iris dataset
iris = datasets.load_iris()
X = iris.data
Y = iris.target

_x = X[:,2:4]
_y = Y

x = torch.from_numpy(_x.astype(np.float32))
y = torch.from_numpy(_y.astype(np.compat.long))
y = y.type(torch.LongTensor)

res = DNDTv2(x,y,2,3,10000, 4, x.shape[0])

with open('diccionario_entrenado.pkl', 'wb') as f:
    pickle.dump(res, f)