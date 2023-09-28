import numpy as np
import torch
from functools import reduce
from sklearn.base import BaseEstimator, ClassifierMixin


class DNDTClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_features, n_classes):
        # Inicializa los atributos de tu modelo

        self.num_cut = [1 for i in range(n_features)]  # "Petal length" and "Petal width"
        self.num_leaf = np.prod(np.array(self.num_cut) + 1)
        self.num_class = n_classes

        self.cut_points_list = [torch.rand([i], requires_grad=True) for i in self.num_cut]

        self.leaf_score = torch.rand([self.num_leaf, self.num_class], requires_grad=True)

        self.loss_function = torch.nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(self.cut_points_list + [self.leaf_score], lr=0.01)

    def fit(self, X, y, epoch):
        # Implementa el código para entrenar tu modelo

        x = torch.from_numpy(X.astype(np.float32))
        y = torch.from_numpy(y)

        for i in range(epoch):
            self.optimizer.zero_grad()
            y_pred = self.nn_decision_tree(x, self.cut_points_list, self.leaf_score, temperature=0.1)
            loss = self.loss_function(y_pred, y)
            loss.backward()
            self.optimizer.step()
            if i % 200 == 0:
                print(loss.detach().numpy())
        print('error rate %.2f' % (1 - (sum(np.argmax(y_pred.detach().numpy(), axis=1) == y.numpy()) / len(y) )))
    def predict(self, X):
        # Implementa el código para realizar predicciones
        x = torch.from_numpy(X.astype(np.float32))


        return np.argmax(self.nn_decision_tree(x, self.cut_points_list, self.leaf_score, temperature=0.1).detach().numpy(), axis=1)

    def score(self, X, y):
        # Implementa el código para evaluar el rendimiento del modelo



        return None

    # Puedes añadir otros métodos y propiedades necesarios para tu modelo
    def torch_kron_prod(self, a, b):
        res = torch.einsum('ij,ik->ijk', [a, b])
        res = torch.reshape(res, [-1, np.prod(res.shape[1:])])
        return res

    def torch_bin(self, x, cut_points, temperature=0.1):
        # x is a N-by-1 matrix (column vector)
        # cut_points is a D-dim vector (D is the number of cut-points)
        # this function produces a N-by-(D+1) matrix, each row has only one element being one and the rest are all zeros
        D = cut_points.shape[0]
        W = torch.reshape(torch.linspace(1.0, D + 1.0, D + 1), [1, -1])
        cut_points, _ = torch.sort(cut_points)  # make sure cut_points is monotonically increasing
        b = torch.cumsum(torch.cat([torch.zeros([1]), -cut_points], 0), 0)
        h = torch.matmul(x, W) + b
        # res = torch.exp(h-torch.max(h))
        # res = res/torch.sum(res, dim=-1, keepdim=True)
        m = torch.nn.Softmax(dim=D)
        res = m(h / temperature)

        # res = torch.nn.functional.softmax(h/temperature, D)

        return res

    def nn_decision_tree(self, x, cut_points_list, leaf_score, temperature=0.1):
        # cut_points_list contains the cut_points for each dimension of feature
        leaf = reduce(self.torch_kron_prod,
                      map(lambda z: self.torch_bin(x[:, z[0]:z[0] + 1], z[1], temperature), enumerate(cut_points_list)))
        return torch.matmul(leaf, leaf_score)
