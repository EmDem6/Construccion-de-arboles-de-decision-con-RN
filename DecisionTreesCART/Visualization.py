from sklearn import tree
import graphviz

from sklearn.datasets import  load_breast_cancer
from sklearn.datasets import  load_iris



dataset = load_iris()

X = dataset.data
y = dataset.target

model = tree.DecisionTreeClassifier()

model.fit(X, y)

dot_data = tree.export_graphviz(model, out_file=None)

graph = graphviz.Source(dot_data)


graph.render("ruta_del_archivo.png", format="png")
