from sklearn.datasets import load_wine
from DNDTv2 import DNDTModel
import numpy as np



dataset = load_wine()

# Split the dataset into features (X) and labels (y)
X = dataset.data
y = dataset.target


num_features = X.shape[1]
num_classes = np.unique(y).size

model = DNDTModel.DNDTClassifier(num_features, num_classes)

model.fit(X,y,5000)


res = model.predict(X)

