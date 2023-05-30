import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 3)

    def forward(self, x):
        return self.fc1(x)

iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris.data[:,2:4], iris.target, test_size=0.2)

# Instantiate the neural network
net = Net()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())

for epoch in range(100000):
    optimizer.zero_grad()
    outputs = net(torch.tensor(X_train[:,1:2], dtype=torch.float32))
    loss = criterion(outputs, torch.tensor(y_train, dtype=torch.long))
    loss.backward()
    optimizer.step()

    # Print the loss every 10 epochs
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss={loss.item():.4f}")

net.eval()

with torch.no_grad():
    test_outputs = net(torch.tensor(X_test[:,1:2], dtype=torch.float32))
    predicted = torch.argmax(test_outputs, dim=1).numpy()

# Plot the data points
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis')

# Create a meshgrid of points
x_min, x_max = X_test[:, 0].min() - 0.1, X_test[:, 0].max() + 0.1
y_min, y_max = X_test[:, 1].min() - 0.1, X_test[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
grid = np.c_[xx.ravel(), yy.ravel()]

# Convert the grid to a tensor
grid_tensor = torch.tensor(grid, dtype=torch.float32)

# Make predictions on the grid points
with torch.no_grad():
    grid_outputs = net(grid_tensor[:,1:2])
    grid_predicted = torch.argmax(grid_outputs, dim=1).numpy()

# Reshape the grid predictions
grid_predicted = grid_predicted.reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, grid_predicted, alpha=0.5, cmap='viridis')

# Set the labels and title
plt.xlabel('Feature 3')
plt.ylabel('Feature 4')
plt.title('Learned Decision Boundary')

# Show the plot
plt.show()

print(accuracy_score(predicted,y_test))

#Crear una serie de puntos uniformemente distribuidos.
X = np.linspace(y_min, y_max, 4000).reshape(4000,1)

#Calcular en que momento cambia la predicción. Ese sería el cutpoint.
y_pred = net(torch.tensor(X, dtype=torch.float32)).argmax(axis=1).numpy()
c = y_pred[0]
Cutpoints = []
Valores = [c]
for i, prediccion in enumerate(y_pred):
    if prediccion != c:
        c = prediccion
        Cutpoints.append(X[i].item())
        Valores.append(c)

print("Cutpoints: ", Cutpoints)