import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Inputs
x = np.array([1.0, 2.0])

# Hidden layer
W1 = np.array([[0.1, 0.2],
               [0.3, 0.4]])
b1 = np.array([0.1, 0.1])

# Output layer
W2 = np.array([0.5, 0.6])
b2 = 0.2

# Forward propagation
z1 = np.dot(W1, x) + b1
a1 = sigmoid(z1)

z2 = np.dot(W2, a1) + b2
output = sigmoid(z2)

print("MLP Output:", output)
