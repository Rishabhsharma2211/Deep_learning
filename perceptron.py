import numpy as np

class Perceptron:
    def __init__(self, weights, bias):
        self.w = np.array(weights)
        self.b = bias

    def step_function(self, z):
        return 1 if z >= 0 else 0

    def predict(self, x):
        z = np.dot(self.w, x) + self.b
        return self.step_function(z)

# Example usage
p = Perceptron([0.2, 0.4], -0.5)
x = np.array([1.0, 2.0])

output = p.predict(x)
print("Perceptron Output:", output)
