import numpy as np
from model import NN1

X = np.array([
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
    [0.0, 1.0, 1.0],
    [1.0, 0.0, 0.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
    [1.0, 1.0, 1.0]
], dtype=np.float32)

y = np.array([
    [0.0],
    [1.0],
    [1.0],
    [0.0],
    [1.0],
    [0.0],
    [0.0],
    [1.0]
], dtype=np.float32)

model = NN1(X,y)
model.train()
# Try these test cases in testing.py:
test_x_zero = np.array([[0.0, 0.0, 0.0]])  # Even ones -> Outputs [[0.]]
test_x_one  = np.array([[0.0, 1.0, 0.0]])  # Odd ones  -> Outputs [[1.]]
test_x_two  = np.array([[0.0, 1.0, 1.0]])  # Even ones -> Outputs [[0.]]

print("Testing [0, 0, 0]:")
model.prediction(test_x_zero)

print("\nTesting [0, 1, 0]:")
model.prediction(test_x_one)

print("\nTesting [0, 1, 1]:")
model.prediction(test_x_two)
