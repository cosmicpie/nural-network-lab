import numpy as np
from model import NN3

X = np.array([
    [1.0, 1.0, 0.0, 0.0],
    [1.2, 0.8, 0.1, 0.0],
    [0.8, 1.1, 0.2, 0.1],

    [0.0, 1.0, 1.0, 0.0],
    [0.1, 1.2, 0.8, 0.1],
    [0.0, 0.9, 1.1, 0.2],

    [0.0, 0.0, 1.0, 1.0],
    [0.1, 0.1, 1.2, 0.9],
    [0.0, 0.2, 0.9, 1.1]
], dtype=np.float32)

y = np.array([
    [1, 0, 0],
    [1, 0, 0],
    [1, 0, 0],

    [0, 1, 0],
    [0, 1, 0],
    [0, 1, 0],

    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1]
], dtype=np.float32)

model = NN3(X,y)

model.train()

test_1 = np.array([[1.1, 0.9, 0.1, 0.0]], dtype=np.float32) #0

test_2 = np.array([[0.1, 1.1, 0.9, 0.1]], dtype=np.float32) #1

test_3 = np.array([[0.0, 0.1, 1.1, 1.0]], dtype=np.float32) #2

test_4 = np.array([[1.0, 1.1, 0.1, 0.1]], dtype=np.float32) #0

test_5 = np.array([[0.1, 0.2, 1.0, 1.1]], dtype=np.float32) #2

pred = model.prediction(test_2)

print("#"*50)
print(" ")
print(pred)