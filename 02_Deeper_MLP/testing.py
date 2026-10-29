import numpy as np

from model import NN2

import numpy as np

X = np.array([
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1]
], dtype=np.float32)

y = np.array([
    [0],
    [0],
    [0],
    [1],
    [0],
    [1],
    [1],
    [1]
], dtype=np.float32)

model = NN2(X,y)
model.train()

test_data_1 = np.array([[1,0,1]])# 1
test_data_2 = np.array([[1,0,0]])# 0
test_data_3 = np.array([[0,0,1]])# 0
test_data_4 = np.array([[1,1,1]])# 1

model.prediction(test_data_1)
model.prediction(test_data_2)
model.prediction(test_data_3)
model.prediction(test_data_4)