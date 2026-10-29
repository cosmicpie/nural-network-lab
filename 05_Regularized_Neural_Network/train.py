import numpy as np
from model import NN5
X = np.array([
    [800,  2, 10, 8],
    [1000, 2,  8, 6],
    [1200, 3,  7, 5],
    [1400, 3,  5, 4],
    [1600, 3,  4, 3],
    [1800, 4,  3, 2],
    [2000, 4,  2, 2],
    [2200, 5,  1, 1]
], dtype=np.float32)

y = np.array([
    [25],
    [30],
    [38],
    [45],
    [52],
    [60],
    [70],
    [82]
], dtype=np.float32)


mean = np.mean(X, axis=0)
std = np.std(X, axis=0)

X_scaled = (X - mean) / std

model = NN5(X_scaled,y)

model.train()

test_X = np.array([
    [900,  2,  9, 7],   # expected ≈ 27–30 lakh
    [1300, 3,  6, 5],   # expected ≈ 40–42 lakh
    [1700, 4,  3, 3],   # expected ≈ 55–60 lakh
    [1900, 4,  2, 2],   # expected ≈ 65–68 lakh
    [2100, 5,  1, 1]    # expected ≈ 75–80 lakh
], dtype=np.float32)

test_X_scaled = (test_X - mean) / std

model.predict(test_X_scaled[0])
model.predict(test_X_scaled[1])
model.predict(test_X_scaled[2])
model.predict(test_X_scaled[3])
model.predict(test_X_scaled[4])