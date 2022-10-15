import numpy as np
from logistic_regression import *


N = 100
d = 10
xt = np.random.randn(N, d)
w_star = np.random.randn(d)
# noise = 10 * np.random.randn(N)
y = (xt.dot(w_star) >0) * 0.99 + .01

initial_w = np.zeros(d)
# w, loss = logistic_regression(y, xt, initial_w, 1000, 0.1)
w, loss = reg_logistic_regression(y, xt, 0.1 ,initial_w, 1000, 0.1)

print(w_star/np.linalg.norm(w_star))
print(w/np.linalg.norm(w))
print(loss)

# print(sigmoid(xt.dot(w)))
