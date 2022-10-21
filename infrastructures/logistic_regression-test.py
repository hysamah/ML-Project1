import numpy as np
from logistic_regression import *

#
# N = 100
# d = 10
# xt = np.random.randn(N, d)
# w_star = np.random.randn(d)
# # noise = 10 * np.random.randn(N)
# y = (xt.dot(w_star) >0) * 0.99 + .01
#
# initial_w = np.zeros(d)
# # w, loss = logistic_regression(y, xt, initial_w, 1000, 0.1)
# w, loss = reg_logistic_regression(y, xt, 0.1 ,initial_w, 1000, 0.1)
#
# print(w_star/np.linalg.norm(w_star))
# print(w/np.linalg.norm(w))
# print(loss)
#
# # print(sigmoid(xt.dot(w)))

MAX_ITERS = 2
GAMMA = 0.1
y = np.array([[0.1], [0.3], [0.5]])
tx = np.array([[2.3, 3.2], [1.0, 0.1], [1.4, 2.3]])
initial_w = np.array([[0.5], [1.0]])
RTOL = 1e-4
ATOL = 1e-8

# print(logistic_regression(y, tx, np.array([[0.463156], [0.939874]]), 0, GAMMA))

def test_logistic_regression_0_step(y, tx):
    expected_w = np.array([[0.463156], [0.939874]])
    y = (y > 0.2) * 1.0
    w, loss = logistic_regression(y, tx, expected_w, 0, GAMMA)

    expected_loss = 1.533694

    np.testing.assert_allclose(loss, expected_loss, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(w, expected_w, rtol=RTOL, atol=ATOL)
    assert loss.ndim == 0
    assert w.shape == expected_w.shape

test_logistic_regression_0_step(y, tx)


def test_logistic_regression(y, tx, initial_w):
    y = (y > 0.2) * 1.0
    w, loss = logistic_regression(
        y, tx, initial_w, MAX_ITERS, GAMMA
    )
    # print(w, loss)
    expected_loss = 1.348358
    expected_w = np.array([[0.378561], [0.801131]])

    np.testing.assert_allclose(loss, expected_loss, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(w, expected_w, rtol=RTOL, atol=ATOL)
    assert loss.ndim == 0
    assert w.shape == expected_w.shape

test_logistic_regression(y, tx, initial_w)


def test_reg_logistic_regression(y, tx, initial_w):
    lambda_ = 1.0
    y = (y > 0.2) * 1.0
    w, loss = reg_logistic_regression(
        y, tx, lambda_, initial_w, MAX_ITERS, GAMMA
    )

    print(w, loss)

    expected_loss = 0.972165
    expected_w = np.array([[0.216062], [0.467747]])

    np.testing.assert_allclose(loss, expected_loss, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(w, expected_w, rtol=RTOL, atol=ATOL)
    assert loss.ndim == 0
    assert w.shape == expected_w.shape

# test_reg_logistic_regression(y, tx, initial_w)

def test_reg_logistic_regression_0_step(y, tx):
    lambda_ = 1.0
    expected_w = np.array([[0.409111], [0.843996]])
    y = (y > 0.2) * 1.0
    w, loss = reg_logistic_regression(
        y, tx, lambda_, expected_w, 0, GAMMA
    )

    expected_loss = 1.407327

    np.testing.assert_allclose(loss, expected_loss, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(w, expected_w, rtol=RTOL, atol=ATOL)
    assert loss.ndim == 0
    assert w.shape == expected_w.shape

test_reg_logistic_regression_0_step(y, tx)

print(reg_logistic_regression(
    y, tx, 9, np.array([[0.409111], [0.843996]]), 1000, GAMMA
))
