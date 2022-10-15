import numpy as np

def sigmoid(t):
    return 1.0 / (1 + np.exp(-t))

def mle_loss(y, tx, w):
    """negative log likelihood."""
    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    return np.squeeze(- loss)

def grad(y, tx, w):
    """the gradient of loss"""
    pred = sigmoid(tx.dot(w))
    grad = tx.T.dot(pred - y)
    return grad

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    running gradient descent for max_iters iterations (at most)
    return weights, loss
    """
    w = initial_w
    threshold = 1e-6
    for t in range(max_iters):
        loss = mle_loss(y, tx, w)
        gradient = grad(y, tx, w)
        w -= gamma * gradient

    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    running gradient descent for max_iters iterations (at most)
    return weights, loss
    """
    w = initial_w
    threshold = 1e-6
    for t in range(max_iters):
        loss = mle_loss(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))
        gradient = grad(y, tx, w) + 2 * lambda_ * w
        w -= gamma * gradient

    loss = mle_loss(y, tx, w)
    return w, loss
