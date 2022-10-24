import numpy as np

def sigmoid(t):
    p = 1.0 / (1 + np.exp(-t))
    p[p<0.00001] = 0.00001
    p[p>0.99999] = 0.99999
    return(p)

def mle_loss(y, tx, w):
    """negative log likelihood."""
    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    return np.squeeze(- loss / y.shape[0])

def grad(y, tx, w):
    """the gradient of loss"""
    pred = sigmoid(tx.dot(w))
    grad = tx.T.dot(pred - y) / y.shape[0]
    return grad

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    running gradient descent for max_iters iterations (at most)
    return weights, loss
    """
    w = initial_w.copy()
    threshold = 1e-6
    loss = mle_loss(y, tx, w)
    for t in range(max_iters):
        gradient = grad(y, tx, w)
        w -= gamma * gradient
        loss = mle_loss(y, tx, w)

    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    running gradient descent for max_iters iterations (at most)
    return weights, loss
    """
    w = initial_w.copy()
    # threshold = 1e-6
    for t in range(max_iters):
        # loss = mle_loss(y, tx, w) + 0.5 * lambda_ * np.sum(w.T.dot(w))
        gradient = grad(y, tx, w) + 2.0 * lambda_ * w
        # print(y, tx, w, grad(y, tx, w))
        w -= gamma * gradient

    loss = mle_loss(y, tx, w)
    return w, loss
