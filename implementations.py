from helpers import *
import numpy as np


def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1/2*np.mean(e**2)

def compute_loss(y, tx, w, l=0):
    # l = 0 for MSE
    """Calculate the loss using either MSE or MAE.
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2,). The vector of model parameters.
    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    if l == 0:
      #compute loss by MSE
      e = y - np.dot(tx, w)
      N = y.shape[0]
      L = 1/(2*N) * e.T * e
      return L.sum()
    else:
      #compute loss by MAE
      e =  np.dot(tx, w)
      e = y - e
      N = y.shape[0]
      L = 1/(2*N) * e
      return L.sum()



def compute_gradient(y, tx, w):
    """Computes the gradient at w.
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.
    Returns:
        An numpy array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of GD
    """
    w = initial_w
    g, e = compute_gradient(y, tx, w)
    loss = calculate_mse(e)
    for n_iter in range(max_iters):
        g, _ = compute_gradient(y, tx, w)
        w = w - gamma*g
        err = y - tx.dot(w)
        loss = calculate_mse(err)
        #print("GD iter. {bi}/{ti}: loss={l}".format(bi=n_iter, ti=max_iters - 1, l=loss))

    return w, loss

def mean_squared_error_gd_es(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm. this function implements early stopping to return the best weights not the last ones
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of GD
    """
    w = initial_w
    g, e = compute_gradient(y, tx, w)
    loss = calculate_mse(e)
    best_loss = 10e25
    best_w = initial_w
    for n_iter in range(max_iters):
        g, _ = compute_gradient(y, tx, w)
        w = w - gamma*g
        err = y - tx.dot(w)
        loss = calculate_mse(err)
        best_w = (w if loss < best_loss else best_w)
        best_loss = (loss if loss < best_loss else best_loss)
        
        #print("GD iter. {bi}/{ti}: loss={l}".format(bi=n_iter, ti=max_iters - 1, l=loss))

    return best_w, best_loss

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient at w from just few examples n and their corresponding y_n labels.
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.

    Returns:
        A numpy array of shape (2, ) (same shape as w), containing the stochastic gradient of the loss at w.
    """
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err

def mean_squared_error_sgd(y, tx, initial_w, max_iters = 1, gamma = 0.01, batch_size = 1, shuffle = False):
    """The Stochastic Gradient Descent algorithm (SGD).

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of SGD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of SGD
    """
    w = initial_w
    g, e = compute_stoch_gradient(y, tx, w)
    loss = compute_loss(y, tx, w)
    for n_iter in range(int(max_iters)):
        for yi, txi in batch_iter(y, tx, batch_size, num_batches=1, shuffle = shuffle):
            g, e = compute_stoch_gradient(yi, txi, w)
            w = w - gamma*g
            loss = compute_loss(yi, txi, w)
        #print("SGD iter. {bi}/{ti}: loss={l}".format(bi=n_iter, ti=max_iters - 1, l=loss))
    return w, loss

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
    """the gradient of loss for logisteic regression"""
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

def ridge_regression(y, tx, lambda_):
    """implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar, mean square error loss for the final weights
    """
    d = tx.shape[1]
    n = tx.shape[0]
    lambda_prime = 2 * n * lambda_
    a = (tx.T @ tx) + (lambda_prime * np.identity(d))
    b = tx.T @ y
    w_ridge = np.linalg.solve(a, b)
    e = y - (tx @ w_ridge)
    mse = np.mean ( e ** 2) / 2
    return w_ridge, mse

def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar, mean square error loss for the final weights

    """
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    e = y - (tx @ w)
    mse = np.mean ( e ** 2) / 2
    return w, mse
