
import numpy as np

def get_accuracy(grnd_truth, input, weights):
        e =  np.dot(input, weights)
        e = grnd_truth - e
        N = grnd_truth.shape[0]
        e = e.round()
        acc = 1-np.sum(abs(e))/N
        return acc

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

def generate_w(input_shape):
    w = np.zeros(input_shape[1])
    return w

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
        print("GD iter. {bi}/{ti}: loss={l}".format(bi=n_iter, ti=max_iters - 1, l=loss))

    return w, loss

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

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

def mean_squared_error_sgd(y, tx, initial_w, max_iters = 1, gamma = 0.01, batch_size = 1):
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
        for yi, txi in batch_iter(y, tx, batch_size, num_batches=1, shuffle= False):
            g, e = compute_stoch_gradient(yi, txi, w)
            w = w - gamma*g
            loss = compute_loss(yi, txi, w)
        print("SGD iter. {bi}/{ti}: loss={l}".format(bi=n_iter, ti=max_iters - 1, l=loss))
    return w, loss

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
    loss = mle_loss(y, tx, w)
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
def ridge_regression(y, tx, lambda_):
    """implement ridge regression.
    
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.
    
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.

    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 0)
    array([ 0.21212121, -0.12121212])
    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 1)
    array([0.03947092, 0.00319628])
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
        mse: scalar.

    >>> least_squares(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]))
    (array([ 0.21212121, -0.12121212]), 8.666684749742561e-33)
    """
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    e = y - (tx @ w)
    mse = np.mean ( e ** 2) / 2
    return w, mse
