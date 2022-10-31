import numpy as np
import csv
from math import sqrt

def load_data(train_path = "../train.csv", test_path = "../test.csv"):
    """load the training and testing data and labels from the csv files ."""
    max_cols = np.loadtxt(train_path, delimiter=",", skiprows=1, unpack=True, dtype = str, max_rows = 1).shape[0]
    n_cols = tuple(i for i in range(2,max_cols,1)) #creating a tuple for the number of colums to be used in loadtxt
    y_tr = np.loadtxt(train_path, delimiter=",", skiprows=1, unpack=True, dtype = str, usecols=(1))
    x_tr = np.loadtxt(train_path, delimiter=",", skiprows=1, unpack=True, usecols=n_cols)

    y_te = np.loadtxt(test_path, delimiter=",", skiprows=1, unpack=True, dtype = str, usecols=(1))
    x_te = np.loadtxt(test_path, delimiter=",", skiprows=1, unpack=True,  usecols=n_cols)
    id_te =  np.loadtxt(test_path, delimiter=",", skiprows=1, unpack=True,  usecols=(0))

    x_tr = x_tr.T
    x_te = x_te.T
    return x_tr, y_tr, x_te, y_te, id_te

def load_data_(train_path = "../train.csv", test_path = "../test.csv"):
    """load data."""
    max_cols = np.loadtxt(train_path, delimiter=",", skiprows=1, unpack=True, dtype = str, max_rows = 1).shape[0]
    n_cols = tuple(i for i in range(2,max_cols,1)) #creating a tuple for the number of colums to be used in loadtxt
    y_tr = np.loadtxt(train_path, delimiter=",", skiprows=1, unpack=True, dtype = str, usecols=(1))
    x_tr = np.loadtxt(train_path, delimiter=",", skiprows=1, unpack=True, usecols=n_cols)

    #y_te = np.loadtxt(test_path, delimiter=",", skiprows=1, unpack=True, dtype = str, usecols=(1))
    #x_te = np.loadtxt(test_path, delimiter=",", skiprows=1, unpack=True,  usecols=n_cols)
    #id_te =  np.loadtxt(test_path, delimiter=",", skiprows=1, unpack=True,  usecols=(0))

    x_tr = x_tr.T
    #x_te = x_te.T
    return x_tr, y_tr 

def standardize(x):
    """Standardize (normalize) the original data set to have and mean of 0 and std of 1."""
    mean_x = np.mean(x, axis = 0)
    x = x - mean_x
    std_x = np.std(x, axis = 0)
    x = x / std_x
    return x

def build_model_data(x, y):
    """Form (y,tX) to get regression data in matrix form."""
    num_samples = len(y)
    tx = np.c_[np.ones(num_samples), x]
    return tx, y

def enumerate_labels(y):  #s = 1, b = 0
    """ converting labels from s & b to zero and one"""
    yb = np.ones(len(y))
    yb[np.where(y == "b")] = 0
    return yb


def generate_w(input_shape, seed = 42):
    """generating random starting weights with a fixed seed"""
    n = input_shape[0]
    std = sqrt(2.0 / n)
    np.random.seed(seed)
    w = np.random.randn(input_shape[1])
    w = w * std
    return w

def log_scale(x):
    "Transforming the data into logscale, please note that we shift the data by a constant value first to be all positive"
    x = x + abs(np.min(x)) +0.00000001
    return np.log(x)

def PCA(x_tr):
    z = np.dot(x_tr.T, x_tr)
    eigenvalues, eigenvectors = np.linalg.eig(z)
    D = np.diag(eigenvalues)
    P = eigenvectors
    z_new = np.dot(z, P)
    x_new = np.dot(x_tr, P[:29].T)
    return x_new

def replace_missing(x):
    """replacing the missing data that is -999 with the mean of each column"""""
    mean_x = np.mean(x>=-999, axis = 0)
    std_x = np.std(x>=-999, axis = 0)
    inds = np.where(x == -999 )
    x[inds] = np.take(np.random.normal(mean_x, std_x), inds[1])
    return x

def remove_outliers(x, y, max_tol = 1.2, min_tol = 1.2):
    """replacing the outliers of datapoints with a maximum and minmum boundary"""
    inds = np.array([i for i in range(x.shape[1]) if (i != 4) and (i != 6) and (i != 28) and (i != 27) and (i!=12) ])
    q75,q25 = np.percentile(x[:,  inds], [75,25], axis = 0)
    intr_qr = q75-q25
    maxn = q75+(max_tol*intr_qr)
    min = q25-(min_tol*intr_qr)
    z = np.where(x[:,  inds]>min, x[:,  inds], min)
    z = np.where(z<maxn, z, maxn)
    x[: , inds] = z
    return x, y

def remove_outliers_(x, y, max_tol = 1.2, min_tol = 1.2):
    """replacing the outliers of datapoints with a maximum and minmum boundary"""
    #inds = np.array([i for i in range(x.shape[1]) if (i != 4) and (i != 6) and (i != 28) and (i != 27) and (i!=12) ])
    #q75,q25 = np.percentile(x[:,  inds], [75,25], axis = 0)
    q75,q25 = np.percentile(x, [75,25], axis = 0)
    intr_qr = q75-q25
    maxn = q75+(max_tol*intr_qr)
    min = q25-(min_tol*intr_qr)
    z = np.where(x>min, x, min)
    z = np.where(z<maxn, z, maxn)
    #x[: , inds] = z
    return z, y

def build_poly(x, degree):
    phi_x = np.zeros((len(x), degree*x.shape[1]))
    for b in range(degree):
        phi_x[:, (b)*x.shape[1]: (b+1)*x.shape[1]] = (x ** (b+1))
    return phi_x

def preprocess_data_nolog(train_path = "../train.csv", test_path = "../test.csv"):
    x_tr, y_tr, x_te, y_te, id_te = load_data(train_path, test_path)
    x_tr = replace_missing(x_tr)
    x_te = replace_missing(x_te)
    x_tr, y_tr = remove_outliers(x_tr, y_tr)
    x_tr = standardize(x_tr)
    # x_tr = log_scale(x_tr)
    x_te = standardize(x_te)
    # x_te = log_scale(x_te)
    #x_tr = PCA(x_tr)
    #x_te = PCA(x_te)
    x_tr, y_tr = build_model_data(x_tr, y_tr)
    x_te, y_te = build_model_data(x_te, y_te)
    y_tr = enumerate_labels(y_tr)
    return x_tr, y_tr, x_te, id_te

def preprocess_data(train_path = "../train.csv", test_path = "../test.csv"):
    """applying all the preprocessing methods"""
    x_tr, y_tr, x_te, y_te, id_te = load_data(train_path, test_path)
    x_tr = replace_missing(x_tr)
    x_te = replace_missing(x_te)
    x_tr, y_tr = remove_outliers(x_tr, y_tr)
    x_te, y_te = remove_outliers(x_te, y_te)
    x_tr = standardize(x_tr)
    x_tr = log_scale(x_tr)
    x_te = standardize(x_te)
    x_te = log_scale(x_te)
    x_tr, y_tr = build_model_data(x_tr, y_tr)
    x_te, y_te = build_model_data(x_te, y_te)
    y_tr = enumerate_labels(y_tr)
    return x_tr, y_tr, x_te, id_te

def preprocess_data_final(train_path = "../train.csv", test_path = "../test.csv"):
    """applying all the preprocessing methods"""
    x_tr, y_tr, x_te, y_te, id_te = load_data(train_path, test_path)
    x_tr, y_tr = remove_outliers_(x_tr, y_tr)
    x_te, y_te = remove_outliers_(x_te, y_te)
    x_tr = standardize(x_tr)
    x_tr = log_scale(x_tr)
    x_te = standardize(x_te)
    x_te = log_scale(x_te)
    x_tr, y_tr = build_model_data(x_tr, y_tr)
    x_te, y_te = build_model_data(x_te, y_te)
    y_tr = enumerate_labels(y_tr)
    return x_tr, y_tr, x_te, id_te

def split_data(x, y, ratio, seed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)
    # generate random indices
    num_row = len(y)
    indices = np.random.permutation(num_row)
    index_split = int(np.floor(ratio * num_row))
    index_tr = indices[: index_split]
    index_v = indices[index_split:]
    # create split
    x_tr = x[index_tr]
    x_v = x[index_v]
    y_tr = y[index_tr]
    y_v = y[index_v]
    return x_tr, x_v, y_tr, y_v

def postprocess_preds(y):
    yb = np.ones(len(y))
    yb[np.where(y == 0)] = -1
    return yb


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, "w") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})

def get_accuracy(grnd_truth, pred):
        """calculating the accuracy from predictions"""
        e = grnd_truth - pred
        acc = 1 - np.mean(np.abs(e))
        return acc

def predict_logistic(input, w):
    pred = input.dot(w) > 0
    return pred

def predict_mse(input, w):
    pred = input.dot(w)>0.5
    return pred


def test(id, x, w):
    p =  np.dot(x, w)
    p = p.round()
    p = postprocess_preds(p)
    return id, p

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