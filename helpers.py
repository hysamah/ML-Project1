import numpy as np
import csv
from math import sqrt

def load_data(train_path, test_path):
    """load data."""
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

def standardize(x):
    """Standardize the original data set."""
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
    lables= np.unique(y)
    yb = np.array([i for j in y for i in range(len(lables)) if j == lables[i]])
    #yb = np.ones(len(y))
    #yb[np.where(y == "b")] = -1
    return yb 

def remove_outliers(x, y):
    q75,q25 = np.percentile(x, [75,25], axis = 0)
    intr_qr = q75-q25
    maxn = q75+(1.5*intr_qr)
    min = q25-(1.5*intr_qr)
    z = np.where(x>min, x, min)
    z = np.where(z<maxn, z, maxn)
    # z = np.where(x>min, x, np.nan)
    # z = np.where(z<maxn, z, np.nan)
    y = y[~np.isnan(z).any(axis = 1)]
    z = z[~np.isnan(z).any(axis = 1)]
    return z, y

def generate_w(input_shape, seed = 1):
    n = input_shape[0]
    std = sqrt(2.0 / n)
    np.random.seed(seed)
    w = np.random.randn(input_shape[1])
    w = w * std
    #w = np.zeros(input_shape[1])
    return w

def log_scale(x):
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

def preprocess_data(train_path = "../train.csv", test_path = "../test.csv"):
    x_tr, y_tr, x_te, y_te, id_te = load_data(train_path, test_path)
    x_tr, y_tr = remove_outliers(x_tr, y_tr)
    x_tr = standardize(x_tr)
    x_tr = log_scale(x_tr)
    x_te = standardize(x_te)
    x_te = log_scale(x_te)
    #x_tr = PCA(x_tr)
    #x_te = PCA(x_te)
    x_tr, y_tr = build_model_data(x_tr, y_tr)
    x_te, y_te = build_model_data(x_te, y_te)
    y_tr = enumerate_labels(y_tr)
    return x_tr, y_tr, x_te, id_te

def preprocess_data_logscale(train_path = "../train.csv", test_path = "../test.csv"):
    x_tr, y_tr, x_te, y_te, id_te = load_data(train_path, test_path)
    x_tr, y_tr = remove_outliers(x_tr, y_tr)
    x_tr = standardize(x_tr)
    x_tr = log_scale(x_tr)
    x_te = standardize(x_te)
    x_te = log_scale(x_te)
    #x_tr = PCA(x_tr)
    #x_te = PCA(x_te)
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

def get_accuracy(grnd_truth, input, weights):
        e =  np.dot(input, weights)
        e = grnd_truth - e
        N = grnd_truth.shape[0]
        e = e.round()
        acc = 1-np.sum(abs(e))/N
        return acc

def test(id, x, w):
    p =  np.dot(x, w)
    p = p.round()
    p = postprocess_preds(p)
    return id, p
