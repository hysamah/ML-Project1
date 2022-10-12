import numpy as np
def load_data(train_path, test_path):
    """load data."""
    max_cols = np.loadtxt(train_path, delimiter=",", skiprows=1, unpack=True, dtype = str, max_rows = 1).shape[0]
    n_cols = tuple(i for i in range(2,max_cols,1)) #creating a tuple for the number of colums to be used in loadtxt
    y_tr = np.loadtxt(train_path, delimiter=",", skiprows=1, unpack=True, dtype = str, usecols=(1))
    x_tr = np.loadtxt(train_path, delimiter=",", skiprows=1, unpack=True, usecols=n_cols)
    
    y_te = np.loadtxt(test_path, delimiter=",", skiprows=1, unpack=True, dtype = str, usecols=(1))
    x_te = np.loadtxt(test_path, delimiter=",", skiprows=1, unpack=True,  usecols=n_cols)

    x_tr = x_tr.T
    x_te = x_te.T
    return x_tr, y_tr, x_te, y_te

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

def enumerate_labels(y):
    y = y.view(np.int32)
    return y 

def preprocess_data(train_path = "../train.csv", test_path = "../test.csv"):
    x_tr, y_tr, x_te, y_te = load_data(train_path, test_path)
    x_tr = standardize(x_tr)
    x_te = standardize(x_te)
    x_tr, y_tr = build_model_data(x_tr, y_tr)
    x_te, y_te = build_model_data(x_te, y_te)
    y_tr = enumerate_labels(y_tr)
    y_te = enumerate_labels(y_te)
    return x_tr, y_tr, x_te, y_te
