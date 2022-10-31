#This file contains the steps to reproduce the current best results (from the least_squares function)
import numpy as np
import matplotlib.pyplot as plt
from implementations import *
from helpers import *

predictions_out_file  = "best_least_squares_test_logscale.csv"
x_tr, y_tr, x_te, id_te = preprocess_data_final("train.csv", "test.csv") #preprocess intput data from the training and test sets

#the preprocessing currently includes removing outliers, standarization, and logscaling the data
w, loss = least_squares(y_tr, x_tr)
pred = predict_mse(x_tr, w)
print("The training accuracy is:", get_accuracy(y_tr, pred))
id, preds = test(id_te, x_te, w)
create_csv_submission(id, preds, predictions_out_file)

print("Testing predictions can be found in ", predictions_out_file)