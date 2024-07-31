import math
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error


def masked_mape(truth, pred, null_val):
    pred[np.abs(pred) < 1e-4] = 0
    if np.isnan(null_val):
        mask = ~np.isnan(truth)
    return mean_absolute_percentage_error(truth, pred)


def masked_mae(truth, pred, null_val):
    return mean_absolute_error(truth, pred)


def masked_rmse(truth, pred, null_val):
    return math.sqrt(mean_squared_error(truth, pred))
