from math import sqrt
from sklearn.metrics import mean_squared_error

def mse(y_true, y_pred):
    """
    calculates the mean squared error for two series using
    the mean_squared_error implementation of sklearn

    y_true: series. Expected values
    y_pred: series. Predicted values
    """
    return mean_squared_error(y_true, y_pred)

def rmse(y_true, y_pred):
    """
    calculates the root mean squared error for two series using
    the mean_squared_error implementation of sklearn

    y_true: series. Expected values
    y_pred: series. Predicted values
    """
    output = mse(y_true, y_pred)
    return sqrt(output)

def nrmse(y_true, y_pred):
    """
    calculates the normalized root mean squared error for two
    series using the mean_squared_error implementation of sklearn

    y_true: series. Expected values
    y_pred: series. Predicted values
    """
    output = rmse(y_true, y_pred)
    return output / y_true.mean()
