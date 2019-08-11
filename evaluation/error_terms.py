from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score

def mse(y_true, y_pred):
    """
    calculates the mean squared error for two series using
    the mean_squared_error implementation of sklearn

    y_true: series. Expected values
    y_pred: series. Predicted values
    """
    return round(mean_squared_error(y_true, y_pred), 2)

def rmse(y_true, y_pred):
    """
    calculates the root mean squared error for two series using
    the mean_squared_error implementation of sklearn

    y_true: series. Expected values
    y_pred: series. Predicted values
    """
    result = mean_squared_error(y_true, y_pred)
    return round(sqrt(result), 2)

def nrmse(y_true, y_pred):
    """
    calculates the normalized root mean squared error for two
    series using the mean_squared_error implementation of sklearn

    y_true: series. Expected values
    y_pred: series. Predicted values
    """
    result = sqrt(mean_squared_error(y_true, y_pred))
    result = round(result / y_true.mean(), 2)
    if result > 3 or result < -3: raise RuntimeError(f'error nrmse {result} out of bounds')
    return result

def r2(y_true, y_pred):
    """
    calculates the r2 score or coefficient of determination for two
    series using the r2_score implementation of sklearn

    y_true: series. Expected values
    y_pred: series. Predicted values
    """
    result = round(r2_score(y_true, y_pred), 2)
    if result > 3 or result < -3: raise RuntimeError(f'error r2 {result} out of bounds')
    return result
