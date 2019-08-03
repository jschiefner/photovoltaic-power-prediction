from pmdarima import auto_arima, ARIMA
import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings

def validate_fit_params(order, seasonal_order, filter, use_exogenous):
    if not (isinstance(order, tuple) and len(order) == 3):
        raise TypeError('order has to be a tuple with three values (p,d,q)')
    if (seasonal_order is not None) and not (isinstance(seasonal_order, tuple) and len(seasonal_order) == 4):
        raise TypeError('seasonal_order has to be a tuple with four values (P,D,Q,s)')
    warn_on_filter_exogenous(filter, use_exogenous)

def validate_fit_auto_params(hyperparameters, filter, use_exogenous):
    for index, argument in enumerate(hyperparameters):
        if not (isinstance(argument, tuple) and len(argument) == 2):
            argument_name = ['p', 'q', 'P', 'Q'][index]
            raise TypeError(f'{argument_name} needs to be a two-dimensional tuple specifying the minumum and maximum value')
    warn_on_filter_exogenous(filter, use_exogenous)

def warn_on_filter_exogenous(filter, use_exogenous):
    if not use_exogenous and filter:
        warnings.warn('When using no exogenous variables the filter is ignored, since only the power output will be used for prediction')

class ARIMAModel:
    """
    -----------------------------
    ###### ARIMA Predictor ######
    -----------------------------

    ARIMA Model based on the pmdarima library
    """
    def __init__(self, scaling=True):
        self.training_data = None
        self.prediction = None
        self.model = None
        self.use_exogenous = None

        self._scaling = scaling
        self._scaler = StandardScaler()
        self._filter = None

    def fit(self, data, order=None, seasonal_order=None, filter=None, use_exogenous=True):
        """
        Fit the model with a dataset and specif hyperparameters. If you want to let
        pmdarima search for good hyperparameters use the 'fit_auto' method instead.

        data: DataFrame. Dataset including the training data
        order: tuple. Set the (p,d,q) parameters for the model
        seasonal_order: tuple. Set the seasonal order (P,D,Q,s) for the model
        filter: list. Set the features that you want to use for the regression (optional)
                      will be ignored if use_exogenous is False
        use_exogenous: Boolean. Set whether exogenous features from 'data' should
                       be used for fitting and prediction (optional). default = True
        """
        validate_fit_params(order, seasonal_order, filter, use_exogenous)
        self.use_exogenous = use_exogenous
        self.model = ARIMA(order=order, seasonal_order=seasonal_order, with_intercept=False)

        if use_exogenous:
            if filter:
                filter.append('power')
                self._filter = filter
                data = data.filter(self._filter)
            else:
                self._filter = list(data.keys())
            if self._scaling:
                scaled_data = self._scaler.fit_transform(data)
                data = pd.DataFrame(scaled_data, index=data.index, columns=data.columns)
            self.training_data = data
            self.model.fit(data.power, exogenous=data.drop('power', axis=1))
        else:
            data = data.filter(['power'])

            if self._scaling:
                scaled_data = self._scaler.fit_transform(data)
                data = pd.DataFrame(scaled_data, index=data.index, columns=['power'])
            self.training_data = data
            self.model.fit(data.power)


    def fit_auto(self, data, p, q, P, Q, d, D, filter=None, use_exogenous=True):
        """
        Fit the model with a dataset. This method finds suitable hyperparameters in a specified range.
        If you want to fit a model with specific parameters use the 'fit' method instead.

        data: DataFrame. Dataset including the training data
        p: tuple. Set the (start_p, max_p) range for the model
        q: tuple. Set the (start_q, max_q) range for the model
        P: tuple. Set the (start_P, max_P) range for the model
        Q: tuple. Set the (start_Q, max_Q) range for the model
        d: positive int. Amount of differencing. If left out, an algorithm will be used
                         to determine a suitable value (optional)
        D: positive int. Amount of seasonal differencing. If left out, an algorithm will be used
                         to determine a suitable value (optional)
        filter: list. Set the features that you want to use for the regression (optional)
                      will be ignored if use_exogenous is False
        use_exogenous: Boolean. Set whether exogenous features from 'data' should
                       be used for fitting and prediction (optional). default = True
        """
        validate_fit_auto_params([p, q, P, Q], filter, use_exogenous)
        (start_p, max_p), (start_q, max_q), (start_P, max_P), (start_Q, max_Q) = p, q, P, Q
        self.use_exogenous = use_exogenous

        if use_exogenous:
            if filter:
                filter.append('power')
                self._filter = filter
                data = data.filter(self._filter)
            else:
                self._filter = list(data.keys())
            if self._scaling:
                scaled_data = self._scaler.fit_transform(data)
                data = pd.DataFrame(scaled_data, index=data.index, columns=data.columns)
            self.training_data = data
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                self.model = auto_arima(data.power, start_p=start_p, start_q=start_q, max_p=max_p, max_q=max_q,
                                        start_P=start_P, start_Q=start_Q, max_P=max_P, max_Q=max_Q, m=24, d=d, D=D, trace=True,
                                        with_intercept=False, exogenous=data.drop('power', axis=1))
        else:
            data = data.filter(['power'])

            if self._scaling:
                scaled_data = self._scaler.fit_transform(data)
                data = pd.DataFrame(scaled_data, index=data.index, columns=['power'])
            self.training_data = data
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                self.model = auto_arima(data.power, start_p=start_p, max_p=max_p, start_q=start_q, max_q=max_q,
                                        start_P=start_P, max_P=max_P, start_Q=start_Q, max_Q=max_Q,
                                        m=24, d=d, D=D, trace=True, with_intercept=False)

    def predict(self, hours=None, testing_data=None):
        """
        Make a prediction. Either pass in how many hours you want to predict or pass in testing_data
        if the model was fit with exogenous variables.

        hours: int. Amount of predicted values. Parameter is ignored if model was fit with exogenous variables.
                    If not specified, length of prediction will be inferred from training data (optional)
        testing_data: DataFrame. Supply if model was fit with exogenous variables. Hours of forecasting will then be according to
                                 the length of the testing_data. Ignored if model was fit without exogenous variables (optional)
        """
        if self.use_exogenous:
            if testing_data is None: raise TypeError('Model uses exogenous variables so the testing_data parameter is mandatory')
            if hours: warnings.warn("'hours' parameter is ignored. Length of prediction will be inferred from testing_data")
            hours = len(testing_data)
            testing_data = testing_data.filter(self._filter)
            if self._scaling:
                scaled_data = self._scaler.transform(testing_data)
                data_frame = pd.DataFrame(scaled_data, index=testing_data.index, columns=testing_data.columns)
            else:
                data_frame = testing_data
            prediction = self.model.predict(n_periods=hours, exogenous=data_frame.drop('power', axis=1))
            data_frame['power'] = prediction
            if self._scaling:
                inverse_transformed = self._scaler.inverse_transform(data_frame)
                data_frame = pd.DataFrame(inverse_transformed, index=testing_data.index, columns=testing_data.columns)
        else:
            if testing_data is not None: warnings.warn('testing_data is ignored since model was fit without exogenous variables')
            if hours is None: hours = len(self.training_data)
            index_start = self.training_data.index[len(self.training_data)-1] + pd.Timedelta(hours=1)
            data_frame = pd.DataFrame(index=pd.date_range(index_start, periods=hours, freq='H'))
            prediction = self.model.predict(n_periods=hours)
            if self._scaling:
                data_frame['power'] = prediction
                data_upscaled = self._scaler.inverse_transform(data_frame)
                data_frame['power'] = data_upscaled
            else:
                data_frame['power'] = prediction
        data_frame['power'] = data_frame.power.clip(0)
        self.prediction = data_frame
        return data_frame
