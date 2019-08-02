import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import warnings

def validate_init(base_data, scaling):
    if base_data is None and scaling:
        raise TypeError('if scaling is activated, a dataframe containing sample data must be provided for fitting the scaler')

def fit_scaler(data):
    scaler = StandardScaler()
    scaler.fit(data)
    return scaler

class SVRModel:
    """
    ---------------------------
    ###### SVR Predictor ######
    ---------------------------

    Support Vector Regression Model based on
    the Support Vector Machine implementation of Scikit-Learn

    base_data: DataFrame. One whole dataset. This is only used to adjust the scaling parameters (optional).
                          Must be specified if scaling is activated.
    scaling: Boolean. Whether the data should be scaled before using the SVR algorithm (optional). default = true
    """
    def __init__(self, base_data=None, scaling=True):
        validate_init(base_data, scaling)
        self._base_data = base_data
        self._scaler = None
        self._filter = None

        self.model = None
        self.scaling = scaling
        self.prediction = None

    def fit(self, data, filter=None, kernel='rbf', C=1e3, gamma=0.1, epsilon=0.1):
        """
        Fit the model with a dataset.

        data: DataFrame. Dataset including the training data
        filter: list. Set the features that you want to use for the regression (optional)

        kernel: String. Set the kernel used by the SVR algorithm (optional). default = 'rbf'
        C: float. Penalty Parameter (optional). default = 1e3
        gamma: float. Kernel coefficient (optional). default = 0.1
        epsilon: float. Epsilon-tube distance (optional). default = 0.1
        """
        if filter: filter.append('power')
        else: filter = list(data.keys())
        self._filter = filter
        data = data.filter(filter)

        if self.scaling:
            self._scaler = fit_scaler(self._base_data.filter(filter))
            scaled_values = self._scaler.transform(data)
            data_frame = pd.DataFrame(scaled_values, index=data.index, columns=data.columns)
        else:
            data_frame = data

        self.model = SVR(kernel=kernel, C=C, gamma=gamma, epsilon=epsilon)
        self.model.fit(data_frame.drop('power', axis=1), data_frame.power)

    def predict(self, data):
        """
        Make a prediction based on the features specified. Returns the predicted values as a dataframe
        together with the testing data. This dataframe can also be accessed with 'prediction'.

        data: DataFrame. Dataset including all test features
        """
        data = data.filter(self._filter)

        if self.scaling:
            scaled_values = self._scaler.transform(data)
            data_frame = pd.DataFrame(scaled_values, index=data.index, columns=data.columns)
        else:
            data_frame = data

        prediction = self.model.predict(data_frame.drop('power', axis=1))
        data_frame['power'] = prediction
        if self.scaling:
            inversed = self._scaler.inverse_transform(data_frame)
            data_frame = pd.DataFrame(inversed, index=data_frame.index, columns=data_frame.columns)
        self.prediction = data_frame
        return data_frame
