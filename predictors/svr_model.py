import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

class SVRModel:
    """
    ---------------------------
    ###### SVR Predictor ######
    ---------------------------

    Support Vector Regression Model based on
    the Support Vector Machine implementation of Scikit-Learn

    base_data: DataFrame. One whole dataset. Used to adjust the scaling parameters.
    scaling: Boolean. Whether the data should be scaled before using the SVR algorithm. default = true
    """
    def __init__(self, base_data, scaling=True):
        self._base_data = base_data
        self._scalers = {}
        self._filter = list(base_data.keys())
        self._fit_scalers()

        self.training_original = None
        self.training_scaled = None
        self.testing_original = None
        self.testing_scaled = None
        self.scaling = scaling
        self.prediction = None

    def _fit_scalers(self):
        for key in self._base_data:
            self._scalers[key] = StandardScaler()
            self._scalers[key].fit(self._base_data[key].values.reshape(-1, 1))

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
        if filter:
            filter.append('power')
            self._filter = filter

        self.training_original = data.filter(self._filter)

        if self.scaling:
            self.training_scaled = pd.DataFrame(index=self.training_original.index)
            for key in self.training_original:
                scaler = self._scalers[key]
                values = self.training_original[key].values.reshape(-1, 1)
                self.training_scaled[key] = scaler.transform(values)
            training_data = self.training_scaled
        else:
            training_data = self.training_original

        self.svr = SVR(kernel=kernel, C=C, gamma=gamma, epsilon=epsilon)
        self.svr.fit(training_data.drop('power', axis=1).values, training_data['power'])

    def predict(self, data):
        """
        Make a prediction based on the features specified

        data: DataFrame. Dataset including all test features
        """
        self.testing_original = data.filter(self._filter)

        if self.scaling:
            self.testing_scaled = pd.DataFrame(index=self.testing_original.index)
            for key in self.testing_original:
                scaler = self._scalers[key]
                values = self.testing_original[key].values.reshape(-1, 1)
                self.testing_scaled[key] = scaler.transform(values)
            testing_data = self.testing_scaled
        else:
            testing_data = self.testing_original

        prediction = self.svr.predict(testing_data.drop('power', axis=1))
        if self.scaling:
            self.prediction = self._scalers['power'].inverse_transform(prediction)
        else:
            self.prediction = prediction
        return self.prediction
