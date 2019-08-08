from importers import uq
from predictors.svr_model import SVRModel
from evaluation.error_terms import mse, rmse, nrmse
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters; register_matplotlib_converters() # fix pandas future warnings

# load data
data = uq.load('power.csv', 'weather.csv')

# prepare testing and training data
# make sure the date ranges fit your data
training = data['20140104':'20140131'] # use 4 weeks in january for training
testing = data['20140201':'20140202'] # use 2 days in february for testing

# prepare model
model = SVRModel(base_data=data, scaling=True)
model.fit(training, filter=['airtemp', 'humidity'])
model.predict(testing)

# evaluate model
print(f'MSE: {mse(testing.power, model.prediction.power)}')
print(f'RMSE: {rmse(testing.power, model.prediction.power)}')
print(f'nRMSE: {nrmse(testing.power, model.prediction.power)}')

# plot testing features
plt.plot(testing.airtemp, color='red')
plt.plot(testing.humidity, color='blue')
plt.show()

# plot prediction and expected power output
plt.plot(testing.power, color='red')
plt.plot(model.prediction.power, color='orange')
plt.show()
