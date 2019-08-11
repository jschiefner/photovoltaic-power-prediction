from importers import pvwatts
from predictors.svr_model import SVRModel
from evaluation.error_terms import mse, rmse, nrmse, r2
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters; register_matplotlib_converters() # fix pandas future warnings

# load data
data = pvwatts.load_city_from_list('data/pvwatts/stations_list.csv', 'Berlin')

# prepare testing and training data
training = data['20190604':'20190630'] # use 4 weeks in june for training
testing = data['20190701':'20190702'] # use 2 days in july for testing

# prepare model
model = SVRModel(base_data=data, scaling=True)
model.fit(training, filter=['tamb'])
model.predict(testing)

# evaluate model
print(f'MSE: {mse(testing.power, model.prediction.power)}')
print(f'RMSE: {rmse(testing.power, model.prediction.power)}')
print(f'nRMSE: {nrmse(testing.power, model.prediction.power)}')
print(f'R2: {r2(testing.power, model.prediction.power)}')

# plot testing features
plt.plot(testing.tamb, color='red')
plt.show()

# plot prediction and expected power output
plt.plot(testing.power, color='red')
plt.plot(model.prediction.power, color='orange')
plt.show()
