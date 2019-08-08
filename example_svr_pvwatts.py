from importers import pvwatts
from predictors.svr_model import SVRModel
from evaluation.error_terms import mse, rmse, nrmse
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters; register_matplotlib_converters() # fix pandas future warnings

# load data
data = pw.load(lat=52.470001220703125, lon=13.399999618530273) # station in berlin

# prepare testing and training data
training = data['20190604':'20190630'] # use 4 weeks in january for training
testing = data['20190701':'20190702'] # use 2 days in february for testing

# prepare model
model = SVRModel(base_data=data, scaling=True)
model.fit(training, filter=['tamb'])
model.predict(testing)

# evaluate model
print(f'MSE: {mse(testing.power, model.prediction.power)}')
print(f'RMSE: {rmse(testing.power, model.prediction.power)}')
print(f'nRMSE: {nrmse(testing.power, model.prediction.power)}')

# plot testing features
plt.plot(testing.tamb, color='red')
plt.show()

# plot prediction and expected power output
plt.plot(testing.power, color='red')
plt.plot(model.prediction.power, color='orange')
plt.show()
