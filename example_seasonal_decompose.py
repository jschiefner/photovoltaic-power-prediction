from importers import uq
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

data = uq.load('data/uq/power/car_park_1/2014.csv', 'data/uq/weather/2014.csv')
week = data['20140101':'20140107']
decomposed = seasonal_decompose(week.power)

# access decomposed components with:
# decomposed.observed
# decomposed.trend
# decomposed.seasonal
# decomposed.resid

decomposed.plot()
