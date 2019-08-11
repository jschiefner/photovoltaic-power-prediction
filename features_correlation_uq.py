from importers import uq
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters; register_matplotlib_converters() # fix pandas future warnings

data = uq.load('data/uq/power/uq_centre/2014.csv', 'data/uq/weather/2014.csv', with_insolation=True)

month = data['20140301':'20140331']

plt.scatter(month.airtemp, month.power)
plt.show()

plt.scatter(month.humidity, month.power)
plt.show()

plt.scatter(month.insolation, month.power)
plt.show()

plt.scatter(month.windspeed, month.power)
plt.show()

plt.scatter(month.winddirection, month.power)
plt.show()
