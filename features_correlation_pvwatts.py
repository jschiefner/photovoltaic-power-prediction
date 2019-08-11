from importers import pvwatts
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters; register_matplotlib_converters() # fix pandas future warnings

json = pvwatts.load_from_json('data/pvwatts/duesseldorf.json', return_json=True)
data = pvwatts.json_to_dataframe(json, keys=['ac', 'poa', 'dn', 'df', 'tcell', 'tamb', 'wspd'])

month = data['20190301':'20190330']

plt.scatter(month.poa, month.power)
plt.show()

plt.scatter(month.dn, month.power)
plt.show()

plt.scatter(month.df, month.power)
plt.show()

plt.scatter(month.tcell, month.power)
plt.show()

plt.scatter(month.tamb, month.power)
plt.show()

plt.scatter(month.wspd, month.power)
plt.show()
