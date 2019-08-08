from importers import uq
from predictors.svr_model import SVRModel
from predictors.arima_model import ARIMAModel
from evaluation.error_terms import rmse, nrmse
import pandas as pd
from datetime import datetime
import math

# prepare

st_lucia_2014 = uq.load('data/uq/advanced_engineering_building/2014/power.csv', 'data/uq/advanced_engineering_building/2014/weather.csv')
st_lucia_2015 = uq.load('data/uq/advanced_engineering_building/2015/power.csv', 'data/uq/advanced_engineering_building/2015/weather.csv')

df = pd.DataFrame(columns=['features', 'arima_jan', 'arima_apr', 'arima_jul', 'arima_oct', 'svr_jan', 'svr_apr', 'svr_jul', 'svr_oct'])
features = [['airtemp'], ['humidity'], ['airtemp', 'humidity'], ['airtemp', 'humidity', 'windspeed'], ['airtemp', 'humidity', 'windspeed', 'winddirection']]
datestrings = [
    ['jan', ['20150104', '20150131', '20150201', '20150202']],
    ['apr', ['20150103', '20150430', '20150501', '20150502']],
    ['jul', ['20150704', '20150731', '20150801', '20150802']],
    ['oct', ['20151004', '20151031', '20151101', '20151102']],
]

# run
print('--------------------------------')
print('run started at')
print(datetime.now())
print('--------------------------------')
print()

# one time without exogenous variables (arima only)
columns = {'features': 'none', 'svr_jan': math.nan, 'svr_apr': math.nan, 'svr_jul': math.nan, 'svr_oct': math.nan}
for month, dates in datestrings:
    training_start, training_end, testing_start, testing_end = dates
    training = st_lucia_2015[training_start:training_end]
    testing = st_lucia_2015[testing_start:testing_end]

    # arima
    print(f'fitting arima model without exogenous variables for {month}...')
    arima = ARIMAModel(scaling=True)
    arima.fit_auto(training, p=(1,3), q=(1,3), P=(1,3), Q=(1,3), d=0, D=0, use_exogenous=False)
    print(f'fitted arima model successfully with order={arima.model.order}, seasonal_order={arima.model.seasonal_order}')
    prediction = arima.predict(hours=48)
    error = rmse(testing.power, prediction.power)
    print(f'RMSE: {error}')
    print()
    key = f'arima_{month}'
    columns[key] = error

print('--------------------------------')
print(f'appending row without features to data frame')
df = df.append(columns, ignore_index=True)
print('--------------------------------')
print()

for filter in features:
    columns = {'features': str(filter)}
    for month, dates in datestrings:
        training_start, training_end, testing_start, testing_end = dates
        training = st_lucia_2015[training_start:training_end]
        testing = st_lucia_2015[testing_start:testing_end]

        # arima
        print(f'fitting arima model for {month} with features {filter} at {datetime.now()}...')
        arima = ARIMAModel(scaling=True)
        arima.fit_auto(training, p=(1,3), q=(1,3), P=(1,3), Q=(1,3), d=0, D=0, filter=filter)
        print(f'fitted arima model successfully with order={arima.model.order}, seasonal_order={arima.model.seasonal_order}')
        prediction = arima.predict(testing_data=testing)
        error = rmse(testing.power, prediction.power)
        print(f'RMSE: {error}')
        print()
        key = f'arima_{month}'
        columns[key] = error

        # svr
        print(f'fitting svr model for {month} with features {filter}...')
        svr = SVRModel(st_lucia_2014, scaling=True)
        svr.fit(training, filter=filter)
        print('fitted svr model successfully')
        prediction = svr.predict(testing)
        error = rmse(testing.power, prediction.power)
        print(f'RMSE: {error}')
        print()
        key = f'svr_{month}'
        columns[key] = error

    print('--------------------------------')
    print(f'appending row with features {filter} to data frame')
    df = df.append(columns, ignore_index=True)
    print('--------------------------------')
    print()

# average
df.loc['average'] = df.mean().round(2)

print('--------------------------------')
path = 'out/pvwatts/st_lucia_2015.csv'
print(f'saving dataframe to {path}')
df.to_csv(path)
print(f'run finished at {datetime.now()}')
print('--------------------------------')
