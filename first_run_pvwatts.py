from importers import pvwatts
from predictors.svr_model import SVRModel
from predictors.arima_model import ARIMAModel
from evaluation.error_terms import rmse, nrmse
import pandas as pd
from datetime import datetime
import math

# prepare
data = pvwatts.load_city_from_list('data/pvwatts/station_export.csv', 'Berlin')
df = pd.DataFrame(columns=['features', 'arima_jan', 'arima_april', 'arima_july', 'arima_oct', 'svr_jan', 'svr_april', 'svr_july', 'svr_oct'])
features = [['tamb'], ['wspd'], ['tamb', 'wspd']]
datestrings = [
    ['jan', ['20190104', '20190131', '20190201', '20190202']],
    ['april', ['20190103', '20190430', '20190501', '20190502']],
    ['july', ['20190704', '20190731', '20190801', '20190802']],
    ['oct', ['20191004', '20191031', '20191101', '20191102']],
]

# run
print('--------------------------------')
print('run started at')
print(datetime.now())
print('--------------------------------')
print()

# one time without exogenous variables (arima only)
columns = {'features': 'none', 'svr_jan': math.nan, 'svr_april': math.nan, 'svr_july': math.nan, 'svr_oct': math.nan}
for month, dates in datestrings:
    training_start, training_end, testing_start, testing_end = dates
    training = data[training_start:training_end]
    testing = data[testing_start:testing_end]

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
        training = data[training_start:training_end]
        testing = data[testing_start:testing_end]

        # arima
        print(f'fitting arima model for {month} with features {filter} at {datetime.now()}...')
        arima = ARIMAModel(scaling=True)
        arima.fit_auto(training, p=(1,3), q=(1,3), P=(1,3), Q=(1,3), d=0, D=0, filter=filter)
        print(f'fitted arima model successfully with order={arima.model.order}, seasonal_order={arima.model.seasonal_order} at {datetime.now()}')
        prediction = arima.predict(testing_data=testing)
        error = rmse(testing.power, prediction.power)
        print(f'RMSE: {error}')
        print()
        key = f'arima_{month}'
        columns[key] = error

        # svr
        print(f'fitting svr model for {month} with features {filter}...')
        svr = SVRModel(data, scaling=True)
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
path = 'out/pvwatts/first.csv'
print(f'saving dataframe to {path}')
df.to_csv(path)
print(f'run finished at {datetime.now()}')
print('--------------------------------')
