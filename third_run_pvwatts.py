import warnings
from importers import pvwatts
from predictors.arima_model import ARIMAModel
from predictors.svr_model import SVRModel
from evaluation.error_terms import nrmse, r2
import pandas as pd
from datetime import datetime
import numpy as np
import itertools

# prepare
locations = pvwatts.bulk_load_from_list('data/pvwatts/stations_list.csv', range=(0, 50))
length = len(locations)
length_monthly = length * 12

filter = ['tamb', 'wspd']
order = (2,0,1)
seasonal_order = (2,0,1,24)
datestrings = [
    ['jan', ['20190104', '20190131', '20190201', '20190202']],
    ['feb', ['20190201', '20190228', '20190301', '20190302']],
    ['mar', ['20190304', '20190331', '20190401', '20190402']],
    ['apr', ['20190403', '20190430', '20190501', '20190502']],
    ['may', ['20190504', '20190531', '20190601', '20190602']],
    ['jun', ['20190603', '20190630', '20190701', '20190702']],
    ['jul', ['20190704', '20190731', '20190801', '20190802']],
    ['aug', ['20190804', '20190831', '20190901', '20190902']],
    ['sep', ['20190903', '20190930', '20191001', '20191002']],
    ['oct', ['20191004', '20191031', '20191101', '20191102']],
    ['nov', ['20191103', '20191130', '20191201', '20191202']],
    ['dec', ['20191202', '20191229', '20191230', '20191231']],
]
months = [item[0] for item in datestrings]
index = ['_'.join(product) for product in itertools.product(locations.keys(), months)]
df = pd.DataFrame(index=index, columns=['nrmse_arima', 'r2_arima', 'nrmse_svr', 'r2_svr'], dtype=np.float64)

print('--------------------------------')
print('run started at')
print(datetime.now())
print(f'using filter: {filter}')
print(f'ARIMA order={order}; seasonal_order={seasonal_order}')
print('--------------------------------')
print()

for index, [location, data] in enumerate(locations.items()):
    for month, dates in datestrings:
        training_start, training_end, testing_start, testing_end = dates
        training = data[training_start:training_end]
        testing = data[testing_start:testing_end]

        print(f'iteration {index}/{length}')
        print('--------------------------------')
        print()

        # arima
        print(f'fitting ARIMA model for location {location} and month {month} at {datetime.now()}')
        with warnings.catch_warnings():
            warnings.filterwarnings('error', message='divide by zero encountered in double_scalars')
            try:
                arima = ARIMAModel(scaling=True)
                arima.fit(training, order=order, seasonal_order=seasonal_order, filter=filter, use_exogenous=True)
                prediction = arima.predict(testing_data=testing)
                error_nrmse = nrmse(testing.power, prediction.power)
                error_r2 = r2(testing.power, prediction.power)

                df.loc[f'{location}_{month}'].nrmse_arima = error_nrmse
                df.loc[f'{location}_{month}'].r2_arima = error_r2

                print(f'nRMSE: {error_nrmse}; R2: {error_r2}')
            except Exception as e:
                print(f'ERROR: {str(e)}. Leaving {location}_arima for {month} out')
        print()

        # svr
        print(f'fitting SVR model for location {location} and month {month} at {datetime.now()}')
        with warnings.catch_warnings():
            warnings.filterwarnings('error', message='divide by zero encountered in double_scalars')
            try:
                svr = SVRModel(data, scaling=True)
                svr.fit(training, filter=filter)
                prediction = svr.predict(testing)
                error_nrmse = nrmse(testing.power, prediction.power)
                error_r2 = r2(testing.power, prediction.power)

                df.loc[f'{location}_{month}'].nrmse_svr = error_nrmse
                df.loc[f'{location}_{month}'].r2_svr = error_r2

                print(f'nRMSE: {error_nrmse}; R2: {error_r2}')
            except Exception as e:
                print(f'ERROR: {str(e)}. Leaving {location}_svr for {month} out')
        print()

# extract results
print('--------------------------------')
path = 'out/pvwatts/third/full.csv'
df.to_csv(path)
print(f'saved full output to {path}')

qt = df.quantile([0, 0.25, 0.5, 0.75, 1]).round(2)
qt.loc['average'] = df.mean().round(2)

path = 'out/pvwatts/third/quantiles.csv'
qt.to_csv(path)
print(f'saved extracted data to {path}')
print(f'run finished at {datetime.now()}')
print('--------------------------------')
print()
