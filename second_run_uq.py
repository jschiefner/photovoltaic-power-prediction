from importers import uq
from predictors.svr_model import SVRModel
from predictors.arima_model import ARIMAModel
from evaluation.error_terms import rmse, nrmse
import pandas as pd
from datetime import datetime

# prepare
edwards = uq.load('data/uq/power/sir_llew_edwards/2014.csv', 'data/uq/weather/2014.csv')
edwards_base = uq.load('data/uq/power/sir_llew_edwards/2013.csv', 'data/uq/weather/2013.csv')
car_park = uq.load('data/uq/power/car_park_1/2014.csv', 'data/uq/weather/2014.csv')
car_park_base = uq.load('data/uq/power/car_park_1/2013.csv', 'data/uq/weather/2013.csv')

df_rmse = pd.DataFrame(index=['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec',], columns=['edwards_arima', 'edwards_svr', 'car_park_arima', 'car_park_svr'])
df_nrmse = df_rmse.copy()

filter = ['airtemp', 'humidity']
order = (2,0,1)
seasonal_order = (2,0,1,24)
# p, q, P, Q = (1,3), (1,3), (1,3), (1,3)
datestrings = [
    ['jan', ['20140104', '20140131', '20140201', '20140202']],
    ['feb', ['20140201', '20140228', '20140301', '20140302']],
    ['mar', ['20140304', '20140331', '20140401', '20140402']],
    ['apr', ['20140403', '20140430', '20140501', '20140502']],
    ['may', ['20140504', '20140531', '20140601', '20140602']],
    ['jun', ['20140603', '20140630', '20140701', '20140702']],
    ['jul', ['20140704', '20140731', '20140801', '20140802']],
    ['aug', ['20140804', '20140831', '20140901', '20140902']],
    ['sep', ['20140903', '20140930', '20141001', '20141002']],
    ['oct', ['20141004', '20141031', '20141101', '20141102']],
    ['nov', ['20141103', '20141130', '20141201', '20141202']],
    ['dec', ['20141202', '20141229', '20141230', '20141231']],
]

# run
print('--------------------------------')
print('run started at')
print(datetime.now())
print(f'using filter: {filter}')
print(f'ARIMA order={order}; seasonal_order={seasonal_order}')
print('--------------------------------')
print()

for location, [data, base_data] in {'edwards': [edwards, edwards_base], 'car_park': [car_park, car_park_base]}.items():
    columns = {}
    for month, dates in datestrings:
        training_start, training_end, testing_start, testing_end = dates
        training = data[training_start:training_end]
        testing = data[testing_start:testing_end]

        # arima
        print(f'fitting ARIMA model for location {location} and month {month} at {datetime.now()}')
        try:
            arima = ARIMAModel(scaling=True)
            arima.fit(training, order=order, seasonal_order=seasonal_order, filter=filter, use_exogenous=True)
            # arima.fit_auto(training, p=p, q=q, P=P, Q=Q, d=0, D=0, filter=filter, trace=False, use_exogenous=True)
            # print(f'successfully fitted ARIMA model with order={arima.model.order}; seasonal_order={arima.model.seasonal_order}')
            prediction = arima.predict(testing_data=testing)
            error_rmse = rmse(testing.power, prediction.power)
            error_nrmse = nrmse(testing.power, prediction.power)
            df_rmse[f'{location}_arima'][month] = error_rmse
            df_nrmse[f'{location}_arima'][month] = error_nrmse
            print(f'RMSE: {error_rmse}; nRMSE: {error_nrmse}')
        except Exception as e:
            print(f'ERROR: {str(e)}. Leaving {location}_arima blank for {month}')
        print()

        # svr
        print(f'fitting SVR model for location {location} and month {month} at {datetime.now()}')
        try:
            svr = SVRModel(base_data, scaling=True)
            svr.fit(training, filter=filter)
            prediction = svr.predict(testing)
            error_rmse = rmse(testing.power, prediction.power)
            error_nrmse = nrmse(testing.power, prediction.power)
            df_rmse[f'{location}_svr'][month] = error_rmse
            df_nrmse[f'{location}_svr'][month] = error_nrmse
            print(f'RMSE: {error_rmse}; nRMSE: {error_nrmse}')
        except Exception as e:
            print(f'ERROR: {str(e)}. Leaving {location}_svr blank for {month}')
        print()

# calculate averages
print('--------------------------------')
print('calculating averages')
print('--------------------------------')
print()

df_rmse.loc['average'] = df_rmse.mean().round(2)
df_rmse.to_csv('out/uq/second/rmse.csv')

df_nrmse.loc['average'] = df_nrmse.mean().round(2)
df_nrmse.to_csv('out/uq/second/nrmse.csv')

print('--------------------------------')
print('saved dataframes to out/uq/second/(n)rmse.csv')
print(f'run finished at {datetime.now()}')
print('--------------------------------')
print()
