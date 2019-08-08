from importers import pvwatts
from predictors.svr_model import SVRModel
from predictors.arima_model import ARIMAModel
from evaluation.error_terms import rmse, nrmse
import pandas as pd
from datetime import datetime

# prepare

london = pvwatts.load_from_json('data/pvwatts/london.json')
new_york = pvwatts.load_from_json('data/pvwatts/new_york.json')

df_rmse = pd.DataFrame(index=['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec',], columns=['london_arima', 'london_svr', 'new_york_arima', 'new_york_svr'])
df_nrmse = df_rmse.copy()

filter = ['tamb', 'wspd']
order = (2,0,1)
seasonal_order = (2,0,1,24)
# p, q, P, Q = (1,3), (1,3), (1,3), (1,3)
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

# run
print('--------------------------------')
print('run started at')
print(datetime.now())
print(f'using filter: {filter}')
print(f'ARIMA order={order}; seasonal_order={seasonal_order}')
print('--------------------------------')
print()

for location, data in {'london': london, 'new_york': new_york}.items():
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
            svr = SVRModel(data, scaling=True)
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
df_rmse.to_csv('out/pvwatts/second_rmse.csv')

df_nrmse.loc['average'] = df_nrmse.mean().round(2)
df_nrmse.to_csv('out/pvwatts/second_nrmse.csv')

print('--------------------------------')
print('saved dataframes to out/pvwatts/second_(n)rmse.csv')
print(f'run finished at {datetime.now()}')
print('--------------------------------')
print()
