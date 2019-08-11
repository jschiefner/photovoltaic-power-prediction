import warnings
from importers import uq
from predictors.arima_model import ARIMAModel
from predictors.svr_model import SVRModel
from evaluation.error_terms import nrmse, r2
import pandas as pd
from datetime import datetime
import numpy as np
import itertools

# prepare
locations = [
    [2012, 'stl', uq.load('data/uq/power/uq_centre_st_lucia/2012.csv', 'data/uq/weather/2012.csv')],
    [2013, 'stl', uq.load('data/uq/power/uq_centre_st_lucia/2013.csv', 'data/uq/weather/2013.csv')],
    [2014, 'stl', uq.load('data/uq/power/uq_centre_st_lucia/2014.csv', 'data/uq/weather/2014.csv')],
    [2015, 'stl', uq.load('data/uq/power/uq_centre_st_lucia/2015.csv', 'data/uq/weather/2015.csv')],
    [2016, 'stl', uq.load('data/uq/power/uq_centre_st_lucia/2016.csv', 'data/uq/weather/2016.csv')],
    [2017, 'stl', uq.load('data/uq/power/uq_centre_st_lucia/2017.csv', 'data/uq/weather/2017.csv')],
    [2012, 'car', uq.load('data/uq/power/car_park_1/2012.csv', 'data/uq/weather/2012.csv')],
    [2013, 'car', uq.load('data/uq/power/car_park_1/2013.csv', 'data/uq/weather/2013.csv')],
    [2014, 'car', uq.load('data/uq/power/car_park_1/2014.csv', 'data/uq/weather/2014.csv')],
    [2015, 'car', uq.load('data/uq/power/car_park_1/2015.csv', 'data/uq/weather/2015.csv')],
    [2016, 'car', uq.load('data/uq/power/car_park_1/2016.csv', 'data/uq/weather/2016.csv')],
    [2017, 'car', uq.load('data/uq/power/car_park_1/2017.csv', 'data/uq/weather/2017.csv')],
    [2012, 'con', uq.load('data/uq/power/concentrating_array/2012.csv', 'data/uq/weather/2012.csv')],
    [2013, 'con', uq.load('data/uq/power/concentrating_array/2013.csv', 'data/uq/weather/2013.csv')],
    [2014, 'con', uq.load('data/uq/power/concentrating_array/2014.csv', 'data/uq/weather/2014.csv')],
    [2015, 'con', uq.load('data/uq/power/concentrating_array/2015.csv', 'data/uq/weather/2015.csv')],
    [2016, 'con', uq.load('data/uq/power/concentrating_array/2016.csv', 'data/uq/weather/2016.csv')],
    [2017, 'con', uq.load('data/uq/power/concentrating_array/2017.csv', 'data/uq/weather/2017.csv')],
]
length = len(locations)
length_monthly = length * 12

filter = ['airtemp', 'humidity']
order = (2,0,1)
seasonal_order = (2,0,1,24)
datestrings = {
    2012: [
        ['jan', ['20120104', '20120131', '20120201', '20120202']],
        ['feb', ['20120201', '20120228', '20120301', '20120302']],
        ['mar', ['20120304', '20120331', '20120401', '20120402']],
        ['apr', ['20120403', '20120430', '20120501', '20120502']],
        ['may', ['20120504', '20120531', '20120601', '20120602']],
        ['jun', ['20120603', '20120630', '20120701', '20120702']],
        ['jul', ['20120704', '20120731', '20120801', '20120802']],
        ['aug', ['20120804', '20120831', '20120901', '20120902']],
        ['sep', ['20120903', '20120930', '20121001', '20121002']],
        ['oct', ['20121004', '20121031', '20121101', '20121102']],
        ['nov', ['20121103', '20121130', '20121201', '20121202']],
        ['dec', ['20121202', '20121229', '20121230', '20121231']],
    ],
    2013: [
        ['jan', ['20130104', '20130131', '20130201', '20130202']],
        ['feb', ['20130201', '20130228', '20130301', '20130302']],
        ['mar', ['20130304', '20130331', '20130401', '20130402']],
        ['apr', ['20130403', '20130430', '20130501', '20130502']],
        ['may', ['20130504', '20130531', '20130601', '20130602']],
        ['jun', ['20130603', '20130630', '20130701', '20130702']],
        ['jul', ['20130704', '20130731', '20130801', '20130802']],
        ['aug', ['20130804', '20130831', '20130901', '20130902']],
        ['sep', ['20130903', '20130930', '20131001', '20131002']],
        ['oct', ['20131004', '20131031', '20131101', '20131102']],
        ['nov', ['20131103', '20131130', '20131201', '20131202']],
        ['dec', ['20131202', '20131229', '20131230', '20131231']],
    ],
    2014: [
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
    ],
    2015: [
        ['jan', ['20150104', '20150131', '20150201', '20150202']],
        ['feb', ['20150201', '20150228', '20150301', '20150302']],
        ['mar', ['20150304', '20150331', '20150401', '20150402']],
        ['apr', ['20150403', '20150430', '20150501', '20150502']],
        ['may', ['20150504', '20150531', '20150601', '20150602']],
        ['jun', ['20150603', '20150630', '20150701', '20150702']],
        ['jul', ['20150704', '20150731', '20150801', '20150802']],
        ['aug', ['20150804', '20150831', '20150901', '20150902']],
        ['sep', ['20150903', '20150930', '20151001', '20151002']],
        ['oct', ['20151004', '20151031', '20151101', '20151102']],
        ['nov', ['20151103', '20151130', '20151201', '20151202']],
        ['dec', ['20151202', '20151229', '20151230', '20151231']],
    ],
    2016: [
        ['jan', ['20160104', '20160131', '20160201', '20160202']],
        ['feb', ['20160201', '20160228', '20160301', '20160302']],
        ['mar', ['20160304', '20160331', '20160401', '20160402']],
        ['apr', ['20160403', '20160430', '20160501', '20160502']],
        ['may', ['20160504', '20160531', '20160601', '20160602']],
        ['jun', ['20160603', '20160630', '20160701', '20160702']],
        ['jul', ['20160704', '20160731', '20160801', '20160802']],
        ['aug', ['20160804', '20160831', '20160901', '20160902']],
        ['sep', ['20160903', '20160930', '20161001', '20161002']],
        ['oct', ['20161004', '20161031', '20161101', '20161102']],
        ['nov', ['20161103', '20161130', '20161201', '20161202']],
        ['dec', ['20161202', '20161229', '20161230', '20161231']],
    ],
    2017: [
        ['jan', ['20170104', '20170131', '20170201', '20170202']],
        ['feb', ['20170201', '20170228', '20170301', '20170302']],
        ['mar', ['20170304', '20170331', '20170401', '20170402']],
        ['apr', ['20170403', '20170430', '20170501', '20170502']],
        ['may', ['20170504', '20170531', '20170601', '20170602']],
        ['jun', ['20170603', '20170630', '20170701', '20170702']],
        ['jul', ['20170704', '20170731', '20170801', '20170802']],
        ['aug', ['20170804', '20170831', '20170901', '20170902']],
        ['sep', ['20170903', '20170930', '20171001', '20171002']],
        ['oct', ['20171004', '20171031', '20171101', '20171102']],
        ['nov', ['20171103', '20171130', '20171201', '20171202']],
        ['dec', ['20171202', '20171229', '20171230', '20171231']],
    ],
}

years = [str(key) for key in datestrings.keys()]
months = [item[0] for item in datestrings[2012]]
index = ['_'.join(product) for product in itertools.product(['stl', 'car', 'con'], years, months)]
df = pd.DataFrame(index=index, columns=['nrmse_arima', 'r2_arima', 'nrmse_svr', 'r2_svr'], dtype=np.float64)

print('--------------------------------')
print('run started at')
print(datetime.now())
print(f'using filter: {filter}')
print(f'ARIMA order={order}; seasonal_order={seasonal_order}')
print('--------------------------------')
print()

for index, [year, location, data] in enumerate(locations):
    for month, dates in datestrings[year]:
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

                df.loc[f'{location}_{year}_{month}'].nrmse_arima = error_nrmse
                df.loc[f'{location}_{year}_{month}'].r2_arima = error_r2

                print(f'nRMSE: {error_nrmse}; R2: {error_r2}')
            except Exception as e:
                print(f'ERROR: {str(e)}. Leaving out {location}_{year}_{month} for svr')
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

                df.loc[f'{location}_{year}_{month}'].nrmse_svr = error_nrmse
                df.loc[f'{location}_{year}_{month}'].r2_svr = error_r2

                print(f'nRMSE: {error_nrmse}; R2: {error_r2}')
            except Exception as e:
                print(f'ERROR: {str(e)}. Leaving out {location}_{year}_{month} for svr')
        print()

# extract results
print('--------------------------------')
path = 'out/uq/third/full.csv'
df.to_csv(path)
print(f'saved full output to {path}')

qt = df.quantile([0, 0.25, 0.5, 0.75, 1]).round(2)
qt.loc['average'] = df.mean().round(2)

path = 'out/uq/third/quantiles.csv'
qt.to_csv(path)
print(f'saved extracted data to {path}')
print(f'run finished at {datetime.now()}')
print('--------------------------------')
print()
