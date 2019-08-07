# Photovoltaic Power Prediction

This repository contains two alternative implementations for forecasting photovoltaic power prediction based on weather data. It was developed for my bachelor thesis that compares those two approaches. Two data sources serve as weather and power output sample data.

## Setup the Project

I recommend [pipenv]([https://github.com/pypa/pipenv]) to setup the project. With pipenv installed run `pipenv install` from the project root and it will install all necessary dependencies. Then use `pipenv shell` to enter the shell where all dependencies will be available or just `pipenv run <command>` to execute a specific command.

If you don't use pipenv, dependencies are specified in the [Pipfile](Pipfile) and the versions i use can be found in the [Pipfile.lock](Pipfile.lock).

To use the PVWatts Service described later you need to obtain an API key. Then, you have to set it as environment variable (or put it in the code directly). If you use pipenv this can be done in a `.env` file.

Examples on how to use the project can be found in the example files in the root directory. You can directly run those files ***after*** completing the necessary steps to obtain data described in the next section.

```
pipenv run python example_arima_pvwatts.py
pipenv run python example_arima_uq.py
pipenv run python example_svr_pvwatts.py
pipenv run python example_svr_uq.py
```

For plotting with matplotlib, pandas issues a future warning, so i put the following line after the imports:

``` python
from pandas.plotting import register_matplotlib_converters; register_matplotlib_converters()
```

## Load Data

There are two datasources for which an importer is available that parses the data in the right format. Of course it is possible to use other sources as well. Firstly there is a [PVWatts](https://pvwatts.nrel.gov/) wrapper and secondly one for data obtained from [UQ Solar](https://solar-energy.uq.edu.au/).

### PVWatts

Prepared data from PVWatts looks similar to this:

|         time        | tamb | wspd |  power   |
|---------------------|:----:|:----:|:--------:|
| 2019-06-20 06:00:00 | 16.0 | 0.0  | 957.201  |
| 2019-06-20 07:00:00 | 18.0 | 1.5  | 1648.060 |
| 2019-06-20 08:00:00 | 21.0 | 3.1  | 2260.658 |
| 2019-06-20 09:00:00 | 23.0 | 4.1  | 2647.271 |
| 2019-06-20 10:00:00 | 24.0 | 2.6  | 2838.976 |

It can be obtained via the public [PVWatts API](https://developer.nrel.gov/docs/solar/pvwatts/v6/). In order to get an [API key](https://developer.nrel.gov/docs/api-key/) you need to sign up at the NREL Developer Network. The API key then needs to be put in the environment variable `PVWATTS_API_KEY`. Alternatively, you can insert it into [pvwatts](importers/pvwatts.py) directly. When that is done you can use

``` python
from importers import pvwatts
data = pvwatts.load()
```

This calls the API and parses the result as a pandas DataFrame which can be passed to the forecasting modules. You can also pass the following optional parameters to `load`:

 - system_capacity
 - module_type
 - losses
 - array_type
 - tilt
 - azimuth
 - address
 - lat
 - lon
 - radius

For information on these parameters please refer to the PVWatts V6 [API Description](https://developer.nrel.gov/docs/solar/pvwatts/v6/). All parameters have default values so calling the method without parameters is possible as well. The return data is indexed using a DateTimeIndex. Since PVWatts does not specify dates but always returns data for a whole year, the year 2019 will be set fixed for each DataFrame returned from this module.

### UQ Solar Photovoltaic Data

Prepared data from the UQ Solar live feed looks similar to this:

|         time        | airtemp | humidity | windspeed | winddirection |   power   |
|---------------------|:-------:|:--------:|:---------:|:-------------:|:---------:|
| 2015-06-20 07:00:00 |  9.27   |   75.24  |    1.72   |     191.47    | 25302.00  |
| 2015-06-20 08:00:00 |  11.28  |   68.72  |    2.14   |     201.45    | 78240.83  |
| 2015-06-20 09:00:00 |  13.38  |   59.02  |    2.59   |     214.77    | 128523.08 |
| 2015-06-20 10:00:00 |  14.49  |   54.19  |    2.92   |     201.67    | 162968.83 |
| 2015-06-20 11:00:00 |  15.65  |   51.89  |    2.34   |     203.23    | 172535.83 |

It can downloaded on the [live feed](http://solar.uq.edu.au/user/reportPower.php) website. Choose a `PV Site` and a `PV Array` from the sidebar on the right. Then click `Download Data` and then `Download Daily Logs`. From there you can specify a date range (I recommend a year), and then download a `Power & Energy` file as well as a `Weather` file. Make sure you download both with the same date ranges specified.

Now you can use

``` python
from importers import uq
data = uq.load('power_file.csv', 'weather_file.csv')
```

and substitute both parameters with the file paths for each respective file. The `load` method combines both files again into a pandas DataFrame, ready to be passed to a forecasting module.

## Forecast Power Output

Now that a DataFrame with features and power data is present you can make forecasts. Both importers return a DataFrame which has different features, but both have a `power` column which represents the power output.

Since the data from both importers has a DateTimeIndex it can be split up into training and testing data like this:

``` python
training_data = data['20190601':'20190607'] # first week of june 2019
testing_data = data['20190608':'20190614'] # second week of june 2019
```

Here the dates are converted from strings implicitly. For example `'20190601'` depicts 2019/06/01 or June 01, 2019.

### Support Vector Regresion

This algorithm uses the [Scikit-Learn implementation](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html) of SVR which is based on libsvm. To use it the following class is available:

``` python
from predictors.svr_model import SVRModel
```

Making a prediction without scaling the data:

``` python
model = SVRModel(scaling=False)
model.fit(training_data, kernel='rbf', C=1e3, gamma=0.1, epsilon=0.1)
model.predict(testing_data) # returns the prediction data frame containing the testing features and the power prediction
model.prediction.power # use this to access the power prediction
```

Filtering out the `power` column from the `testing_data` frame is not necessary, the `SVRModel` class does that automatically. Optionally it is possible to pass `filter=['airtemp', 'humidity']` or similar to `fit` to restrict the features used for forecasting. It is not necessary to filter the data manually this way.

The SVR variables `kernel`, `C`, `gamma` and `epsilon` are all optional, they all have default values.

To let the model scale the data before applying the regression you can set the scaling parameter to `True` (this is the default value):

``` python
model = svr_model(base_data=data, scaling=True)
```

That way you do have to specify `base_data` which can be a dataset for another year. This will not be used for regression, solely for fitting a feasible `scaler` Object. For more information on that refer to the [StandartScaler implementation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) of Scikit-Learn.

### ARIMA

This algorithm makes use on the [pmdarima package](https://www.alkaline-ml.com/pmdarima/) (formerly pyramid-arima). It can be used by importing the necessary class:

``` python
from predictors.arima_model import ARIMAModel
```

Now to make a prediction without using exogenous variables do the following:

``` python
model = ARIMAModel() # scaling is set to true automatically
model.fit(training_data, order=(2,1,4), seasonal_order(3,1,2, 24), use_exogenous=False)
model.predict(hours=48)  # returns the prediction
model.prediction.power # use this to access the power predictionn
```

Providing testing features is not necessary because the model was fit without exogenous variables. In this case, the returned DataFrame will only have a `power` column. The `order` specifies the `(p,d,q)` parameters of the model. The `seasonal_order` specifies the `(P,D,Q,s)` parameters. The scaling parameter can be set when creating the model, by default it is set to `True` but it can be unset with:

``` python
model = ARIMAModel(scaling=false)
```

If you want to add exogenous variables set this parameter:

``` python
model.fit(training_data, order=(2,1,4), seasonal_order(3,1,2, 24), use_exogenous=True) # use_exogenous is True by default
model.predict(testing_data=testing_data)
```

This way you have to specify `testing_data` when predicting. The `power` column will be automatically removed when making the actual prediction. The prediction DataFrame will now have the testing_data columns where the `power` column contains the predicted values.

If you want to let the algorithm automatically find appropriate `p, q, P and Q` parameters use the following call:

``` python
model.fit_auto(training_data, p=(0,5), q=(0,5), P=(0,5), Q=(0,5), d=1, D=1)
```

The tuples define ranges in which the algorithm will search for optimal parameters. The `d` and `D` can be left out to automatically determine those parameters aswell. As with the `fit` method, `use_exogenous` will be set to `True` by default, but can also be specified to be
`False`.

As it is possible with the `SVRModel`, you can also pass a filter array to `fit` and `fit_auto` to restrict which features should be included in the forecasting process.
