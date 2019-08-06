import pandas as pd

def load(power_file, weather_file):
    """
    Imports data from a power file and an associated weather file
    Such files can be downloaded from 'http://solar.uq.edu.au/user/reportPower.php'

    power_file: path to the file containing the power output values
    weather_file: path to the file containting the weather data

    returns: a pandas DataFrame containing the combined values
    """

    power = pd.read_csv(power_file, parse_dates=['time']).set_index('time')
    weather = pd.read_csv(weather_file, parse_dates=['time']).set_index('time')

    if power.index[0].date() != weather.index[0].date() or power.index[len(power)-1].date() != weather.index[len(weather)-1].date():
        raise pd.errors.ParserError('The dates of the power and weather file need to match')

    weather['power'] = power['power (W)'] # integrate power into weather data
    data = weather.loc[~weather.index.duplicated(keep='first')] # remove duplicates
    data = data.fillna(0) # fill NaN values with zeros, because power is only specified for daytime
    data = data.resample('H').mean() # resample with hourly average
    data = data.fillna(0) # necessary again after resampling
    data = data.round(2) # cutoff unnessecary decimal points
    data.drop('insolation', axis=1, inplace=True)
    return data
