import os
import pandas as pd
import requests
import json
import warnings

API_KEY = os.environ['PVWATTS_API_KEY']

def _json_to_dataframe(json):
    outputs = json['outputs']
    data = {key: outputs[key] for key in ['ac', 'tamb', 'wspd']}
    data['power'] = data.pop('ac')
    data = pd.DataFrame(data)
    data['time'] = pd.date_range('20190101', periods=len(data), freq='H')
    data.set_index('time', inplace=True)
    return data

warnings.simplefilter('always')
def load(system_capacity=4, module_type=0, losses=14, array_type=0, tilt=25, azimuth=180, address=None, lat=51.9607, lon=7.6261, radius=100, dataset='intl', suppress_warnings=False):
    """
    Imports data from PVWatts using the requests package.
    Only fields that are of importance for this forecasting purpose
    can be specified.
    """
    params = {
        'api_key': API_KEY,
        'system_capacity': system_capacity,
        'module_type': module_type,
        'losses': losses,
        'array_type': array_type,
        'tilt': tilt,
        'azimuth': azimuth,
        'address': address,
        'lat': lat,
        'lon': lon,
        'radius': radius,
        'timeframe': 'hourly',
        'dataset': dataset
    }

    response = requests.get('https://developer.nrel.gov/api/pvwatts/v6.json', params)
    json = response.json()
    if not suppress_warnings:
        if json['errors']: warnings.warn(f'API ERROR: {json["errors"]}')
        if json['warnings']: warnings.warn(f'API WARNING: {json["warnings"]}')
    response.raise_for_status()
    print(f"loaded {json['station_info']['city']}")
    return _json_to_dataframe(json)

def load_from_json(filepath):
    """
    Imports PVWatts data that has been downloaded and stored in a json file

    filepath: str. path to a json file containing a valid json document
                   as specified by the PVWatts API.
    """
    with open(filepath) as file:
        data = json.load(file)
    return _json_to_dataframe(data)

def bulk_load_from_list(filepath, range=None):
    """
    Bulk Imports data from PVWatts using the load method.

    filepath: str. Path to a csv file containing the columns 'city', 'lat' and 'lon'
    range: tuple. range of cities to load
    """
    list = pd.read_csv(filepath)
    if not range: range = (0, len(list))
    start, stop = range
    list = list[start:stop]
    cities = {}
    for index, (city, lat, lon) in enumerate(list.values):
        try:
            cities[city] = load(lat=lat, lon=lon, suppress_warnings=True)
        except requests.HTTPError as e:
            cities[city] = load(lat=lat, lon=lon, dataset='tmy3')
    return cities

def load_city_from_list(filepath, city):
    """
    Import a specific City

    filepath: str. Path to a csv file containing the columns 'city', 'lat' and 'lon'
    city: str. City name
    """
    list = pd.read_csv(filepath).set_index('city')
    city = list.loc[city]
    try:
        result = load(lat=city.lat, lon=city.lon, suppress_warnings=True)
    except requests.HTTPError as e:
        result = load(lat=city.lat, lon=city.lon, dataset='tmy3')
    return result
