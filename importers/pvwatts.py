import os
import pandas as pd
import requests

def load(system_capacity=4, module_type=0, losses=14, array_type=1, tilt=25, azimuth=180, address=None, lat=51.9607, lon=7.6261, radius=0):
    """
    Imports data from PVWatts using the pypvwatts package.
    Only fields that are of importance for this forecasting purpose
    can be specified.
    """
    api_key = os.environ['PVWATTS_API_KEY']
    params = {
        'api_key': api_key,
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
        'dataset': 'tmy3'
    }

    response = requests.get('https://developer.nrel.gov/api/pvwatts/v6.json', params)
    response.raise_for_status()
    outputs = response.json()['outputs']

    data = {key: outputs[key] for key in ['ac', 'tamb', 'wspd']}
    data['power'] = data.pop('ac')
    data = pd.DataFrame(data)
    data['time'] = pd.date_range('20190101', periods=len(data), freq='H')
    data.set_index('time', inplace=True)
    return data
