from datetime import datetime
import numpy as np
import itertools as it
import pandas as pd
import requests
import urllib.parse

import matplotlib.pyplot as plt

import pytides
from pytides.tide import Tide
import pytides.constituent as cons

import arrow

################################################################

# Pull raw data.  Perhaps we should save this out so that we can survive a change in the service.
# Note units are english!

# Pull 2019 tide heights (measured)
tide_height_vars = dict(
    begin_date = '20190101',
    end_date   = '20191104',
    station    = '9447130',
    product    = 'hourly_height',
    datum      = 'MLLW',
    time_zone  = 'lst',
    units      = 'english',
    format     = 'csv'
)

BASE_URL = 'https://tidesandcurrents.noaa.gov/api/datagetter?'

height_df = pd.read_csv(BASE_URL + urllib.parse.urlencode(tide_height_vars),
                        index_col=0, parse_dates=True)

# Pull the tide predictions
predictions_vars = dict(
    begin_date = '20190101',
    end_date   = '20191231',
    station    = '9447130',
    product    = 'predictions',
    datum      = 'MLLW',
    time_zone  = 'lst',
    units      = 'english',
    format     = 'csv'
)

predictions_df = pd.read_csv(BASE_URL + urllib.parse.urlencode(predictions_vars),
                             index_col=0, parse_dates=True)

def compute_tide_model(height_df, predictions_df):
    water_level = height_df[' Water Level']
    water_level.plot(figsize=(13, 3.5))

    demeaned = water_level.values - water_level.values.mean()
    tide = Tide.decompose(demeaned, list(water_level.index))

    dates = pd.date_range(start='2019-09-01', end='2019-10-31', freq='6T')

    hours = np.cumsum(np.r_[0, [t.total_seconds() / 3600.0 for t in np.diff(dates.to_pydatetime())]])

    times = Tide._times(dates[0], hours)

    prediction = pd.Series(tide.at(times) + water_level.values.mean(), index=dates)

    ax = water_level.plot(figsize=(13, 3.5), label='Observed data')
    ax = prediction.plot(ax=ax, color='red', label='Prediction')
    ax = predictions_df['2019-09-01':].plot(ax=ax, color='green', label='NOAA Prediction')
    leg = ax.legend(loc='best')

# Pull the metadata for Seattle!  
METADATA_URL = 'https://tidesandcurrents.noaa.gov/mdapi/latest/webapi/stations/9447130.json?expand=datums,harcon&units=english'

result = requests.get(METADATA_URL)
json_result = result.json()
station_data = json_result['stations'][0]

datums = pd.DataFrame(station_data['datums']['datums'])
harmonic_constituents = pd.DataFrame(station_data['harmonicConstituents']["HarmonicConstituents"])


################################################################

def build_tide_model(datums, harmonic_constituents):
    # phases = list(harmonic_constituents.phase_GMT)
    phases = list(harmonic_constituents.phase_local)
    amplitudes = list(harmonic_constituents.amplitude)

    MTL = datums[datums.name == 'MTL'].value.values[0]
    MLLW = datums[datums.name == 'MLLW'].value.values[0]
    offset = MTL - MLLW

    constituents = [c for c in cons.noaa if c != cons._Z0]
    constituents.append(cons._Z0)
    phases.append(0)
    amplitudes.append(offset)

    # Build the model.
    assert(len(constituents) == len(phases) == len(amplitudes))
    model = np.zeros(len(constituents), dtype = Tide.dtype)
    model['constituent'] = constituents
    model['amplitude'] = amplitudes
    model['phase'] = phases

    return Tide(model = model, radians = False)

tide = build_tide_model(datums, harmonic_constituents)


for i, row in it.islice(height_df.iterrows(), 0, 200, 10):
    date = arrow.get(row['Date Time'])  # .replace(tzinfo='US/Pacific')
    level = row[' Water Level']
    prediction = tide.at([date.datetime])[0]
    print(date, level, prediction, level-prediction)

    
pacific = arrow.now('US/Pacific')
nyc = arrow.now('America/New_York').tzinfo
pacific.astimezone(nyc)

################################################################

if True:
    # This is all in meters, degrees, and GMT
    noaa_tides = pd.read_csv('/Users/viola/canlogs/NOAA_kings_pt.csv')
    MTL = 5.113
    MLLW = 3.928
    offset = MTL - MLLW



    # Trying for feet, degrees, and local time
    noaa_tides = pd.read_csv('/Users/viola/canlogs/NOAA_seattle_9447130.csv')
    MTL = 14.60
    MLLW = 7.94
    offset = MTL - MLLW

# These are the NOAA constituents, in the order presented on their website.

constituents = [c for c in cons.noaa if c != cons._Z0]
phases = list(noaa_tides.Phase)
amplitudes = list(noaa_tides.Amplitude)

constituents.append(cons._Z0)
phases.append(0)
amplitudes.append(offset)

#Build the model.
assert(len(constituents) == len(phases) == len(amplitudes))
model = np.zeros(len(constituents), dtype = Tide.dtype)
model['constituent'] = constituents
model['amplitude'] = amplitudes
model['phase'] = phases

tide = Tide(model = model, radians = False)

print(tide.at([datetime(2013,1,1,0,0,0), datetime(2013,1,1,6,0,0)]))
print(tide.at([datetime(2019,11,2,0,0,0), datetime(2019,11,2,12,0,0)]))

year = 2013
p1, p2 = tide.at([datetime(year,11,2,0,0,0), datetime(year,11,2,12,0,0)])

n1 = 1.325
n2 = 9.833

p1 - n1
p2 - n2

################################################################



    
    
