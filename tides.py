"""
Loads a cached set of NOAA tide prediction data and then looks up the closest tide for
any time.
"""

import os
import os.path

import pandas as pd
import numpy as np
import scipy
import scipy.signal
import cv2

import urllib
import itertools as it
import datetime

from global_variables import G
import utils
import chart
from utils import DictClass
import uw

from nbutils import display_markdown, display

G.init_seattle()


import matplotlib.pyplot as plt

TIDE_DATA = None

NOAA_STATIONS = {
    "port_townsend" : 9444900,
    "seattle" : 9447130,
    "tacoma" : 9446484
}

# These do not have direct tide predictions.  The model used by NOAA is pretty primitive,
# and I have not implemented it.  Each subordinate station gets a multiplier and a hi/lo
# time offset.
NOAA_SUBORDINATE_STATIONS = {
    "meadow_point" : 9447265,
    "point_vashon" : 9446025,
    "gig_harbor" : 9446369,
    "foulweather_bluff" : 9445016,
    "marrowstone_point" : 9444972,
}

# Super helpful page: https://tidesandcurrents.noaa.gov/PageHelp.html
#
# Specifically, subordinate stations only have time offsets for high/low (in minutes) and
# multipliers for high/low.
#
# Harmonic constituents for Seattle:
# https://tidesandcurrents.noaa.gov/harcon.html?unit=1&timezone=0&id=9447130&name=Seattle&state=WA
# https://tidesandcurrents.noaa.gov/mdapi/latest/webapi/stations/9447130.json?expand=harcon&units=english
#
# https://tidesandcurrents.noaa.gov/api-helper/url-generator.html
#


def load_tide_data(station_number):
    "Load NOAA tide predictions for a station. Fetch from NOAA and cache."
    tides_path = tide_data_path(station_number)
    if not os.path.exists(tides_path):
        G.logger.info(f"Caching tide predictions for station {station_number} : {tides_path}.")
        df = fetch_data(station_number)
        df.to_pickle(tides_path)
    else:
        df = pd.read_pickle(tides_path)
    df.date_time = df.date_time.dt.tz_convert(G.TIMEZONE)
    return df

def init(station_number):
    "Initialize tides predictions for a NOAA tide station."
    global TIDE_DATA
    tides_df = load_tide_data(station_number)
    TIDE_DATA = DictClass(
        station_number = station_number,
        df = tides_df,
        base_time = tides_df.date_time.iloc[0],
        last_time = tides_df.date_time.iloc[-1]
    )

def tide_data_path(station_number):
    return os.path.join(G.DATA_DIRECTORY, f"Tides/{station_number}.pd")    

def init_location(location_name):
    "Initialize tides for a location.  Not many currently supported!"
    if location_name.casefold() in NOAA_STATIONS:
        init(NOAA_STATIONS[location_name])
    else:
        raise Exception(f"Could not find {location_name}.")

# Utils ################################################################
def normalize_column_names(df):
    "Rationalize the column names from a foreign source.  No spaces. Lowercase. Underscores separated."
    fold = df.columns.str.lower()
    strip = fold.str.strip()
    under = strip.str.replace(' ', '_')
    df.columns = under
    return df

def fetch_data(station_number):
    "Fetch the data from NOAA and create a pandas dataframe."
    predictions_vars = dict(
        station    = str(station_number),
        product    = 'predictions',
        datum      = 'MLLW',
        time_zone  = 'gmt',
        units      = 'english',
        format     = 'csv'
    )
    dates = [dict(begin_date='20190101', end_date='20191231'),
             dict(begin_date='20200101', end_date='20201231'),
             dict(begin_date='20210101', end_date='20211231')]
    BASE_URL = 'https://tidesandcurrents.noaa.gov/api/datagetter?'
    dfs = []
    for d in dates:
        predictions_vars.update(d)
        url = BASE_URL + urllib.parse.urlencode(predictions_vars)
        G.logger.info(f"Loading NOAA predicitions.")
        G.logger.info(url)
        df = pd.read_csv(url, parse_dates=True)
        dfs.append(df)

    df = pd.concat(dfs, sort=True, ignore_index=True)
    df = normalize_column_names(df)
    df.date_time = pd.to_datetime(df.date_time)
    df.date_time = df.date_time.dt.tz_localize('UTC')

    return df

################################################################
# Code to retrieve the tides at times.

def tides_at_closest(times):
    times = convert_to_pd_times(times)
    if (times < TIDE_DATA.base_time).any():
        raise Exception("Tide time too early {0}".format(times.min()))
    if (times > TIDE_DATA.last_time).any():
        raise Exception("Time time too late {0}".format(times.max()))
    tide_index = ((times - TIDE_DATA.base_time).dt.total_seconds() / 360).round().astype(np.int)
    return TIDE_DATA.df.iloc[tide_index].prediction.values

def tides_at(times):
    times = convert_to_pd_times(times)
    if (times < TIDE_DATA.base_time).any():
        raise Exception("Tide time too early {0}".format(times.min()))
    if (times > TIDE_DATA.last_time).any():
        raise Exception("Time time too late {0}".format(times.max()))
    delta_sec = np.asarray((times - TIDE_DATA.base_time).dt.total_seconds())
    delta_step = delta_sec / 360  # One prediction every 6 minutes
    before = np.floor(delta_step).astype(np.int32)
    after = np.ceil(delta_step).astype(np.int32)
    alpha = delta_step - before
    val = (1 - alpha) * np.asarray(TIDE_DATA.df.iloc[before].prediction)
    val += alpha * np.asarray(TIDE_DATA.df.iloc[after].prediction)
    return val

def convert_to_pd_times(times):
    "Times and timezones are annoying.  This is an attempt to hide some of the complexity.  Not clear it will work!"
    if isinstance(times, pd.Series):
        res = times.dt.tz_convert(G.TIMEZONE)
    else:
        res = pd.Series(times)
    return res.dt.tz_convert(G.TIMEZONE)

def race_tides(date):
    "For a given race day, load the tides and display."
    noon = utils.time_from_string(f"{date} 12:00:00").datetime
    start = noon + datetime.timedelta(hours=-18)
    end = noon + datetime.timedelta(hours=+18)    
    times = pd.date_range(start, end, 36*10)
    return times, tides_at(times)


def example_plot_tides(date="2020-06-12"):
    init_location("seattle")
    rtimes, rtides = race_tides(date)
    plt.clf()
    plt.plot(rtimes, rtides)

################################################################
# Match the tides to the tide prints

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = it.tee(iterable)
    next(b, None)
    return zip(a, b)

def nwise(iterable, n=2):                                                      
    iters = it.tee(iterable, n)                                                     
    for c, i in enumerate(iters):                                               
        next(it.islice(i, c, c), None)                                               
    return zip(*iters)

def create_tide_print(time, index, tide, name):
    return DictClass(time = time, index = index, tide = tide, name=name)

def tide_prints(date, start_time, finish_time):
    tstart = utils.time_from_string(f"{date} {start_time}")
    tfinish = utils.time_from_string(f"{date} {finish_time}")
    rtimes, rtides = race_tides(date)
    plt.clf()
    plt.plot(rtides)
    highs, _ = scipy.signal.find_peaks(rtides)
    lows, _ = scipy.signal.find_peaks(-rtides)
    extrema = sorted(highs.tolist() + lows.tolist())
    res = []
    for a, b, c in nwise(extrema, 3):
        print(a, b, c)
        mid = np.int((b+c)/2)
        ta, tb, tc, tmid = [rtides[i] for i in [a, b, c, mid]]
        if tb > ta and tb > tc:
            # maxima
            if (tb - ta) > 7.5 and (tb - tc) > 7.5:
                # big swing before and big after
                res.append(create_tide_print(time = rtimes[b], index = b, tide = tb, name='hihi'))
                # is this big drop?
                res.append(create_tide_print(time = rtimes[mid], index = mid, tide = tmid, name='bigdrop'))
            elif (tb - tc) > 7.5:
                # swing after is big (but not before)
                res.append(create_tide_print(time = rtimes[b], index = b, tide = tb, name='lohi'))
                res.append(create_tide_print(time = rtimes[mid], index = mid, tide = tmid, name='bigdrop'))
            elif (tb - ta) > 7.5:
                # swing before is big (but not after)
                res.append(create_tide_print(time = rtimes[b], index = b, tide = tb, name='hihi'))
                res.append(create_tide_print(time = rtimes[mid], index = mid, tide = tmid, name='smalldrop'))
            else:
                # swing before and after small
                res.append(create_tide_print(time = rtimes[b], index = b, tide = tb, name='lohi'))
                res.append(create_tide_print(time = rtimes[mid], index = mid, tide = tmid, name='smalldrop'))
        elif tb < ta and tb < tc:
            # minima
            if (ta - tb) > 7.5 and (tc - tb) > 7.5:
                # big swing before and after
                res.append(create_tide_print(time = rtimes[b], index = b, tide = tb, name='lolo'))
                res.append(create_tide_print(time = rtimes[mid], index = mid, tide = tmid, name='bigrise'))
            elif (tc - tb) > 7.5:
                # swing after is big (but not before)
                res.append(create_tide_print(time = rtimes[b], index = b, tide = tb, name='lolo'))
                res.append(create_tide_print(time = rtimes[mid], index = mid, tide = tmid, name='bigrise'))
            elif (ta - tb) > 7.5:
                # swing before is big (but not after)
                res.append(create_tide_print(time = rtimes[b], index = b, tide = tb, name='hihi'))
                res.append(create_tide_print(time = rtimes[mid], index = mid, tide = tmid, name='smallrise'))
            else:
                # swing before and after small
                res.append(create_tide_print(time = rtimes[b], index = b, tide = tb, name='hilo'))
                res.append(create_tide_print(time = rtimes[mid], index = mid, tide = tmid, name='smallrise'))
        else:
            G.logger.warning(f"{(ta, tb, tc)} is neither maxima nor minima.")
        delta = datetime.timedelta(hours = 3)    
        fres = [r for r in res if utils.time_after(tstart-delta, r.time) and utils.time_after(r.time, tfinish+delta)]
    return fres

TIDE_PRINT_NAMES = {
    'lolo': 1,
    'bigrise': 2,
    'hihi': 3,
    'smalldrop': 4,
    'hilo': 5,
    'smallrise': 6,
    'lohi': 7,
    'bigdrop': 8
}

def tide_print_path(filename):
    return os.path.join("Data/Currents/TidePrints", filename)

def tide_print_filename(name):
    "Handles both the tide print number (1-8) and 'name'." 
    if isinstance(name, int):
        if name in TIDE_PRINT_NAMES.values():
            index = name + 24
        else:
            raise Exception(f"Invalid tide print name {name}.")
    elif isinstance(name, str):
        if name in TIDE_PRINT_NAMES:
            index = TIDE_PRINT_NAMES[name] + 24
        else:
            raise Exception(f"Invalid tide print name {name}.")
    else:
        raise Exception(f"Invalid tide print name {name}.")
    return f"geo_im-0{index}.tiff"

################################################################

def display_stats(values):
    stats = scipy.stats.describe(values)
    dd = stats._asdict()
    display(f"median = {np.median(values)}")
    for k, v in dd.items():
        display(f"{k} = {v}")

def tide_data_analysis(tides):
    tides = TIDE_DATA.df.prediction
    highs, _ = scipy.signal.find_peaks(tides)
    lows, _ = scipy.signal.find_peaks(-tides)
    extrema = sorted(highs.tolist() + lows.tolist())
    rises = []
    drops = []
    for a, b in pairwise(extrema):
        ta = tides[a]
        tb = tides[b]
        if ta > tb:
            rises.append(ta-tb)
        else:
            drops.append(ta-tb)
    rises = np.array(rises)
    drops = np.array(drops)
    display(f"Tide rise stats.")
    display_stats(rises)
    display(f"Tide drop stats.")
    display_stats(drops)

    
def tide_print_chart(tp_path, region, pixels=1000):
    ch = region.union(dict(proj=G.PROJ4, pixels=pixels))
    ch = chart.gdal_extract_chart(ch, tp_path, "/tmp/mbtile.tif")
    image = cv2.imread(ch.path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ch = ch.union(dict(image=image))

    ch.fig = plt.figure()
    ch.fig.clf()

    ch.ax = ch.fig.add_subplot(111)
    ch = chart.draw_chart(ch)
    return ch


date = "2020-06-12"
uw_model = uw.read_live_ocean_model(date)
region = chart.region_from_marks("nmwnrwn")
pixels = 1000
sail_date = "2020-06-13"
start_time = "11:00:00"
end_time = "14:00:00"
time_indices, times  = uw.find_times(uw_model, f"{sail_date} {start_time}", f"{sail_date} {end_time}")

plt.close('all')
fig_num = 0
tp_path = tide_print_path(tide_print_filename('lohi'))
ch = tide_print_chart(tp_path, region, pixels)
uw.draw_current(ch, uw_model, time_indices[0])
ch.fig.savefig(f"/tmp/fig_{fig_num}.pdf", orientation='portrait')
fig_num += 1

tp_path = tide_print_path(tide_print_filename('lohi'))
ch = tide_print_chart(tp_path, region, pixels)
uw.draw_current(ch, uw_model, time_indices[1])
ch.fig.savefig(f"/tmp/fig_{fig_num}.pdf", orientation='portrait')
fig_num += 1

tp_path = tide_print_path(tide_print_filename('smalldrop'))
ch = tide_print_chart(tp_path, region, pixels)
uw.draw_current(ch, uw_model, time_indices[1])
ch.fig.savefig(f"/tmp/fig_{fig_num}.pdf", orientation='portrait')
fig_num += 1

tp_path = tide_print_path(tide_print_filename('smalldrop'))
ch = tide_print_chart(tp_path, region, pixels)
uw.draw_current(ch, uw_model, time_indices[2])
ch.fig.savefig(f"/tmp/fig_{fig_num}.pdf", orientation='portrait')
fig_num += 1



################################################################
# Pull raw data.  Above we read saved data so that we can survive a change in the service.
# Note units are english!


def test():
    tests = TIDE_DATA.df.iloc[10::10000]

    delta = pd.Timedelta("0s")
    tides = tides_at(tests.date_time + delta)
    errors = (tests.prediction - tides).abs()
    print("Total errors {1} for delta: {0}".format(delta, (errors > 0.001).sum()))

    delta = pd.Timedelta("30s")
    tides = tides_at(tests.date_time + delta)
    errors = (tests.prediction - tides).abs()
    print("Total errors {1} for delta: {0}".format(delta, (errors > 0.001).sum()))

    delta = pd.Timedelta("-50s")
    tides = tides_at(tests.date_time + delta)
    errors = (tests.prediction - tides).abs()
    print("Total errors {1} for delta: {0}".format(delta, (errors > 0.001).sum()))
