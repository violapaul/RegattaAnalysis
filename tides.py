"""
Loads a cached set of NOAA tide prediction data and then looks up the closest tide for
any time.
"""

import pandas as pd
import numpy as np
import os
import os.path
import urllib

BASE_DIR = '/Users/viola/canlogs/Tides'
TIDES_DF = pd.read_pickle(os.path.join(BASE_DIR, "seattle_tides2.pd"))

BASE_TIME = TIDES_DF.date_time.iloc[0]
LAST_TIME = TIDES_DF.date_time.iloc[-1]


# Utils ################################################################
def normalize_column_names(df):
    df = TIDES_DF
    fold = df.columns.str.lower()
    strip = fold.str.strip()
    under = strip.str.replace(' ', '_')
    df.columns = under
    return df

def fetch_data():
    "Fetch the data from NOAA and create a pandas dataframe."
    predictions_vars = dict(
        station    = '9447130',
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
        print(url)
        df = pd.read_csv(url, parse_dates=True)
        dfs.append(df)

    TIDES_DF = pd.concat(dfs, sort=True, ignore_index=True)
    TIDES_DF = normalize_column_names(TIDES_DF)
    TIDES_DF.date_time = pd.to_datetime(TIDES_DF.date_time)
    TIDES_DF.date_time = TIDES_DF.date_time.dt.tz_localize('UTC').dt.tz_convert('US/Pacific')

    TIDES_DF.to_pickle(os.path.join(BASE_DIR, "seattle_tides2.pd"))


def load_tide_csv():
    # DEPRECATED
    PARSE_ARGS = dict(parse_dates=["Date Time"])

    TIDES_DFS = [
        pd.read_csv(os.path.join(BASE_DIR, 'seattle_noaa_tides_2019.csv'), **PARSE_ARGS),
        pd.read_csv(os.path.join(BASE_DIR, 'seattle_noaa_tides_2020.csv'), **PARSE_ARGS),
        pd.read_csv(os.path.join(BASE_DIR, 'seattle_noaa_tides_2021.csv'), **PARSE_ARGS)
    ]

    TIDES_DF = pd.concat(TIDES_DFS, sort=True, ignore_index=True)
    return normalize_column_names(TIDES_DF)


def tides_at_closest(times):
    if (times < BASE_TIME).any():
        raise Exception("Tide time too early {0}".format(times.min()))
    if (times > LAST_TIME).any():
        raise Exception("Time time too late {0}".format(times.max()))
    tide_index = ((times - BASE_TIME).dt.total_seconds() / 360).round().astype(np.int)
    return TIDES_DF.iloc[tide_index].prediction.values

def tides_at(times):
    if (times < BASE_TIME).any():
        raise Exception("Tide time too early {0}".format(times.min()))
    if (times > LAST_TIME).any():
        raise Exception("Time time too late {0}".format(times.max()))
    delta_sec = np.asarray((times - BASE_TIME).dt.total_seconds())
    delta_step = delta_sec / 360  # One prediction every 6 minutes
    before = np.floor(delta_step).astype(np.int32)
    after = np.ceil(delta_step).astype(np.int32)
    alpha = delta_step - before
    val = (1 - alpha) * np.asarray(TIDES_DF.iloc[before].prediction)
    val += alpha * np.asarray(TIDES_DF.iloc[after].prediction)
    return val


################################################################
# Pull raw data.  Above we read saved data so that we can survive a change in the service.
# Note units are english!

def raw_data():
    "Here mostly for documentation."

    # The Seattle Station is 9447130.  The next nearest are Port Townsend: 9444900 and
    # Tacoma 9446484.

    # Pull 2019 tide heights (measured, rather than predicted).
    tide_height_vars = dict(
        begin_date = '20190101',
        end_date   = '20191104',
        station    = '9447130',
        product    = 'hourly_height',
        datum      = 'MLLW',
        time_zone  = 'lst',  # local time that include day light savings (likely a
                             # mistake)
        units      = 'english',
        format     = 'csv'
    )

    BASE_URL = 'https://tidesandcurrents.noaa.gov/api/datagetter?'
    height_df = pd.read_csv(BASE_URL + urllib.parse.urlencode(tide_height_vars),
                            index_col=0, parse_dates=True)

    # Pull the tide predictions
    predictions_vars = dict(
        station    = '9447130',
        product    = 'predictions',
        datum      = 'MLLW',
        time_zone  = 'lst',
        units      = 'english',
        format     = 'csv'
    )

    predictions_vars.update(begin_date='20190101', end_date='20191231')
    predictions_vars.update(begin_date='20200101', end_date='20201231')
    predictions_vars.update(begin_date='20210101', end_date='20211231')

    predictions_df = pd.read_csv(BASE_URL + urllib.parse.urlencode(predictions_vars),
                                 index_col=0, parse_dates=True)


def test():
    tests = TIDES_DF.iloc[10::10000]

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
