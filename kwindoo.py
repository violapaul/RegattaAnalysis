"""
Code to download and process boat tracking data from the Kwindoo service.
"""

import requests
import json
import pandas as pd
from matplotlib import pyplot as plt

import numpy as np
import scipy
import scipy.interpolate

from global_variables import G
import chart as c
import process as p

G.init_seattle()

column_names = dict(
    l = 'latitude',
    o = 'longitude',
    u = 'boat_id',
    t = 'timestamp',
    i = 'some_other_timestamp',
    b = 'cog',
    s = 'sog',
    y = 'battery',
    a = 'mystery',
    datetime = 'datetime'
)

def boat_data(kwindoo_boat_data_response):
    "Create a Dataframe from the Kwindoo boat data (id name pairs)."
    json_response = json.loads(kwindoo_boat_data_response.text)

    # Data for all "users" (really boats)
    data = json_response['response']['users']

    rows = []
    for user_data in data:
        row = {}
        for k in "id first_name last_name".split():
            row[k] = user_data[k]
        boat_data = user_data['boat_data']
        for k in "boat_name".split():
            row[k] = boat_data[k]        
        rows.append(row)

    if len(rows) > 0:
        return pd.DataFrame(rows)
    else:
        return None


def tracking_data(kwindoo_locations_response):
    "Create a Dataframe from the kwindoo tracking data."
    json_response = json.loads(kwindoo_locations_response.text)
    data = json_response['response']['tracking_locations']

    rows = []
    # Data for each boat is stored under that boats id, as the key
    for boat_id in data.keys():
        for pt in data[boat_id]:  # Array of data points
            pt['id'] = boat_id
            rows.append(pt)

    if len(rows) > 0:
        df = pd.DataFrame(rows)
        datetime = pd.to_datetime(df['t'],unit='s')
        df['row_times'] = datetime.dt.tz_localize('UTC').dt.tz_convert('US/Pacific')
        
        return df.rename(columns=column_names)
    else:
        return None


def fetch_locations(raceId, fromTimestamp, toTimestamp):
    "Fetch the location information from Kwindoo, for a particular race and timeframe."
    url_params = dict(
        stream = 'archive',
        raceId = raceId,
        fromTimestamp = fromTimestamp,
        toTimestamp = toTimestamp
    )
    r = requests.get('https://api.kwindoo.com/tracking/get-locations', params=url_params)
    G.logger.info(f"Fetch, status {r.status_code}, {r.url}")
    G.logger.info(f"Response length {len(r.text)}")
    return r


def fetch_boat_data(raceId):
    "Fetch the boat data information from Kwindoo."
    # example_boat_data_url = 'https://api.kwindoo.com/tracking/get-boat-data?raceId=27057'    
    url_params = dict(
        raceId = raceId,
    )
    r = requests.get('https://api.kwindoo.com/tracking/get-boat-data', params=url_params)
    G.logger.info(f"Fetch, status {r.status_code}, {r.url}")
    G.logger.info(f"Response length {len(r.text)}")
    return r


def fit_spline(df, basetime=None, plot=False, sample_seconds=5):
    if basetime is None:
        basetime = df.row_times.min()
    # Get the times, in seconds.
    x = np.array((df.row_times - basetime) / pd.Timedelta('1s'))
    # The dependent variable is position.  Use map position rather than lat/lon
    y = np.vstack(G.MAP(np.asarray(df.longitude), np.asarray(df.latitude))).T
    # Decompose the derivative into two components, north and east
    vog_north = df.sog * p.cos_d(df.cog)
    vog_east = df.sog * p.sin_d(df.cog)
    # Combine into a single matrix
    dy = np.vstack((vog_east, vog_north)).T

    # Compute spline
    cubic_spline = scipy.interpolate.CubicHermiteSpline(x, y, dy)

    # Create a new set of spline points, every N seconds.
    x_new = np.linspace(0, x.max(), int((x.max() - 0) / sample_seconds))
    loc_new = cubic_spline(x_new)
    # Convert back to lat/lon
    lon_new, lat_new = G.MAP(loc_new[:,0], loc_new[:,1], inverse=True)

    # datetime conversions (painful)
    datetime64 = np.datetime64(basetime.tz_convert('UTC').tz_localize(None))
    time_new = datetime64 + x_new * pd.Timedelta('1s')

    # New DataFrame
    ndf = pd.DataFrame(dict(row_times=time_new, latitude = lat_new, longitude = lon_new))
    ndf.row_times = ndf.row_times.dt.tz_localize('UTC').dt.tz_convert('US/Pacific')
    # Current GPX needs an altitude, make one up!
    ndf['altitude'] = 0
    if plot:
        ch = c.plot_chart(ndf)
        c.draw_track(ndf, ch)
        c.draw_track(df, ch, color='red', linestyle = 'None', marker='.')
    return ndf


def test3():
    raceId = 27057    
    r = fetch_boat_data(raceId)
    bd = json.loads(r.text)
    dfb = boat_data(r)

    file = "Data/OneOff/2021_RTTS_boat_data.gz"
    if False:
        dfb.to_pickle(file)
    else:
        dfb = pd.read_pickle(file)
        

def test1():
    # test for the round the sound race
    raceId = 27057
    start_time = 1622311200
    # end_time = 1622311200 + 48 * 60 * 60
    offset = 0
    length = 32 * 60 * 60
    r = fetch_locations(raceId, start_time+offset, start_time+offset+length)
    df = tracking_data(r)
    df.latitude = df.latitude.astype(float)
    df.longitude = df.longitude.astype(float)

    file = "Data/OneOff/2021_RTTS_tracking.gz"
    if False:
        df.to_pickle(file)
    else:
        df = pd.read_pickle(file)

    df = df[['boat_id', 'timestamp', 'latitude', 'longitude',
             'cog', 'sog', 'row_times']]

    # Select just the boats of interest
    df_jubilee = df[df.boat_id == 43409]
    df_pg = df[df.boat_id == 43383]

    # get started at the same time
    basetime = min(df_pg.row_times.iloc[0], df_jubilee.row_times.iloc[0])    

    # Fit a spline to deal with sampling issues.
    df_jubilee = fit_spline(df_jubilee, basetime=basetime, sample_seconds=1)
    df_pg      = fit_spline(df_pg, basetime=basetime, sample_seconds=1)

    # Select the time of interest
    length = 30 * 60 * 60  # 5 hours
    offset = 0 * length  
    sl = slice(offset,offset+length)

    # Slice down to the time of interst
    sdf_jubilee = df_jubilee.iloc[sl]
    sdf_pg = df_pg.iloc[sl]    
    
    ch = c.plot_chart(sdf_pg)
    c.draw_track(sdf_pg, ch, color='green')
    c.draw_track(sdf_pg.iloc[::500], ch, color='green', linestyle = 'None', marker='o')
    c.draw_track(sdf_jubilee, ch, color='red')
    c.draw_track(sdf_jubilee.iloc[::500], ch, color='red', linestyle = 'None', marker='o')


def initial_test():
    example_boat_data_url = 'https://api.kwindoo.com/tracking/get-boat-data?raceId=27057'
    example_locations_url = 'https://api.kwindoo.com/tracking/get-locations?stream=archive&raceId=27057&fromTimestamp=1622311020&toTimestamp=1622312220'

    
    
