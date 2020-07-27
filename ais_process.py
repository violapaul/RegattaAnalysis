# Load some libraries

import copy
import itertools as it
import datetime

import numpy as np
import scipy
import scipy.interpolate
import pandas as pd
import arrow

from global_variables import G
import canboat as cb
import chart as c
import process as p

# Helpful constancts for NMEA 2K PGNs
PGN_GNSS         = cb.pgn_code('GNSS Position Data')
PGN_AIS_POSITION = 129039  # AIS Class B Position Report
PGN_AIS_DATA     = 129809  # AIS Class B static data (msg 24 Part A)

MIN_DELTA_TIME = datetime.timedelta(microseconds = 10000)  # 10 milliseconds
MAX_VALID_DISTANCE = 100000  # 100 kilometers 

def ais_dataframes(records):
    "Return two dataframes, one that records the positions of all AIS craft.  The second that records the names."
    # The only thing that makes this a bit tricky is getting the correct GPS times.  The
    # row timestamps are incorrect, so we need to wait for a GNSS record to go by, and
    # then determines the true GPS time.
    pgns_to_collect = {PGN_AIS_POSITION, PGN_AIS_DATA}
    pgn_rows = {k:[] for k in pgns_to_collect}
    delta_time = None
    for record_num, record in zip(it.count(), records):
        if record_num % 500000 == 0:
            G.logger.info(f"Processed {record_num} json lines.")
        pgn = record['pgn']
        src = record['src']
        if delta_time is None and pgn == PGN_GNSS:
            # Row time (from Raspberry PI)
            timestamp = arrow.get(record['timestamp'])
            # GPS time, from the GNSS record.
            gnss_datetime = cb.convert_gnss_date_time(record['fields']['Date'],
                                                      record['fields']['Time'])
            
            # difference between the two clocks
            delta_time = gnss_datetime - timestamp
            previous_timestamp = timestamp
            G.logger.info(f"Found GNSS record at {gnss_datetime}")
            G.logger.info(f"Delta time is {delta_time}.")
        elif (pgn in pgns_to_collect) and (delta_time is not None): 
            # Wait until we've seen at least one GNSS message
            row = copy.copy(record['fields'])
            timestamp = arrow.get(record['timestamp'])
            # Had a weird special cases where the same timestamp appeared twice.  Let's just skip.
            if (timestamp - previous_timestamp) > MIN_DELTA_TIME:
                previous_timestamp = timestamp
                row['src'] = src
                row['timestamp'] = timestamp.datetime
                row['row_times'] = (timestamp + delta_time).datetime
                pgn_rows[pgn].append(row)
    res = {}
    for pgn, rows in pgn_rows.items():
        df = pd.DataFrame(rows)
        # Rename columns so that they make sense
        res[pgn] = df.rename(columns=cb.canonical_field_name)
    return res

def json_to_ais_data_frame(json_file, max_rows=None):
    "Read the JSON log file and produce a dataframe contains all lat/lon fixes contained in the AIS logs."
    records = cb.json_records( cb.file_lines(json_file), None, 1)
    records = it.islice(records, 0, max_rows)

    dfs = ais_dataframes(records)

    df_pos = dfs[PGN_AIS_POSITION]
    # Just grab the critical columns, and then remove dups
    df_data = dfs[PGN_AIS_DATA][['user_id', 'name']].drop_duplicates()
    # Join the two dataframes, adds the name column to the position data
    df_joined = df_pos.merge(df_data, on='user_id')
    # Sort by row_times, seems natural
    return df_joined.sort_values(by=['row_times'])

def clean_ais_df(df):
    """
    Test that the AIS data is valid.  (Not perfect!)

    Sometimes the data extracted from AIS is corrupt.  Missing values (NaN) and corrupt values.
    """
    df = df[df.isna().sum(1) == 0]
    locations = np.vstack(G.MAP(np.asarray(df.longitude), np.asarray(df.latitude))).T
    # We will reject distances that are too far from the center of our map.
    is_valid = np.all(np.logical_and((locations < MAX_VALID_DISTANCE), (locations > -MAX_VALID_DISTANCE)), axis=1)
    return df[is_valid]

def fit_spline(df, plot=False):
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

    # Create a new set of spline points, every 5 seconds.
    x_new = np.linspace(x.min(), x.max(), int((x.max() - x.min())/5))
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

def row_to_lla(index_row):
    _, row = index_row
    return dict(timestamp = arrow.get(row.row_times),
                lat = row.latitude,
                lon = row.longitude,
                alt = row.altitude)

def ais_df_to_gpx(df, gpx_file, skip=1):
    lla_rows = map(row_to_lla, it.islice(df.iterrows(), 0, None, skip))
    gpx = cb.lla_to_gpx(lla_rows)
    G.logger.info(f"Writing GPX result to {gpx_file}")
    with open(gpx_file, 'w') as gpx_fs:
        gpx_fs.write(gpx.to_xml())


def extract_ais():

    
    p.json_from_matching_named_file
