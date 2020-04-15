# Load some libraries

import logging
import copy
import itertools as it

import numpy as np
import scipy
import scipy.interpolate
import pandas as pd
import arrow

import global_variables
G = global_variables.init_seattle()

import canboat as cb
import chart as c
import process as p

# Helpful constancts for NMEA 2K PGNs
PGN_GNSS         = cb.pgn_code('GNSS Position Data')
PGN_AIS_POSITION = 129039  # AIS Class B Position Report
PGN_AIS_DATA     = 129809  # AIS Class B static data (msg 24 Part A)


def ais_dataframes(records):
    "Return two dataframes, one that records the positions of all AIS craft.  The second that records the names."
    pgns_to_collect = {PGN_AIS_POSITION, PGN_AIS_DATA}
    pgn_rows = {k:[] for k in pgns_to_collect}
    gnss_datetime = None
    for record_num, record in zip(it.count(), records):
        if record_num % 50000 == 0:
            logging.info(f"Processed {record_num} json lines.")
        pgn = record['pgn']
        if pgn == PGN_GNSS:
            gnss_timestamp = arrow.get(record['timestamp'])
            gnss_datetime = cb.convert_gnss_date_time(record['fields']['Date'],
                                                      record['fields']['Time'])
        elif pgn in pgns_to_collect and gnss_datetime is not None:
            # Wait until we've seen at least one GNSS message
            row = copy.copy(record['fields'])
            timestamp = arrow.get(record['timestamp']).datetime
            row['timestamp'] = timestamp
            delta_time = timestamp - gnss_timestamp
            gps_datetime = gnss_datetime + delta_time
            row['gnss_datetime'] = gnss_datetime
            row['row_times'] = gps_datetime.datetime
            pgn_rows[pgn].append(row)
    res = {}
    for pgn, rows in pgn_rows.items():
        df = pd.DataFrame(rows)
        df = df.rename(columns=cb.canonical_field_name)
        res[pgn] = df
    return res

def json_to_ais_data_frame(json_file, max_rows=None):
    "Read the JSON log file and produce a dataframe contains all lat/lon fixes contained in the AIS logs."
    records = cb.json_records( cb.file_lines(json_file), None, 1)
    records = it.islice(records, 0, max_rows)

    dfs = ais_dataframes(records)

    df_pos = dfs[PGN_AIS_POSITION]
    df_data = dfs[PGN_AIS_DATA][['user_id', 'name']].drop_duplicates()
    df_joined = df_pos.merge(df_data, on='user_id')
    return df_joined

def fit_spline(df, plot=False):
    basetime = df.row_times.min()
    x = np.array((df.row_times - basetime) / pd.Timedelta('1s'))
    y = np.vstack(G.MAP(np.asarray(df.longitude), np.asarray(df.latitude))).T
    vog_latitude = df.sog * p.cos_d(df.cog)
    vog_longitude = df.sog * p.sin_d(df.cog)
    dy = np.vstack((vog_longitude, vog_latitude)).T
    ss = scipy.interpolate.CubicHermiteSpline(x, y, dy)
    x_new = np.linspace(x.min(), x.max(), int((x.max() - x.min())/5))
    ll = ss(x_new)
    lon_new, lat_new = G.MAP(ll[:,0], ll[:,1], inverse=True)
    datetime64 = np.datetime64(basetime.tz_convert('UTC').tz_localize(None))
    time_new = datetime64 + x_new * pd.Timedelta('1s')
    ndf = pd.DataFrame(dict(row_times=time_new, latitude = lat_new, longitude = lon_new))
    ndf.row_times = ndf.row_times.dt.tz_localize('UTC').dt.tz_convert('US/Pacific')
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
    logging.info(f"Writing GPX result to {gpx_file}")
    with open(gpx_file, 'w') as gpx_fs:
        gpx_fs.write(gpx.to_xml())


    

json_path = p.json_from_matching_named_file("2020-04-12")
df = json_to_ais_data_frame(json_path, None) # 100000)
df_creative = df[df.name=='CREATIVE']
sdf = fit_spline(df_creative, True)
ais_df_to_gpx(sdf, "/tmp/foo.gpx")
