"""
Reads race logs and prepares for visualization and analysis.

Also manages metadata for each race.
"""
import os
import itertools as it
import re
import datetime
import pytz
from types import SimpleNamespace

import pandas as pd
import numpy as np

from global_variables import G
import utils
import process as p

# Race Log Metadata is stored in the "log info" dataframe. #############################################

EMPTY_LOG_INFO_ROW = dict(file='', race='', begin=0, end=-1, description='')

def datetime_from_log_filename(filename, time_zone='US/Pacific'):
    "Extracts the datetime from a log filename."
    dt_string = re.sub(r".gz$", "", filename)   # Could be compressed
    dt_string = re.sub(r".pd$", "", dt_string)  # Standard .pd
    dt = datetime.datetime.strptime(dt_string, '%Y-%m-%d_%H:%M')
    return pytz.timezone(time_zone).localize(dt)


def loginfo_new_row(file, race='', begin=0, end=-1, description=''):
    dt = datetime_from_log_filename(file)
    new_row = dict(file=file, datetime=dt, race=race, begin=begin, end=end, description=description)
    return {**EMPTY_LOG_INFO_ROW, **new_row}


def add_log_info_rows(log_info, files):
    new_rows = []
    for f in files:
        print(f"Adding {f}")
        new_rows.append(loginfo_new_row(f))
    return log_info.append(new_rows, ignore_index=True)


def save_updated_log_info(log_info_df):
    "Save an updated log_info DataFrame."

    # Backup the old data (since data is precious)
    backup_file = utils.backup_file(G.LOG_INFO_PATH)
    print(f"Backed up log info data to {backup_file}")

    # Overwrite the original file.
    log_info_df.to_pickle(G.LOG_INFO_PATH)


def read_log_info():
    "Read the current log info dataframe."
    return pd.read_pickle(G.LOG_INFO_PATH)


def find_new_logs(log_info):
    "Let's find new logs that do not yet have an entry."

    # All the known logs (may not include new files)
    known_logs = set(log_info.file)

    # Grab all the log files.
    files = os.listdir(G.LOGS_DIRECTORY)

    # Keep those that are race logs
    race_files = sorted([f for f in files if re.match('^20.*.pd.gz$', f)])

    # Just keep those that are missing
    return [f for f in race_files if f not in known_logs]


def get_log(log_info, filename_prefix):
    "Return a log_info entry that where the provided filename_prefix matches."
    match = log_info.file.str.startswith(filename_prefix)
    matches = log_info[match]
    if len(matches) > 0:
        return log_info[match].iloc[0]
    else:
        return None


# COLUMN CLEANUP ################################################################
# White list of columns that have practical meaning.
COL_WHITE_LIST = [
    'altitude',           'awa',             'aws',
    'cog',                'depth',           'filename',
    'geoidal_separation', 'hdg',             'latitude',
    'longitude',          'rawa',            'raws',
    'rcog',               'rhdg',
    'row_seconds',
    'rsog',
    'rspd',               'rtwa',            'rtwd',
    'rtws',               'rudder',           'sog',
    'spd',                'timestamp',
    'turn_rate',          'twa',             'variation',
    'zeus_altitude',      'zeus_cog',        'zeus_gnss_type',
    'zeus_sog',           'zg100_pitch',     'zg100_roll'
]

COL_RENAME = dict(wind_angle='rawa',
                  wind_speed='raws',
                  sog='rsog',
                  cog='rcog',
                  true_wind_speed='rtws',
                  true_wind_angle='rtwa',
                  true_north_wind_angle='rtwd',
                  heading='rhdg',
                  rudder_position='rudder',
                  speed_water_referenced='rspd')

def df_retain_columns(df, white_list):
    "Drop columns that bring no value.  Ultimately important because the data gets big."
    white_list = set(white_list)
    to_drop = [col for col in df.columns if col not in white_list]
    return df.drop(columns=to_drop)

# Log Reading ######################################################################

def process_sensors(df, causal=False, cutoff=0.3):
    """
    Process sensor data to normalize and smooth.  Note, this is likely done on the boat in
    real-time, but not recorded.

    """
    df.rawa = p.sign_angle(df.rawa)

    aw_n = df.raws * p.cos_d(df.rawa)
    aw_e = df.raws * p.sin_d(df.rawa)

    theta = 0.8
    alpha = 0.97
    saw_n, _ = p.exponential_filter(np.array(aw_n), alpha, theta)
    saw_e, _ = p.exponential_filter(np.array(aw_e), alpha, theta)

    df['awa'] = np.degrees(np.arctan2(saw_e, saw_n))
    df['aws'] = np.sqrt(np.square(saw_e) + np.square(saw_n))

    if 'rtwa' in df.columns:
        df['twa'] = p.sign_angle(df.rtwa)
    if 'rtws' in df.columns:
        df['tws'] = df.rtws
    if 'rtwd' in df.columns:
        df['twd'] = df.rtwd

    # Less noise and smaller theta values.
    df['spd'], _ = p.exponential_filter(np.array(df.rspd), 0.8, 0.5)
    df['sog'], _ = p.exponential_filter(np.array(df.rsog), 0.8, 0.5)

    df['hdg'], _ = p.exponential_filter_angle(np.array(df.rhdg), 0.8, 6)
    df['cog'], _ = p.exponential_filter_angle(np.array(df.rcog), 0.9, 6)


def find_sail_logs(directory):
    "Relies on the extension .pd"
    return sorted([f for f in os.listdir(directory) if re.match(r'.*\.pd', f)])

def compute_record_times(df):
    # There are three sources of time in the logs: 1) infrequent GPS time (which I assume
    # are very accurate but are only updated once per second) 2) per message Raspberry PI
    # timestamps, and 3) per row synthesized timestamps (which are computed from the RPI
    # timestamps while assembling the dataframe).
    #
    # Generally RPI clock is inaccurate (about 60msec per hour or a second per day).  The
    # RPI does *NOT* have a realtime clock.
    #
    # This code tries to track the GPS times while still preserving the frequency of the
    # row updates.
    #
    # Additionally, since it is a feedback loop, it can deal with log drop outs (where the
    # GPS time might change a lot, but the row times do not).
    base_time = df.timestamp.min()
    gps_delta_sec = np.array((df.timestamp - base_time) / pd.Timedelta('1s'))
    gps_delta_diff = np.diff(gps_delta_sec, prepend=0)
    row_delta_sec = np.array(df.row_seconds - df.row_seconds.min())

    # Code attempts to minimize the "error" to infrequent GPS time measurements, by
    # adjusting an offset.  Positive error reduces the offset, negative error increases
    # the offset.
    deltas = np.zeros(row_delta_sec.shape)
    feed_forward = gps_delta_diff.mean()
    offset = 0.0
    proportional = 0.01
    integral = 0.000
    sum_error = 0
    for i in range(1, len(gps_delta_sec)):
        deltas[i] = deltas[i-1] + feed_forward + offset
        # Only compute error if the GPS time has been updated.
        if gps_delta_diff[i] > 0.1:
            error = gps_delta_sec[i] - deltas[i]
            sum_error += error
            offset = proportional * error + integral * sum_error
            # If the error gets large, then increase the tracking gain.
            if np.abs(error) > 10:
                offset = 2.0 * proportional * error + integral * sum_error

    res_sec = deltas * pd.Timedelta('1s')
    df['row_times'] = (np.datetime64(base_time) + res_sec).reshape(-1)
    df.row_times = df.row_times.dt.tz_localize('UTC').dt.tz_convert('US/Pacific')

def compute_record_times_old(df):
    # NOT USED ANY MORE, because it was not robust to log drop outs (it assumed that there
    # was a single global offset between local and GPS time).  Still eduational.
    # 
    # This code computes the drift and offset from GPS to synthesized row stamps.
    # Generally it is good (about 60msec per hour or a second per day).  The RPI does
    # *NOT* have a realtime clock.
    #
    # From this I conclude that one can simply add the row_seconds to the base_time to
    # compute row times.
    base_time = df.timestamp.min()
    gps_times = (df.timestamp - base_time) / pd.Timedelta('1s')
    A = np.vstack([df.row_seconds, np.ones(df.row_seconds.shape)]).T
    b = np.array(gps_times).reshape(-1, 1)
    result = np.linalg.lstsq(A, b, rcond=None)
    fit = result[0]
    offsets = np.dot(A, fit) * pd.Timedelta('1s')
    df['row_times'] = (np.datetime64(base_time) + offsets).reshape(-1)
    df.row_times = df.row_times.dt.tz_localize('UTC').dt.tz_convert('US/Pacific')


def read_log_file(pickle_file, skip_dock_only=True, discard_columns = True,
                  rename_columns=True, trim=True,
                  process=True, cutoff=0.3,
                  path=None):
    """
    SKIP_DOCK_ONLY filters examples where the boat never gets of the dock, otherwise returns None
    DISCARD_COLUMNS drops columns that are not particular helpful
    RENAME_COLUMNS provides more friendly standard names
    TRIM trims off beginning of the log where some values are problematic or missing
    PROCESS proceses the data values to be more useable (originals are preserved)
    CUTOFF is the Butterworth cutoff frequency for smoothing, in Hz.
    PATH is the path to the pickle file.
    """
    if path is None:
        path = G.LOGS_DIRECTORY
    full_path = os.path.join(path, pickle_file) if path else pickle_file
    df = pd.read_pickle(full_path)
    df = df[df[['date', 'time']].isna().sum(1) == 0]
    df['timestamp'] = pd.to_datetime(df.date + " " + df.time)
    session_length = df.timestamp.iloc[-1] - df.timestamp.iloc[0]
    print("Session from {0}, {1} rows, {2} hours.".format(df.timestamp.iloc[0], len(df), session_length.seconds / (60 * 60)))
    if skip_dock_only and df.speed_water_referenced.max() < 0.1:
        print("... skipping, the boat does not move!")
        return None
    # Not sure why cog is often missing
    if 'zeus_cog' in df.columns:
        if 'cog' in df.columns:
            df.cog = df.cog.fillna(value=df.zeus_cog)
        else:
            df['cog'] = df.zeus_cog
    if rename_columns:
        df = df.rename(columns=COL_RENAME)
    if discard_columns:
        df = df_retain_columns(df, COL_WHITE_LIST)
    # trim off the initial startup where data is missing
    if trim:
        df = df[df.isna().sum(1) == 0]
    df.filename = pickle_file
    if process:
        process_sensors(df, causal=False, cutoff=cutoff)
        compute_record_times(df)
    return df


def read_logs(log_entries, skip_dock_only=True, discard_columns = True,
              rename_columns=True, trim=True,
              race_trim=True,
              process=True, cutoff=0.3,
              path=None):
    """
    SKIP_DOCK_ONLY filters examples where the boat never gets of the dock
    CUTOFF is the Butterworth cutoff frequency for smoothing, in Hz.
    """
    dfs = []
    for i, log in zip(it.count(), log_entries):
        pickle_file = log.file
        df = read_log_file(pickle_file,
                           skip_dock_only=skip_dock_only, discard_columns=discard_columns,
                           rename_columns=rename_columns, trim=trim, process=process, cutoff=cutoff,
                           path=path)
        if race_trim:
            df = df.loc[log.begin: log.end].copy()
        ns = SimpleNamespace(**log)
        df.log = ns
        dfs.append(df)
    valid_dfs = [df for df in dfs if df is not None]
    return valid_dfs, pd.concat(valid_dfs, sort=True, ignore_index=True)
