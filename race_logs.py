"""
Reads race logs and prepares for visualization and analysis.

Also manages metadata for each race.
"""
import os
import itertools as it
import re
from types import SimpleNamespace

from numba import jit

import pandas as pd
import numpy as np
import scipy

from global_variables import G
from utils import DictClass
import process as p
import metadata


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

# Pre-processing of Log Data ##################################################################

@jit(nopython=True)
def estimate_true_wind_helper(epsilon, aws, awa, hdg, spd, cog, sog,
                              tws_init, twd_init, variation, tws_mult, time_delay):
    "Documented in True_Wind.ipynb"
    # TWD/TWS are ouptputs.  This sets the initial conditions.
    ptwd = np.radians(twd_init) + np.zeros(awa.shape)
    ptws = tws_init + np.zeros(awa.shape)
    # Residuals are stored here.
    res_n = np.zeros(awa.shape)
    res_e = np.zeros(awa.shape)
    deps = np.zeros(awa.shape)
    # Process apparent wind to decompose into boat relative components.
    aw_n = aws * np.cos(np.radians(awa))
    aw_e = aws * np.sin(np.radians(awa))

    # preconvert some values
    rhdg = np.radians(hdg)
    rcog = np.radians(cog)
    variation = np.radians(variation)
    
    eps = epsilon
    for i in range(1, len(aws)-time_delay):
        # Transform to boat relative angles.
        twa = ptwd[i-1] - (rhdg[i] + variation)
        course_angle = rcog[i+time_delay]  - (rhdg[i] + variation)
        
        # Useful below
        c = np.cos(twa)
        s = np.sin(twa)
        
        # Boat relative vector of true wind
        twn = c * ptws[i-1]
        twe = s * ptws[i-1]
        
        # Boat relative vector of travel
        btn = np.cos(course_angle) * sog[i+time_delay]
        bte = np.sin(course_angle) * sog[i+time_delay]
        
        # The forward predictions, introduce leeway
        f_aw_n = twn + btn
        f_aw_e = twe - bte  # WHY IS THIS NEGATIVE???  SEEMS INCORRECT

        # Residuals
        res_n[i] = (aw_n[i] - f_aw_n)
        res_e[i] = (aw_e[i] - f_aw_e)

        # derivatives
        delta_tws = res_n[i] * c + res_e[i] * s
        delta_twd = res_n[i] * ptws[i-1] * -s + res_e[i] * ptws[i-1] * c
        
        # The mathematics allows for solutions were tws is negative, particularly when the
        # wind directions switches rapidly.  Need to ensure its positive.
        ptws[i] = np.absolute(eps * tws_mult * delta_tws + ptws[i-1])
        ptwd[i] = eps * delta_twd + ptwd[i-1]

        # Check the current residuals.  If they are too large then gradually decrease the
        # smoothness (increase epsilon).
        res_error = (np.abs(res_n[i]) + np.abs(res_e[i]))
        if res_error > 1.0 and res_error < 5.0:
            eps = min(1.05 * eps, 10*epsilon)
        elif res_error > 5.0:
            eps = min(1.1 * eps, 1000*epsilon)
        else:
            eps = (eps + epsilon) / 2
        deps[i] = eps

    return np.degrees(ptwd), ptws, res_n, res_e, deps


def estimate_true_wind(epsilon, df, awa_mult=1.0, aws_mult=1.0, spd_mult=1.0, awa_offset=0, tws_mult=30, time_delay=12):
    variation = df.variation.mean()
    tws_init = df.aws.iloc[0]
    twd_init = df.rawa.iloc[0] + df.rhdg.iloc[0] + variation
    (twd, tws, res_n, res_e, deps) = estimate_true_wind_helper(epsilon,
                                                         aws = aws_mult * np.asarray(df.caws),
                                                         awa = awa_mult * np.asarray(df.cawa) + awa_offset,
                                                         hdg = np.asarray(df.rhdg),
                                                         spd = spd_mult * np.asarray(df.rspd),
                                                         cog = np.asarray(df.rcog),
                                                         sog = np.asarray(df.rsog),
                                                         tws_init = tws_init,
                                                         twd_init = twd_init,
                                                         variation = variation,
                                                         tws_mult = tws_mult,
                                                         time_delay = time_delay)
    df['twd'] = p.compass_angle(twd)
    df['tws'] = tws
    df['twa'] = p.sign_angle(twd - (df.rhdg + variation))
    return deps

    
def heel_corrected_apparent_wind(df):
    """
    Heel (aka roll) introduces motion at the masthead, which can in turn impact AWA.
    Correct this.  See True_Wind.ipynb
    """

    # Compute a smoothed version of heel.  Note the cutoff is set to 0.3 Hz.  Since the
    # signal is sampled at 1 Hz.  Nyquist says we cannot hope measure any signal higher
    # than 0.5 Hz.
    coeff = p.butterworth_filter(cutoff=0.3, order=5)

    # Smooth the heel signal.
    # Note causal is false, which will filter the signal but not introduce any delay.
    sheel = p.smooth_angle(coeff, df.zg100_roll, causal=False)

    # Take the derivative
    heel_rate = G.SAMPLES_PER_SECOND * np.diff(sheel, prepend=sheel[0])

    mast_vel = G.MAST_HEIGHT * np.radians(heel_rate)

    aw_n = df.raws * p.cos_d(df.rawa)
    aw_e = df.raws * p.sin_d(df.rawa)

    # Corrected
    caw_e = aw_e - mast_vel

    # Unsmoothed
    df['cawa'] = np.degrees(np.arctan2(caw_e, aw_n))
    df['caws'] = np.sqrt(np.square(aw_n) + np.square(caw_e))

    # Compute smoothed versions
    theta = 0.8
    alpha = 0.97
    saw_n, _ = p.exponential_filter(np.array(aw_n), alpha, theta)
    scaw_e, _ = p.exponential_filter(np.array(caw_e), alpha, theta)

    df['scawa'] = np.degrees(np.arctan2(scaw_e, saw_n))
    df['scaws'] = np.sqrt(np.square(saw_n) + np.square(scaw_e))


def process_sensors(df, causal=False, cutoff=0.3, compute_true_wind=True):
    """
    Process sensor data to normalize and smooth.  Note, this done on the boat, before
    display, in real-time, but not logged.  We do our best to simulate that here.
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

    heel_corrected_apparent_wind(df)

    # If TWA/TWD/TWS is available from the boat, grab it.
    if 'rtwa' in df.columns:
        df['boat_twa'] = p.sign_angle(df.rtwa)
        df['twa'] = p.sign_angle(df.rtwa)
        df['rtwa'] = p.sign_angle(df.rtwa)        
    if 'rtws' in df.columns:
        df['boat_tws'] = df.rtws
        df['tws'] = df.rtws
    if 'rtwd' in df.columns:
        df['boat_twd'] = df.rtwd
        df['twd'] = df.rtwd

    # Replace with computed quantity
    estimate_true_wind(0.00001, df, tws_mult=16)
    df['stwd'] = p.compass_angle(df.twd)
    df['stws'] = df.tws
    df['stwa'] = df.twa
    
    # Replace with computed quantity
    estimate_true_wind(0.0001, df, tws_mult=16)
    df['twd'] = p.compass_angle(df.twd)

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
    
    # These are GPS times, which are updated only every second
    base_time = df.timestamp.min()
    gps_delta_sec = np.array((df.timestamp - base_time) / pd.Timedelta('1s'))  # Conversion to seconds
    gps_delta_diff = np.diff(gps_delta_sec, prepend=0)  # Value is non-zero if there is an update.

    # RPi time stamps, which are inaccurate but frequent!
    row_delta_sec = np.array(df.row_seconds - df.row_seconds.min())

    # Code attempts to minimize the "error" to infrequent GPS time measurements, by
    # adjusting an offset.  Positive error reduces the offset, negative error increases
    # the offset.
    deltas = np.zeros(row_delta_sec.shape)  # deltas which will result.
    feed_forward = gps_delta_diff.mean()
    offset = 0.0  # Offset between the RPi clock and the GPS clock
    # PID gains for adjusting the offset
    proportional = 0.01
    integral = 0.000
    sum_error = 0
    for i in range(1, len(gps_delta_sec)):
        deltas[i] = deltas[i-1] + feed_forward + offset
        # Only compute error if the GPS time has been updated.
        if gps_delta_diff[i] > 0.1:
            error = gps_delta_sec[i] - deltas[i]  # First is accurate, second is frequent
            sum_error += error
            offset = proportional * error + integral * sum_error  # Offset is adjusted to keep them aligned.
            # If the error gets large, then increase the tracking gain.
            if np.abs(error) > 10:
                offset = 2.0 * proportional * error + integral * sum_error

    res_sec = deltas * pd.Timedelta('1s')
    # Convert back to times and set the correct timezone.
    df['row_times'] = (np.datetime64(base_time) + res_sec).reshape(-1)
    df.row_times = df.row_times.dt.tz_localize('UTC').dt.tz_convert('US/Pacific')


def read_log_file(pickle_file, skip_dock_only=True, discard_columns = True,
                  rename_columns=True, trim=True,
                  process=True, cutoff=0.3,
                  path=None, compute_true_wind=True):
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
        path = G.PANDAS_LOGS_DIRECTORY
    full_path = os.path.join(path, pickle_file) if path else pickle_file
    df = pd.read_pickle(full_path)
    df = df[df[['date', 'time']].isna().sum(1) == 0]
    df['timestamp'] = pd.to_datetime(df.date + " " + df.time)
    session_length = df.timestamp.iloc[-1] - df.timestamp.iloc[0]
    hours = session_length.seconds / (60 * 60)
    G.logger.info(f"Session from {df.timestamp.iloc[0]}, {len(df)} rows, {hours} hours.")
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
    clean_gps_outliers(df)
    df.filename = pickle_file
    if process:
        process_sensors(df, causal=False, cutoff=cutoff, compute_true_wind=compute_true_wind)
        compute_record_times(df)
    return df

def clean_gps_outliers(df):
    # Every now and then you see crazy dropouts in GPS
    smooth_longitude = scipy.signal.medfilt(df.longitude, 5)
    smooth_latitude = scipy.signal.medfilt(df.latitude, 5)
    if np.max(np.abs(df.longitude - smooth_longitude)) > 0.1:
        G.logger.info(f"Noise in longitude, using median filter.")
        df.longitude = smooth_longitude
    if np.max(np.abs(df.latitude - smooth_latitude)) > 0.1:
        G.logger.info(f"Noise in latitude, using median filter.")
        df.latitude = smooth_latitude


def read_logs(log_entries, race_trim=True, **args):
    """
    SKIP_DOCK_ONLY filters examples where the boat never gets of the dock
    CUTOFF is the Butterworth cutoff frequency for smoothing, in Hz.
    """
    dfs = []
    for i, log in zip(it.count(), log_entries):
        G.logger.info(f"Reading file {log.file}")
        pickle_file = log.file
        df = read_log_file(pickle_file, **args)
        G.logger.info(f"Found {len(df)} records before trim.")
        if race_trim:
            df = df.iloc[log.begin: log.end].copy()
            G.logger.info(f"Trimming to {log.begin} {log.end}")
        # Weird requirement that you can't add a dict to a Dataframe
        # https://stackoverflow.com/a/54137536
        ns = SimpleNamespace(**log)
        df.log = ns
        dfs.append(df)
    valid_dfs = [df for df in dfs if df is not None]
    return valid_dfs, pd.concat(valid_dfs, sort=True, ignore_index=True)

def read_date(date, **args):
    dfs, races, bigdf = read_dates([date], **args)
    if len(dfs) < 1:
        raise Exception(f"Date not found {date}.")
    return dfs[0], races[0]

def read_dates(dates, **args):
    "Convience function.  Given a list of text dates YYYY-MM-DD, will return races on those dates."
    md = metadata.read_metadata()
    races = []
    for date in dates:
        log = md.dates.get(date, None)
        if log is None:
            print(f"Warning, {date} could not be found in the race logs.")
        else:
            # Casting to a DictClass for convenience
            races.append(DictClass(**log))
    df, big_df = read_logs(races, path=G.PANDAS_LOGS_DIRECTORY, **args)
    return df, races, big_df

def trim_race(race, begin, end):
    "Trim the race data by specifying the begin/end.  Used to trim away time at the dock, etc."
    race.begin = begin
    race.end = end
    # casting back to a dict to support YAML serialization
    race_dict = dict(**race)
    metadata.update_race(race_dict)
