"""
Great resource for historical weather data.

Data is typically hourly for the year.  Challenges include missing data, and complexity
around historical notions of wind (e.g.  wind can be listed as "calm" or "variable").

My hope is to build a service where you can lookup the wind at a particular day/time.
This would help analysis of races where we did not have wind data (e.g. Navionics).  And
it's interesting historical data (looking for correlations and patterns).

Search page.

    https://www.ncei.noaa.gov/access/search/data-search/global-hourly?startDate=%5B%222018-12-01T00:00:00%22%5D&endDate=%5B%222019-06-01T23:59:59%22%5D&stations=%5B%2299435099999%22%5D

This is the file for West Point in Seattle.

    https://www.ncei.noaa.gov/data/global-hourly/access/2019/99435099999.csv

Helpful doc here:

    https://www.visualcrossing.com/blog/how-we-process-integrated-surface-database-historical-weather-data

For example:

The value of 'WND' is 170,1,N,0015,1. WND represents wind measurements of the form
direction,direction quality,observation type, speed, speed quality. Again the speed is
multipled by 10 to avoid using decimal points in the file format. Therefore this record
indicates that the wind direction was 170 degrees with a speed of 1.5 meters per second
(approximately 3.4mph or 5.4 kph).
"""

import os

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

import global_variables as G

pd.set_option('display.max_rows', 100)
pd.reset_option('precision')
pd.set_option('display.float_format', lambda x: '%.3f' % x)

np.set_printoptions(precision=4, suppress=True, linewidth = 180)

REPORT_TYPES = [
    dict(type='AERO', description='Aerological report'),
    dict(type='AUST', description='Dataset from Australia'),
    dict(type='AUTO', description='Report from an automatic station'),
    dict(type='BOGUS', description='Bogus report'),
    dict(type='BRAZ', description='Dataset from Brazil'),
    dict(type='COOPD', description='US Cooperative Network summary of day report'),
    dict(type='COOPS', description='US Cooperative Network soil temperature report'),
    dict(type='CRB', description='Climate Reference Book data from CDMP'),
    dict(type='CRN05', description='Climate Reference Network report, with 5-minute reporting interval'),
    dict(type='CRN15', description='Climate Reference Network report, with 15-minute reporting interval'),
    dict(type='FM-12', description='SYNOP Report of surface observation form a fixed land station'),
    dict(type='FM-13', description='SHIP Report of surface observation from a sea station'),
    dict(type='FM-14', description='SYNOP MOBIL Report of surface observation from a mobile land station'),
    dict(type='FM-15', description='METAR Aviation routine weather report'),
    dict(type='FM-16', description='SPECI Aviation selected special weather report'),
    dict(type='FM-18', description='BUOY Report of a buoy observation'),
    dict(type='GREEN', description='Dataset from Greenland'),
    dict(type='MESOH', description='Hydrological observations from MESONET operated civilian or government agency'),
    dict(type='MESOS', description='MESONET operated civilian or government agency'),
    dict(type='MESOW', description='Snow observations from MESONET operated civilian or government agency'),
    dict(type='MEXIC', description='Dataset from Mexico'),
    dict(type='NSRDB', description='National Solar Radiation Data Base'),
    dict(type='PCP15', description='US 15-minute precipitation network report'),
    dict(type='PCP60', description='US 60-minute precipitation network report'),
    dict(type='S-S-A', description='Synoptic, airways, and auto merged report'),
    dict(type='SA-AU', description='Airways and auto merged report'),
    dict(type='SAO', description='Airways report (includes record specials)'),
    dict(type='SAOSP', description='Airways special report (excluding record specials)'),
    dict(type='SHEF', description='Standard Hydrologic Exchange Format'),
    dict(type='SMARS', description='Supplementary airways station report'),
    dict(type='SOD', description='Summary of day report from U.S. ASOS or AWOS station'),
    dict(type='SOM', description='Summary of month report from U.S. ASOS or AWOS station'),
    dict(type='SURF', description='Surface Radiation Network report'),
    dict(type='SY-AE', description='Synoptic and aero merged report'),
    dict(type='SY-AU', description='Synoptic and auto merged report'),
    dict(type='SY-MT', description='Synoptic and METAR merged report'),
    dict(type='SY-SA', description='Synoptic and airways merged report'),
    dict(type='WBO', description='Weather Bureau Office'),
    dict(type='WNO', description='Washington Naval Observatory'),
    dict(type='99999', description='Missing')
]


FILES = [
    "72290023188.csv",
    "72290693112.csv",
    "99435099999.csv"
]

WEATHER_DIR = os.path.join(G.DATA_DIRECTORY, "NCEI_WEATHER")

def process_raw_data(df):
    # lowercase is nicer
    df.columns = df.columns.str.casefold()
    df = df[df.report_type.isin(set(('FM-13', 'FM-15')))]

    # Get the times to work right
    # UTC is used, and then let's use Pacific time
    df['timestamp'] = pd.to_datetime(df.date).dt.tz_localize('UTC').dt.tz_convert('US/Pacific')

    # Extract the wind data, which is in its own CSV columns
    wdf = df.wnd.str.split(',', expand=True)
    wdf.columns = 'wnd_dir wnd_dir_quality wnd_dir_type wnd_spd wnd_spd_quality'.split()
    wdf.wnd_dir = pd.to_numeric(wdf.wnd_dir)
    # Speed is an integer with a scaling factor of 10
    wdf.wnd_spd = pd.to_numeric(wdf.wnd_spd)/10.0
    return pd.concat((df, wdf), axis=1)

def read_weather(files):
    dfs = []
    for file in files:
        df = pd.read_csv(os.path.join(WEATHER_DIR, file))
        df = process_raw_data(df)
        dfs.append(df)
    return dfs

def find_weird_times(df):
    # Every now and then the times skips a beat.  Normally data is hourly.
    ddd = df.timestamp.diff()
    td_max = pd.Timedelta('0 days 01:05:00')
    td_min = pd.Timedelta('0 days 00:55:00')
    iii = ddd.index[(ddd < td_min) | (ddd > td_max)]
    print(f"Found {len(iii)} occurences of skips.")
    for i in iii[1:]:
        print('**********************')
        print(i)
        print(compact(df.shift(1).loc[i]))
        print(compact(df.loc[i]))

def compact(df):
    return df["station date report_type wnd".split()]

df = extract_wind_data(dfs[2])


# POS: 61-63
# WIND-OBSERVATION direction angle
# The angle, measured in a clockwise direction, between true north and the direction from which the wind is blowing. MIN: 001 MAX: 360 UNITS: Angular Degrees
# SCALING FACTOR: 1
# DOM: A general domain comprised of the numeric characters (0-9).
# 999 = Missing. If type code (below) = V, then 999 indicates variable wind direction.

# POS: 64-64
# WIND-OBSERVATION direction quality code
# The code that denotes a quality status of a reported WIND-OBSERVATION direction angle. DOM: A specific domain comprised of the characters in the ASCII character set.
# 0 = Passed gross limits check
# 1 = Passed all quality control checks
# 2 = Suspect
# 3 = Erroneous
# 4 = Passed gross limits check, data originate from an NCEI data source
# 5 = Passed all quality control checks, data originate from an NCEI data source 6 = Suspect,
#     data originate from an NCEI data source
# 7 = Erroneous, data originate from an NCEI data source
# 9 = Passed gross limits check if element is present

# POS: 65-65
# WIND-OBSERVATION type code
# The code that denotes the character of the WIND-OBSERVATION.
# DOM: A specific domain comprised of the characters in the ASCII character set.
# A = Abridged Beaufort
# B = Beaufort
# C = Calm
# H = 5-Minute Average Speed
# N = Normal
# R = 60-Minute Average Speed
# Q = Squall
# T = 180 Minute Average Speed
# V = Variable
# 9 = Missing
# NOTE: If a value of 9 appears with a wind speed of 0000, this indicates calm winds.

# POS: 66-69
# WIND-OBSERVATION speed rate
# The rate of horizontal travel of air past a fixed point.
# MIN: 0000 MAX: 0900 UNITS: meters per second SCALING FACTOR: 10
# DOM: A general domain comprised of the numeric characters (0-9).
# 9999 = Missing.
# 8

# POS: 70-70
# WIND-OBSERVATION speed quality code
# The code that denotes a quality status of a reported WIND-OBSERVATION speed rate. DOM: A specific domain comprised of the characters in the ASCII character set.
# 0 = Passed gross limits check
# 1 = Passed all quality control checks
# 2 = Suspect
# 3 = Erroneous
# 4 = Passed gross limits check, data originate from an NCEI data source
# 5 = Passed all quality control checks, data originate from an NCEI data source 6 = Suspect, data originate from an NCEI data source
# 7 = Erroneous, data originate from an NCEI data source
# 9 = Passed gross limits check if element is present CRB = Climate Reference Book data from CDMP
