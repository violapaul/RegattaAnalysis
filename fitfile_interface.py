import itertools as it

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# These are libraries written for RaceAnalysis
import global_variables
G = global_variables.init_seattle()
import race_logs
import process as p
import analysis as a
import chart as c

from fitparse import FitFile

pd.set_option('display.max_rows', 100)
pd.reset_option('precision')
pd.set_option('display.float_format', lambda x: '%.3f' % x)

np.set_printoptions(precision=4, suppress=True, linewidth = 180)

SEMICIRCLES_TO_DEGREES = 180/2**31

def to_degrees(semicircles):
    return SEMICIRCLES_TO_DEGREES * semicircles

def fitfile_to_pandas(fitfile):
    rows = []
    # Get all data messages that are of type record
    for record in it.islice(fitfile.get_messages('record'), 0, None, None):
        # Go through all the data entries in this record
        row = {}
        for record_data in record:
            # Print the records name and value (and units if it has any)
            if record_data.units:
                name = record_data.name + "_" + record_data.units
            else:
                name = record_data.name
            row[name] = record_data.value
        rows.append(row)
    df = pd.DataFrame(rows)
    df = df[df.isna().sum(1) == 0]
    df['longitude'] = to_degrees(df.position_long_semicircles)
    df['latitude'] = to_degrees(df.position_lat_semicircles)
    df['row_times'] = df.timestamp.dt.tz_localize('UTC').dt.tz_convert('US/Pacific')
    return df

def test():
    fitfile = FitFile('/Users/viola/canlogs/GARMIN_WATCH/4359549845.fit')
    fitfile = FitFile('/Users/viola/canlogs/GARMIN_WATCH/4324683973.fit')
    fit_df = fitfile_to_pandas(fitfile)
    chart.plot_track(fit_df)

def test2(df):
    df = fit_df
    
    west, north = G.MAP(np.asarray(df.longitude), np.asarray(df.latitude))

    delta_t = df.timestamp.diff()/pd.Timedelta('1s')
    delta_w = np.diff(west, prepend=[delta_w[0]])
    delta_n = np.diff(north, prepend=[delta_n[0]])
    delta_p = np.vstack((delta_n, delta_w))
    distance = np.linalg.norm(delta_p, axis=0)

    c.quick_plot(df.index,
                 (delta_t, delta_w, delta_n, distance),
                 "(delta_t, delta_w, delta_n, distance)".split(),
                 fignum=1)

    c.quick_plot(df.index,
                 (delta_t, distance),
                 "(delta_t, distance)".split(),
                 fignum=1)

    
    
    plt.figure(2)
    plt.clf()
    plt.plot(delta_t, distance, linestyle = 'None', marker='.')
    
    for u in sorted(delta_t.unique()):
        cond_distance = distance[delta_t == u]
        print(f"{u:6.2f} {len(cond_distance):6.2f} {np.mean(cond_distance):6.2f} {np.std(cond_distance):6.2f}")
