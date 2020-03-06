import math
import os

import itertools as it

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline

import scipy
import scipy.interpolate
import scipy.optimize

import pandas as pd

from utils import DictClass

pd.set_option('display.max_rows', 50)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
np.set_printoptions(precision=4, suppress=True, linewidth = 180)

# Code to read legacy ORC files. #############################################
def read_tws_entry(stream):
    """
    Read an ORC entry, cut and paste from PDF file.  Returns a map from col name to row
    values.
    """
    line_number = 0
    column_count = 8
    row_count = 14
    lines = [stream.readline().strip() for i in range(column_count * (1 + row_count))]
    # Last line is the TWS
    _, _, tws, _  = stream.readline().strip().split()
    rows = [dict(tws = float(tws)) for i in range(row_count)]
    for i in range(column_count):
        col_key = lines[line_number]
        line_number += 1
        for j in range(row_count):
            rows[j][col_key] = float(lines[line_number])
            line_number += 1
    return rows


def read_orc_file(filename):
    "Read each TWS entry in turn from the ORC file."
    with open(filename) as f:
        polars = []
        for i in range(7):
            polars.extend(read_tws_entry(f))
    return polars


# Helpers #######################################################################

def entries_from_table(polars_table):
    "Convert to a set of entries, including TWA, TWS, SPD, AWS, AWA."

    tmp = polars_table.copy()
    tmp['twa'] = tmp.index
    entries = tmp.melt(id_vars=['twa'])
    entries = entries.rename(columns=dict(value='spd'))

    twa = np.array(entries.twa)
    tws = np.array(entries.tws, dtype=np.float64)
    spd = np.array(entries.spd)

    tw_n = - (np.cos(np.radians(twa)) * tws)
    tw_e = - (np.sin(np.radians(twa)) * tws)

    boat_n = spd
    boat_e = 0

    aw_n = tw_n - boat_n
    aw_e = tw_e - boat_e
    entries['aws'] = np.sqrt(np.square(aw_n) + np.square(aw_e))
    entries['awa'] = np.degrees(np.arctan2(-aw_e, -aw_n))
    entries['vmg'] = np.cos(np.radians(twa)) * spd

    return entries

def plot_entries(entries, fignum=None, clf=True):

    fig = plt.figure(fignum)
    if clf:
        fig.clf()
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, num=None)

    ax1.grid(True)
    ax2.grid(True)

    df = entries
    ax1.plot(df.twa, df.spd, marker='.')
    ax1.plot(df.twa, df.vmg, marker='.')
    ax1.legend(['spd', 'vmg'], loc='best')

    ax2.plot(df.awa, df.spd, marker='.')
    ax2.plot(df.awa, df.vmg, marker='.')
    ax2.legend(['spd', 'vmg'], loc='best')


def plot_polars_entries(entries, colname, fignum=None, clf=True, clockwise=True):

    fig = plt.figure(fignum)
    if clf:
        fig.clf()
    ax = fig.add_subplot(111, projection='polar')

    ax.set_rmax(10)
    ax.set_rlabel_position(-22.5)
    ax.grid(True)
    ax.set_theta_offset(math.pi/2.0)
    ax.set_theta_direction(-1)
    ax.set_xticks(np.pi/180 * np.arange(0, 360, 15))

    legend = []
    for tws in entries.tws.unique():
        df = entries[entries.tws == tws]
        angles = np.radians(df[colname])
        if not clockwise:
            angles = - angles
        speeds = np.array(df.spd)
        ax.plot(angles, speeds, marker='.')
        legend.append(f"TWS = {tws}")

    plt.legend(legend, loc='lower left')
    plt.show()


def plot_polars_table(polars_table, clear=False):

    fig = plt.figure(1)
    fig.clf()
    ax = fig.add_subplot(111, projection='polar')

    ax.set_rmax(10)
    ax.set_rlabel_position(-22.5)
    ax.grid(True)
    ax.set_theta_offset(math.pi/2.0)
    ax.set_theta_direction(-1)

    angles = np.deg2rad(np.array(polars_table.index))
    for col in polars_table.columns:
        speeds = np.array(polars_table[col])
        ax.plot(angles, speeds)

    plt.show()


# Process and convert sparse polars data and compute dense polars tables. #######
def augment_orc(orc):
    """
    Add additional synthetic entries based on some simple physics assumptions.  This
    ensures that various extrapolations are meaningful.
    """
    new_rows = []

    # Add some imaginary speeds
    true_wind_speeds = [2, 4] + list(orc.tws.unique())
    # Make sure you cannot sail upwind
    for tws in true_wind_speeds:
        new_rows.extend([dict(tws=tws, twa=a, spd=0) for a in np.arange(0, 15, 5)])

    # Add additional, undefined, entries for low wind speeds.  These will be filled in
    # using interpolation (see above).
    for tws in true_wind_speeds:
        new_rows.extend([dict(tws=tws, twa=a) for a in np.arange(15, orc.twa.min(), 5)])

    # You can't move if there is no wind
    for twa in orc.twa.unique():
        new_rows.append(dict(tws=0, twa=twa, spd=0))

    orc = orc.append(new_rows, ignore_index=True)

    orc = orc.sort_values(by=['tws', 'twa'])
    return orc


def full_twa_polars(orc, smooth=100000, function='quintic', epsilon=None, graph=False):
    """
    Compute the full polar table from the subset of data that is available in the VPP/ORC
    file.  The key is that the output is a *square* table with columns of TWS and rows for
    TWA.  The input is sparse and does not include all the values needed.  Will use
    splines to interpolate.
    """

    # Add in additional entries for low speeds and low wind angles (not in the original
    # polars).  Some have unknown spd values.
    augmented = augment_orc(orc)

    defined = augmented[augmented.spd.notna()]
    # After a bit of struggle settled on thin_plate.  Both smooth and interpolating.
    # Note, we could have done this is in radians or in angles (or anything else).
    polars_spline = scipy.interpolate.Rbf(defined.tws, defined.twa, defined.spd,
                                         smooth=smooth, function=function, epsilon=epsilon)

    prediction_error = polars_spline(orc.tws, orc.twa) - orc.spd
    print("Max abs prediction errors is {0}.".format(prediction_error.abs().max()))

    # Sort of subtle.  First form the pivot, and then determine which values are missing
    # (pivot essentially squares a sparse matrix leaving all undefined entries as NaN).
    pp = augmented.pivot(index='twa', columns='tws', values='spd')
    pp['twa'] = pp.index
    # Then convert back to the sparse list (which is not really sparse).
    full = pp.melt(id_vars=['twa'])  # melt is the opposite of pivot

    full['predicted_spd'] = polars_spline(full.tws, full.twa)
    polars_table = full.pivot(index='twa', columns='tws', values='predicted_spd')
    # Values are ever so slightly less than 0.0
    polars_table = polars_table.clip(lower=0)
    
    prediction_error = (polars_table - pp).abs().sum().sum()
    print("Sum of abs prediction errors is {0}.".format(prediction_error))

    if graph:
        plt.figure(2)
        plt.clf()
        for i in polars_table.columns:
            polars_table[i].plot()
            pp[i].plot(marker='.')

        plot_polars_table(polars_table)

    return polars_table, polars_spline


# Explore the polars and how to sail #########################################
def polar_angle(angle):
    angle = angle if angle > 0 else -angle
    angle = angle if angle < 180 else (360 - angle)
    return angle


def best_vmg(tws, twd, tmd, polars):
    """
    Find the best heading to sail, that maximizes VMG.  This is all done in TRUE.
    """

    def vmg(tbd):
        "return the vmg to the mark given, TWS, TWD, TMD, and polars."
        # true_wind_angle = true_wind_direction - true_boat_direction
        twa = polar_angle(twd - tbd)
        # apparent_mark_direction = true_mark_direction - true_boat_direction
        amd = tmd - tbd
        spd = polars(tws, twa)
        vmg = spd * math.cos(math.radians(amd))
        return -vmg

    res = scipy.optimize.minimize_scalar(vmg, bounds=[-180, 180], method='bounded')
    return res


def best_vmg_apparent(tws, twd, tmd, polars):
    """
    Find the best heading to sail, that maximizes VMG.  This is all done in TRUE.
    """

    def vmg(tbd):
        "return the vmg to the mark given, TWS, TWD, TMD, and polars."
        # true_wind_angle = true_wind_direction - true_boat_direction
        twa = twd - tbd
        twa = twa if twa > 0 else -twa
        twa = twa if twa < 180 else (360 - twa)
        # apparent_mark_direction = true_mark_direction - true_boat_direction
        amd = tmd - tbd
        spd = polars(tws, twa)
        vmg = spd * math.cos(math.radians(amd))
        return -vmg

    res = scipy.optimize.minimize_scalar(vmg, bounds=[-180, 180], method='bounded')
    return res.x


def test_best_to_mark(polars_spline):
    print(best_vmg(10, 180,  -30, polars_spline))
    print(best_vmg(10, 170,  -30, polars_spline))
    print(best_vmg(10, -170, -30, polars_spline))
    print(best_vmg(10, 190,  -30, polars_spline))    
    print(best_vmg(10, 180,  -20, polars_spline))    
    print(best_vmg(10, -170, -30, polars_spline))
    print(best_vmg(10, 180,  0,   polars_spline))
    print(best_vmg(10, 170,  0,   polars_spline))
    print(best_vmg(10, 190,  0,   polars_spline))    


def north(pos):
    return pos[0]

def east(pos):
    return pos[1]
    

def sail_to_mark(mark, boat, tws, twd, polars):
    twd = 180
    tws = 10
    polars = polars_spline

    optimal = False
    mark = np.array([1000, 0])
    boat = np.array([0, 0])
    positions = [boat]
    angles = it.cycle([31, -31])
    for i in range(20):
        delta = mark - boat
        if north(delta) < 10:
            break
        tmd = np.degrees(math.atan2(delta[1], delta[0]))
        if optimal:
            res = best_vmg(tws, twd, tmd, polars)
            tbd = res.x
        else:
            tbd = next(angles)
        speed = polars(tws, polar_angle(twd - tbd))
        print(np.linalg.norm(delta), tbd, speed)
        r = np.radians(tbd)
        boat = boat + 20 * speed * np.array([np.cos(r), np.sin(r)])
        positions.append(boat)

    fig = plt.figure(4)
    if optimal:
        plt.clf()
    ss = np.vstack(positions)
    ax = fig.add_subplot(111)
    ax.axis('equal')
    plt.plot(ss[:,1], ss[:,0], marker='.')


def fit_spline(angles, speeds, min_angle=0):
    ius = InterpolatedUnivariateSpline(angles, speeds)

    all_angles = np.linspace(np.radians(min_angle), np.pi, 50)
    spline_speeds = ius(all_angles)

    return all_angles, spline_speeds


def polars_spline_fit(tws_orc):
    "Fit a spline to the twa and spd properties of the orc subtable (for a given TWS)."
    angles = np.radians(np.array(tws_orc.twa))
    speeds = np.array(tws_orc.spd)

    return InterpolatedUnivariateSpline(angles, speeds)


# Code that exercises the above routines.
def run_code():
    """
    Code to exercise most of the functionality in this file.  Note, much of it should only
    be done once.
    """
    ONE_TIME = False

    BASE_DIR = "/Users/viola/canlogs/Polars"
    orc_pandas_file = os.path.join(BASE_DIR, "j105_orc_tables.pd")

    if ONE_TIME:
        # Read the strange VPP/ORC text info (converted from the PDF).
        orc_text_file  = os.path.join(BASE_DIR, "j105_orc_tables.txt")

        orc = pd.DataFrame(read_orc_file(orc_text_file))
        # hate mixed case column names!
        orc.columns = orc.columns.str.lower()
        orc = orc.rename(columns=dict(btv='spd'))
        orc = orc.sort_values(by=['tws', 'twa'])

        orc.to_pickle(orc_pandas_file)
    else:
        orc = pd.read_pickle(orc_pandas_file)

    polars_table, polars_spline = full_twa_polars(orc, graph=True)
    entries = entries_from_table(polars_table)

    plot_polars_entries(entries, 'twa', clockwise=True, fignum=1)
    plot_polars_entries(entries, 'awa', clockwise=False, clf=False, fignum=1)

    polars_file = os.path.join(BASE_DIR, "j105_polars.csv")
    if ONE_TIME:
        # get rid of the zero TWS, just in case its a problem.
        polars_table = polars_table.drop(columns=[0])
        polars_table.to_csv(polars_file, float_format='%.3f')


def check_orc_polars():
    file = '/Users/viola/Python/sailing/Data/Polars/j105_orc_tables.pd'
    orc = pd.read_pickle(file)

    # apparent wind and the boat velocity point in opposite directions, so true wind is
    # apparent PLUS boat.  

    # Compute TW from AW (like on the boat).
    for i, row in orc.iterrows():
        slip_angle = 0
        d = DictClass(orc_twa=row.twa, orc_tws=row.tws, spd=row.spd, orc_awa=row.awa, orc_aws=row.aws)

        # Wind at 45 degrees blows from the North East (negative north, and negative east).
        d.aw_n = - (np.cos(np.radians(d.orc_awa)) * d.orc_aws)
        d.aw_e = - (np.sin(np.radians(d.orc_awa)) * d.orc_aws)
        
        # Boat is moving north,  with some slip to the east.
        d.spd_n = d.spd
        d.spd_e = - np.tan(np.radians(slip_angle)) * d.spd

        d.tw_n = d.aw_n + d.spd_n
        d.tw_e = d.aw_e + d.spd_e
        d.p_tws = np.sqrt(np.square(d.tw_n) + np.square(d.tw_e))
        d.p_twa = np.degrees(np.arctan2(-d.tw_e, -d.tw_n))

        # Are they close?
        if not np.allclose((d.orc_twa, d.orc_tws), (d.p_twa, d.p_tws), rtol=0.02):
            print(i, row)
            print(d)
            print(f"ORC: twa {d.orc_twa:.2f} tws {d.orc_tws:.2f}")
            print(f"   : twa {d.p_twa:.2f} tws {d.p_tws:.2f}")

    # Compute AW from the TW (like on the boat).
    for i, row in orc.iterrows():
        slip_angle = 0
        d = DictClass(orc_twa=row.twa, orc_tws=row.tws, spd=row.spd, orc_awa=row.awa, orc_aws=row.aws)

        # Wind at 45 degrees blows from the North East (negative north, and negative east).
        d.tw_n = - (np.cos(np.radians(d.orc_twa)) * d.orc_tws)
        d.tw_e = - (np.sin(np.radians(d.orc_twa)) * d.orc_tws)
        
        # Boat is moving north,  with some slip to the east.
        d.spd_n = d.spd
        d.spd_e = - np.tan(np.radians(slip_angle)) * d.spd

        d.aw_n = d.tw_n - d.spd_n
        d.aw_e = d.tw_e - d.spd_e
        d.p_aws = np.sqrt(np.square(d.aw_n) + np.square(d.aw_e))
        d.p_awa = np.degrees(np.arctan2(-d.aw_e, -d.aw_n))

        # Are they close?
        if not np.allclose((d.orc_awa, d.orc_aws), (d.p_awa, d.p_aws), rtol=0.002):
            print(i, row)
            print(d)
            print(f"ORC: awa {d.orc_awa:.2f} aws {d.orc_aws:.2f}")
            print(f"   : awa {d.p_awa:.2f} aws {d.p_aws:.2f}")

