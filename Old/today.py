import os

import copy
gimport math
import itertools as it
import importlib

import pandas as pd
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets

from numba import jit

from pyproj import Proj
import cv2

import boat_shape
from boat_shape import OUTLINE as BOAT_OUTLINE

import process as p
import analysis as a
import chart as c
import global_variables as G
from utils import DictClass

SAMPLES_PER_SECOND = 10

# Tacks take 20 seconds or more
# Compute the approximately constant TWD through the tack
# Compute the VMG through the tack (using TWD)
# Find the beginning of the tack (rudder??)
# Measure the lost boat lengths.
# Measure the time before returning to full speed
# Measure the initial "tack angle" and final "tack angle"
def find_tacks(df, buffer=20):
    if False:
        skip = 1
        dt = skip / SAMPLES_PER_SECOND
        sdf = rdf.copy()
    else:
        # a tack is a transition from close hauled port to stbd (and vice versa).
        skip = 5
        dt = skip / SAMPLES_PER_SECOND
        sdf = df[::skip].copy()

    # remove noise
    # sdf['hdg_f'] = scipy.ndimage.median_filter(uhdg, int(10 / dt), origin=0)
    # sdf['awa_f'] = scipy.ndimage.median_filter(np.array(sdf.awa), int(10 / dt), origin=0)

    sdf['hdg_f'], _ = p.exponential_filter_angle(np.array(sdf.hdg), 0.995, 4)
    sdf['awa_f'], _ = p.exponential_filter_angle(np.array(sdf.awa), 0.995, 4)

    # find regions where boat is always close hauled for a period of time
    period = int(5 / dt)
    sdf['is_stbd_hauled_raw'] = scipy.ndimage.minimum_filter1d((sdf.awa_f > 15) & (sdf.awa_f < 45), period)
    sdf['is_port_hauled_raw'] = scipy.ndimage.minimum_filter1d((sdf.awa_f < -15) & (sdf.awa_f > -45), period)

    # expand so that we can find overlaps, i.e. tacks
    sdf['is_stbd_hauled'] = scipy.ndimage.maximum_filter1d(sdf.is_stbd_hauled_raw, 4 * period)
    sdf['is_port_hauled'] = scipy.ndimage.maximum_filter1d(sdf.is_port_hauled_raw, 4 * period)

    # if you are in the overlap between expanded stbd and port hauled then you are
    # tacking!
    sdf['is_tack'] = (sdf.is_stbd_hauled & sdf.is_port_hauled)

    if False:
        c.quick_plot(sdf.index, (sdf.awa, sdf.awa_f, 100*sdf.is_tack, 100*sdf.is_stbd_hauled, 100*sdf.is_port_hauled),
                     "(sdf.hdg, sdf.hdg_f, sdf.awa, sdf.awa_f)".split(),
                     fignum=1)

        c.quick_plot(sdf.index, (sdf.hdg, sdf.hdg_f, sdf.awa, sdf.awa_f, 100*sdf.is_tack),
                     "(sdf.hdg, sdf.hdg_f, sdf.awa, sdf.awa_f)".split(),
                     fignum=1)

    res = []
    for s, e in p.find_runs(np.array(sdf.is_tack)):
        s_extend = int(s-(buffer/dt))
        e_extend = int(e+(buffer/dt))
        hdg_before = np.median(sdf.hdg_f.iloc[s_extend:s])
        hdg_after = np.median(sdf.hdg_f.iloc[e:e_extend])
        tack_angle = np.abs(p.angle_diff(hdg_before, hdg_after))
        if True or tack_angle < 115 and tack_angle > 65:
            res.append(slice(sdf.index[s_extend], sdf.index[e_extend]))

    return res, sdf


# Tacks take 20 seconds or more
# Compute the approximately constant TWD through the tack
# Compute the VMG through the tack (using TWD)
# Find the beginning of the tack (rudder??)
# Measure the lost boat lengths.
# Measure the time before returning to full speed
# Measure the initial "tack angle" and final "tack angle"
def find_tacks_2(df):
    # a tack is a transition from close hauled port to stbd (and vice versa).
    skip = 5
    dt = skip / SAMPLES_PER_SECOND
    sdf = df[::skip]

    # remove noise
    awa_f = scipy.ndimage.median_filter(sdf.awa, int(100 / dt), origin=0)

    # find regions where boat is always close hauled for a period of time
    period = int(20 / dt)
    is_stbd_hauled = scipy.ndimage.minimum_filter1d((awa_f > 10) & (awa_f < 50), period)
    is_port_hauled = scipy.ndimage.minimum_filter1d((awa_f < -10) & (awa_f > -50), period)

    # expand so that we can find overlaps, i.e. tacks
    is_stbd_hauled = scipy.ndimage.maximum_filter1d(is_stbd_hauled, 2 * period)
    is_port_hauled = scipy.ndimage.maximum_filter1d(is_port_hauled, 2 * period)

    # if you are in the overlap between expanded stbd and port hauled then you are
    # tacking!
    is_tack = (is_stbd_hauled & is_port_hauled)

    # remove noise
    twd_f = scipy.ndimage.median_filter(sdf.twd, int(100 / dt), origin=0)

    res = []
    for s, e in a.find_runs(is_tack):
        twd = twd_f[(s + e) // 2]
        ss = slice(sdf.index[s], sdf.index[e])
        tdf = df[ss]
        vmg = tdf.spd * a.cos_d(twd - tdf.hdg)
        # start is when the rudder is turned by 20 degrees? in less than a second
        # when does spd return to 95% (what happens if it does not?)
        # compute duration
        # vmg before times duration VS real distance toward TWD
        res.append((sdf.index[s], sdf.index[e]))

    return res


def find_best_hdg_offset(df):
    delay = 17
    dt = 1
    ss = slice(41844, 111988, dt)
    dss = slice(ss.start+delay, ss.stop+delay, dt)

    mdf = df[ss]
    ddf = df[dss]
    vog_n = dt * ddf.sog * a.north_d(ddf.cog)
    vog_e = dt * ddf.sog * a.east_d(ddf.cog)

    for off in np.linspace(12, 18, num=20):
        hdg = mdf.hdg + off
        btv_n = dt * mdf.spd * a.north_d(hdg)
        btv_e = dt * mdf.spd * a.east_d(hdg)

        cur_n = 10.0 * (np.asarray(vog_n) - np.asarray(btv_n))
        cur_e = 10.0 * (np.asarray(vog_e) - np.asarray(btv_e))

        diff_n = np.diff(cur_n).std()
        diff_e = np.diff(cur_e).std()
        std_n = cur_n.std()
        std_e = cur_e.std()
        all_sum = diff_n + diff_e + std_n + std_e
        print(f"off {off:7.3f} {diff_n:7.3f} {diff_e:7.3f} {std_n:7.3f} {std_e:7.3f} {all_sum:7.3f}")


def show_boat_arrows(df, df_slice, dt_seconds=5, skip=2, current_scale=1):
    delay = 16
    dt = dt_seconds * SAMPLES_PER_SECOND
    scale = dt_seconds
    ss = slice(df_slice.start, df_slice.stop, dt)
    dss = slice(ss.start+delay, ss.stop+delay, dt)
    
    mdf = df[ss]
    ddf = df[dss]
    vog_n = scale * ddf.sog * a.north_d(ddf.cog)
    vog_e = scale * ddf.sog * a.east_d(ddf.cog)

    tw_n = scale * ddf.tws * a.north_d(ddf.twd)
    tw_e = scale * ddf.tws * a.east_d(ddf.twd)

    hdg = mdf.hdg + 13.2
    btv_n = scale * mdf.spd * a.north_d(hdg)
    btv_e = scale * mdf.spd * a.east_d(hdg)

    cur_n = current_scale * (np.asarray(vog_n) - np.asarray(btv_n))
    cur_e = current_scale * (np.asarray(vog_e) - np.asarray(btv_e))

    chart = a.plot_chart(mdf, 3, border=0.0)
    longitudes = np.asarray(mdf.longitude)
    latitudes = np.asarray(mdf.latitude)
    pos = np.vstack(G.MAP(longitudes, latitudes)).T - (chart.west, chart.south)

    color = 'blue'
    hwidth = scale/5
    for (east, north), ve, vn in it.islice(zip(pos, vog_e, vog_n), 0, None, skip):
        chart.ax.arrow(east, north, ve, vn, head_width=hwidth, length_includes_head=True, color=color)

    color = 'green'
    for (east, north), ve, vn in it.islice(zip(pos, tw_e, tw_n), 0, None, skip):
        ee = east + ve
        nn = north + vn
        chart.ax.arrow(ee, nn, -ve, -vn, head_width=hwidth, length_includes_head=True, color=color)

    color = 'red'
    for (east, north), ve, vn in it.islice(zip(pos, btv_e, btv_n), 0, None, skip):
        chart.ax.arrow(east, north, ve, vn, head_width=hwidth, length_includes_head=True, color=color)

    color = 'orange'
    for (east, north), ve, vn in it.islice(zip(pos, cur_e, cur_n), 0, None, skip):
        chart.ax.arrow(east, north, ve, vn, head_width=hwidth, length_includes_head=True, color=color)


def show_vmg(df, df_slice):

    df_slice = slice(79140, 105810)

    delay = 16
    dt = 1
    ss = slice(df_slice.start, df_slice.stop, dt)
    dss = slice(ss.start+delay, ss.stop+delay, dt)
    
    mdf = df[ss]
    ddf = df[dss]

    slow_coeff = p.butterworth_filter(0.03, 5)

    vmg = mdf.spd * a.cos_d(np.asarray(ddf.twa))
    ptwa = np.asarray(ddf.twa)
    ptwa[ptwa < 0] = - ptwa[ptwa < 0]
    ptwa = p.smooth(slow_coeff, ptwa, causal=False)

    pawa = np.asarray(ddf.awa)
    pawa[pawa < 0] = - pawa[pawa < 0]

    pvmg = polars_spline(ddf.tws, ptwa-10)

    a.quick_plot(mdf.index,
                 (ddf.tws, ptwa/10, pawa/10, mdf.spd, vmg, pvmg),
                 "tws ptwa, pawa, spd, vmg, polar".split(),
                 fignum=6)

    
    vog_n = dt * ddf.sog * a.north_d(ddf.cog)
    vog_e = dt * ddf.sog * a.east_d(ddf.cog)

    tw_n = dt * ddf.tws * a.north_d(ddf.twd)
    tw_e = dt * ddf.tws * a.east_d(ddf.twd)

    hdg = mdf.hdg + 13.2
    btv_n = dt * mdf.spd * a.north_d(hdg)
    btv_e = dt * mdf.spd * a.east_d(hdg)

    cur_n = 10.0 * (np.asarray(vog_n) - np.asarray(btv_n))
    cur_e = 10.0 * (np.asarray(vog_e) - np.asarray(btv_e))

    chart = a.plot_chart(mdf, 3)
    longitudes = np.asarray(mdf.longitude)
    latitudes = np.asarray(mdf.latitude)
    pos = np.vstack(a.MAP(longitudes, latitudes)).T - (chart.west, chart.south)

    color = 'blue'
    hwidth = dt/5
    for (east, north), ve, vn in it.islice(zip(pos, vog_e, vog_n), 0, None, 10):
        chart.ax.arrow(east, north, ve, vn, head_width=hwidth, length_includes_head=True, color=color)

    color = 'green'
    for (east, north), ve, vn in it.islice(zip(pos, tw_e, tw_n), 0, None, 10):
        chart.ax.arrow(east, north, ve, vn, head_width=hwidth, length_includes_head=True, color=color)

    color = 'red'
    for (east, north), ve, vn in it.islice(zip(pos, btv_e, btv_n), 0, None, 10):
        chart.ax.arrow(east, north, ve, vn, head_width=hwidth, length_includes_head=True, color=color)

    color = 'orange'
    for (east, north), ve, vn in it.islice(zip(pos, cur_e, cur_n), 0, None, 10):
        chart.ax.arrow(east, north, ve, vn, head_width=hwidth, length_includes_head=True, color=color)
        

def test():
    importlib.reload(p)
    importlib.reload(a)

    example = DictClass(log='2019-12-07_09:47.pd', begin=101881, end=109567)
    example = DictClass(log='2019-12-07_09:47.pd', begin=54316, end=109378)

    sail_logs = [example.log]

    dfs, bdf = p.read_sail_logs(sail_logs, skip_dock_only=False, trim=True, path=a.DATA_DIRECTORY, cutoff=0.3)
    df = dfs[0]
    sdf = df[example.begin : example.end]
    chart = a.plot_track(sdf, 1)

    race_slice = slice(102262, example.end)
    show_boat_arrows(df, race_slice, dt_seconds=10, skip=2)
    rdf = df.loc[race_slice]

    chart = a.plot_track(rdf, 3)
    
    rdf = sdf
    # for polars, all in knots and not m/2 (factor of 2!!!)
    ptwa = rdf.twa.copy()
    ptwa[ptwa < 0] = - ptwa[ptwa < 0]
    # all in meters/se
    ptws = rdf.tws
    knots_to_ms = 0.514
    pspd = polars_spline(ptws/knots_to_ms, ptwa) * knots_to_ms

    
    target = np.minimum(rdf.spd/(np.maximum(pspd, 0.3)), 2.0)
    scale = 10/knots_to_ms
    a.quick_plot(rdf.index,
               (scale*rdf.spd, scale*pspd, scale*rdf.tws, rdf.awa, rdf.twa, 40*target),
               "10*spd, 10*polar, 10*tws, awa, twa target".split(),
               fignum=4)

    scale = 1/knots_to_ms
    quick_plot(rdf.index,
               (scale*rdf.spd, scale*pspd, scale*rdf.tws, target, ptwa/10),
               "spd, polar, tws target, twa/10".split(),
               fignum=4)

    quick_plot(rdf.index,
               (target, rdf.spd, pspd),
               "target spd, polar".split(),
               fignum=4)

    

    show_boat_arrows(df, race_slice, dt_seconds=10, skip=2)

    mdf = rdf[::50]
    vog_n = mdf.sog * a.north_d(mdf.cog)
    vog_e = mdf.sog * a.east_d(mdf.cog)
    
    add_arrows(chart, mdf, vog_e, vog_n, color='blue')

    sdf = df.loc[79140:105810]
    chart = a.plot_track(sdf, 3)
    tacks = find_tacks(sdf)

    num = 5
    ss = slice(tacks[num][0]-2000, tacks[num+1][1]+2000)
    tdf = df.loc[ss]
    chart = a.plot_track(tdf, 4)

    tdf = rdf
    vmg = tdf.spd * a.cos_d(tdf.twa)

    
    quick_plot(tdf.index, (tdf.spd, vmg, (tdf.hdg/30), tdf.rudder/4, tdf.tws), "spd vmg hdg rudder tws".split(), fignum=5)

    
    quick_plot(tdf.index, (30*tdf.spd, 30*vmg, tdf.hdg, 30*tdf.rudder/4, 30*tdf.tws, 30*tdf.sog),
               "spd vmg hdg rudder tws sog".split(),
               fignum=5)    

    quick_plot(df.index, (df.hdg+15.4, df.cog), "hdg cog".split(), fignum=2)
    quick_plot(tdf.index, (tdf.hdg+15.4, tdf.cog), "hdg cog".split(), fignum=2)

    ss = slice(tacks[num][0]-200, tacks[num+1][1]+200, 50)
    a.plot_boat(tdf, ss, 5)


    row = tdf.loc[tacks[num][0]-200]
    a.boat_forces(row, 10)

def for_sara():

    DATA_DIRECTORY = '/Users/viola/canlogs'
    example = DictClass(log='2019-11-16_10:09.pd', begin=41076, end=111668)
    example = DictClass(log='2019-10-26_09:40.pd', doc='Grand Prix Saturday.', begin=40503, end=87408)
    example = DictClass(log='2019-10-12_09:45.pd', doc='CYC PSSC Day 1', begin=19081, end=233893)
    example = DictClass(log='2019-11-16_10:09.pd', doc='Snowbird #1.', begin=42548, end=111668)
    sail_logs = [example.log]

    dfs, bdf = p.read_sail_logs(sail_logs, discard_columns=False, skip_dock_only=False, trim=True, path=DATA_DIRECTORY, cutoff=0.3)
    df = dfs[0]
    cdf = df.copy()

    chart = c.plot_chart(df, 3)
    c.draw_track(cdf, chart, color='green')
    chart = a.plot_track(cdf, 1, skip=1)
    
    rdf = df.loc[example.begin : example.end]

    chart = c.plot_track(rdf, 1)

    tacks, tdf = find_tacks(rdf)
    chart = c.plot_chart(rdf, 3)
    c.draw_track(rdf, chart, color='red')
    for tack_slice in tacks:
        tdf = rdf.loc[tack_slice]
        lon, lat = np.asarray(tdf.longitude), np.asarray(tdf.latitude)
        track = np.vstack(G.MAP(lon, lat)).T - (chart.west, chart.south)
        c.draw_track(tdf, chart, color='green')


        if np.all(np.any(np.abs(track - (1323, 1703)) < 10, 0)):
            print(True)
            break
        

    chart = plot_chart(rdf, 3)
    draw_track(rdf, chart, color='red')
    draw_track(tdf, chart, color='green')

    tdf = rdf.loc[tacks[10]]
    draw_track(rdf, chart, color='red')    
    draw_track(tdf, chart, color='green')

    uhdg = np.degrees(np.unwrap(np.radians(tdf.hdg)))


    scale = a.MS_2_KNOTS
    a.quick_plot(tdf.index, (scale*tdf.spd, scale*tdf.aws, tdf.rudder),
                 legend = "(scale*tdf.spd, scale*tdf.aws, tdf.rudder)".split(),
                 fignum=2, clf=True)


    scale = 10 * a.MS_2_KNOTS
    a.quick_plot(tdf.index, (scale*tdf.spd, uhdg, tdf.awa, scale*tdf.aws, tdf.rudder),
                 legend = "(scale*tdf.spd, uhdg, tdf.awa, scale*tdf.aws, tdf.rudder, awa_f)".split(),
                 fignum=2, clf=True)

    
    vmg = tdf.spd * a.MS_2_KNOTS * a.cos_d(tdf.twa)

    quick_plot(tdf.index, (tdf.tws*MS_2_KNOTS, tdf.twa/27, tdf.spd * MS_2_KNOTS, tdf.awa/27, vmg),
               legend = "tws, twa/27, spd, awa/27, vmg".split(),
               fignum=2, clf=True, title='raw data')
        
