import itertools as it
import importlib

import pandas as pd
import numpy as np
import scipy.ndimage

from numba import jit

import matplotlib.pyplot as plt

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

def show_tacks():
    DATA_DIRECTORY = '/Users/viola/canlogs'

    DATA_DIRECTORY = '/Users/billey/scode/slogs'

    example = DictClass(log='2019-11-16_10:09.pd', begin=41076, end=111668)
    example = DictClass(log='2019-10-12_09:45.pd', doc='CYC PSSC Day 1', begin=19081, end=233893)
    example = DictClass(log='2019-11-16_10:09.pd', doc='Snowbird #1.', begin=42548, end=111668)
    example = DictClass(log='2019-10-26_09:40.pd', doc='Grand Prix Saturday.', begin=40503, end=87408)
    sail_logs = [example.log]

    dfs, bdf = p.read_sail_logs(sail_logs, discard_columns=False, skip_dock_only=False, trim=True, path=DATA_DIRECTORY, cutoff=0.3)
    df = dfs[0]

    chart = c.plot_track(df, 20)

    decay = 0.99
    df['sawa'], _ = p.exponential_filter_angle(np.array(df.rawa), decay, 4)
    df['shdg'], _ = p.exponential_filter_angle(np.array(df.rhdg), decay, 4)

    # decay = 0.9
    df['saws'], _ = p.exponential_filter(np.array(df.raws), decay, 0.8)
    # decay = 0.8
    df['sspd'], _ = p.exponential_filter_angle(np.array(df.rspd), decay, 0.5)

    # Note, TWA is computed from AWA, AWS, and boat speed (SPD)
    tw_north = p.cos_d(df.sawa) * df.saws - df.sspd
    tw_east = p.sin_d(df.sawa) * df.saws
    df['stwa'] = np.degrees(np.arctan2(tw_east, tw_north))
    df['stwd'] = df.stwa + df.shdg + df.variation.mean()

    c.quick_plot(df.index, (df.shdg, df.rhdg, df.hdg), "shdg rhdg hdg".split())
    
    fig = plt.figure(30)
    fig.clf()
    (ax1, ax2, ax3) = fig.subplots(3, 1, sharex=True)
    c.quick_plot_ax(ax1, df.index, (df.sawa, df.rawa, df.twa), "sawa rawa twa".split())
    c.quick_plot_ax(ax2, df.index, (df.saws, df.raws, df.twa), "saws raws twa".split())    
    c.quick_plot_ax(ax3, df.index, (df.twa, df.twd, df.shdg, df.stwa, df.stwd),
                    legend = "twa, twd, shdg, stwa, stwd".split())

    rdf = df.loc[example.begin : example.end]
    

    chart = c.plot_track(rdf, 1)

    tacks, tdf = find_tacks(rdf)
    chart = c.plot_chart(rdf)
    c.draw_track(rdf, chart, color='red')
    tack_list = [7, 8, 9]  # list(range(7, 15))
    for i in tack_list:
        tdf = rdf.loc[tacks[i]]
        c.draw_track(tdf, chart, color='green')

    tdf = rdf.loc[tacks[5]]

    for i in tack_list: # , 6, 7, 8, 9]:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, num=i+1)

        delta_t = 150
        tdf = rdf.loc[tacks[i]]

        ss = slice(int(len(tdf) * 0.7), int(len(tdf) * 1.0))
        twd_average = tdf.iloc[ss].twd.mean()
        twd_average = 185
        tws_average = tdf.tws.mean()

        v_twd = np.array([[p.sin_d(twd_average), p.cos_d(twd_average)]])
        track = np.vstack(G.MAP(tdf.longitude.to_numpy(), tdf.latitude.to_numpy())).T

        dmg = track.dot(v_twd.T)
        dmg = dmg - dmg.min()
        index = tdf.index - tdf.index.min()

        smg = (dmg[delta_t] - dmg[0])/delta_t

        pmg = np.cumsum(smg * np.ones(dmg.shape), axis=0)
        ax2.plot(index, dmg)
        ax2.plot(index, pmg)
        ax2.plot(index, 10 * tdf.spd)
        ax2.plot(index, tdf.twd)

        uhdg = np.degrees(np.unwrap(np.radians(tdf.hdg)))
        scale = 10 * G.MS_2_KNOTS

        c.quick_plot_ax(ax3, tdf.index, (scale*tdf.spd, uhdg, tdf.awa, scale*tdf.aws, tdf.rudder),
                        legend = "(scale*tdf.spd, uhdg, tdf.awa, scale*tdf.aws, tdf.rudder, awa_f)".split())

        ctrack = track - track[0]
        delta = ctrack[delta_t] / delta_t
        dtrack = np.cumsum(delta * np.ones(ctrack.shape), axis=0)
        
        ax1.axis('equal')
        ax1.plot(ctrack[:, 0], ctrack[:, 1])
        ax1.plot(dtrack[:,0], dtrack[:,1])
        middle = track.min(axis=0)
        base = ctrack[0]
        to_arrow = (20 * tws_average * v_twd).reshape(-1)
        c.draw_arrow(ax1, base, base + to_arrow, 'green')
        base = ctrack[-1]
        to_arrow = (-20 * tws_average * v_twd).reshape(-1)
        c.draw_arrow(ax1, base, base + to_arrow, 'green')


@jit(nopython=True)
def estimate_true_wind_helper(epsilon, aws, awa, hdg, spd, tws, twd, variation):
    twd = np.radians(twd) + np.zeros(awa.shape)
    tws = tws + np.zeros(awa.shape)
    res_n = np.zeros(awa.shape)
    res_e = np.zeros(awa.shape)
    aw_n = aws * np.cos(np.radians(awa))
    aw_e = aws * np.sin(np.radians(awa))

    rhdg = np.radians(hdg)
    variation = np.radians(variation)

    for i in range(1, len(aws)):
        twa = twd[i-1] - (rhdg[i] + variation)
        c = np.cos(twa)
        s = np.sin(twa)
        f_aw_n = spd[i] + c * tws[i-1]
        f_aw_e =          s * tws[i-1]

        res_n[i] = (aw_n[i] - f_aw_n)
        res_e[i] = (aw_e[i] - f_aw_e)

        delta_tws = 30 * res_n[i] * c + res_e[i] * s
        delta_twd = res_n[i] * tws[i-1] * -s + res_e[i] * tws[i-1] * c

        tws[i] = epsilon * delta_tws + tws[i-1]
        twd[i] = epsilon * delta_twd + twd[i-1]

    return np.degrees(twd), tws, res_n, res_e


# Now the last parameter can be defined, it is the max error.  This is max_tracking_error threshold.
theta = 6  # the threshold, which should be approximate the noise 
alpha = 0.97

# Note the different signals require different settings of alpha, because they have different 
# dynamics and noise.
df['sawa'], _ = p.exponential_filter_angle(np.array(df.rawa), alpha, theta)
# Less noise and smaller values.  Theta is much smaller.
df['saws'], _ = p.exponential_filter(np.array(df.raws), 0.96, 0.8)
df['sspd'], _ = p.exponential_filter(np.array(df.rspd), 0.8, 0.5)

c.quick_plot(df.index, (df.rawa, df.sawa, df.twa), "rawa, sawa twa".split())
c.quick_plot(df.index, (df.raws, df.saws, df.rspd, df.sspd), "raws saws rspd ssped".split())

sdf = rdf
sdf = df.loc[example.begin : example.end]
chart = c.plot_track(sdf, 1)

def estimate_true_wind(epsilon, df, awa_mult=1.0, aws_mult=1.0, spd_mult=1.0):
    if 'tws' in df.columns:
        initial_tws = df.tws.iloc[0]
        initial_twd = df.twd.iloc[0]
    else:
        initial_tws = df.aws.iloc[0]
        initial_twd = 0
    return estimate_true_wind_helper(epsilon,
                                     aws = aws_mult * np.asarray(df.raws),
                                     awa = awa_mult * np.asarray(df.rawa),
                                     hdg = np.asarray(df.rhdg),
                                     spd = spd_mult * np.asarray(df.rspd),
                                     tws = initial_tws,
                                     twd = initial_twd,
                                     variation = df.variation.mean())


for i, m in zip(it.count(), np.linspace(0.9, 1.0, 7)):
    (twd, tws, res_n, res_e) = estimate_true_wind(0.0002, sdf, m, 1.0, 1.0)
    c.quick_plot(sdf.index, (sdf.twd-180, 10*sdf.tws, twd-180, 10*tws, sdf.awa, sdf.rtwa, sdf.hdg-180, 10*sdf.sspd, 10*sdf.saws),
                 "twd tws etwd etws awa twa hdg spd saws".split(),
                 title = f"awa m = {m}",
                 fignum = i)
    print("*****************************************")
    print(i, m)
    print(np.mean(res_e), np.std(res_e))
    print(np.mean(res_n), np.std(res_n))


for i, m in zip(it.count(), np.linspace(0.9, 1.0, 7)):
    (twd, tws, res_n, res_e) = estimate_true_wind(0.0002, sdf, m, 1.0, 1.0)
    c.quick_plot(sdf.index, (twd, 10*tws, sdf.awa, 10*sdf.sspd, 10*sdf.saws),
                 "etwd etws awa spd saws".split(),
                 title = f"awa m = {m}",
                 fignum = i)
    # c.quick_plot(sdf.index, (twd, 10*tws, sdf.awa, sdf.hdg, 10*sdf.sspd, 10*sdf.saws),
    #              "etwd etws awa hdg spd saws".split(),
    #              title = f"awa m = {m}",
    #              fignum = i)
    # c.quick_plot(sdf.index, (twd, 10*tws, 10*sdf.sspd, 10*sdf.saws),
    #              "etwd etws spd saws".split(),
    #              title = f"awa m = {m}",
    #              fignum = i)
    print("*****************************************")
    print(i, m)
    print(np.mean(res_e), np.std(res_e))
    print(np.mean(res_n), np.std(res_n))
                 
             

#              fignum = 1)


