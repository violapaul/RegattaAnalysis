import os

import copy
import math
import itertools as it
import importlib

import pandas as pd
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt

from boat_shape import OUTLINE as BOAT_OUTLINE

import process as p
import chart as c
import global_variables as G
from utils import DictClass

pd.set_option('display.max_rows', 100)
pd.reset_option('precision')
pd.set_option('display.float_format', lambda x: '%.3f' % x)

np.set_printoptions(precision=4, suppress=True, linewidth = 180)



################################################################

def analyze_gps_receivers(df):
    # Two GPS units on the boat, Zeus inside the cabin, ZG100 on the back rail.  About 3m
    # apart.
    # Reason to beleive that the ZG100 is better (its outside and a dedicated unit).
    # The ZG100 may have a programmed offset.  Got to figure that out.

    pdf = df[::10]
    track = np.vstack(MAP(pdf.longitude.to_numpy(), pdf.latitude.to_numpy())).T
    zeus_track = np.vstack(MAP(pdf.zeus_longitude.to_numpy(), pdf.zeus_latitude.to_numpy())).T

    mean = track.mean(0)
    track = track - mean
    zeus_track = zeus_track - mean

    # Just look at the tracks
    _, ax = new_axis(fignum=1, equal=True)
    ax.plot(track[:, 0], track[:, 1])
    ax.plot(zeus_track[:, 0], zeus_track[:, 1])

    # Shows that the differences are related to heading of the boat
    _, ax = new_axis(fignum=2, equal=False)
    ax.plot(pdf.heading, (track - zeus_track)[:, 0], linestyle = 'None', marker='.')
    ax.plot(pdf.heading, (track - zeus_track)[:, 1], linestyle = 'None', marker='.')

    # North and east differences, in meters
    _, ax = new_axis(fignum=3, equal=False)
    ax.plot((track - zeus_track)[:, 0])
    ax.plot((track - zeus_track)[:, 1])


def analyze_speeds(df, fignum=1, title="boat speed"):
    fig, ax = new_axis(fignum=fignum, clf=True)
    ax.plot(df.sog, linestyle = 'None', marker='.')
    ax.plot(df.spd, linestyle = 'None', marker='.')
    fig.suptitle(title, fontsize=14, fontweight='bold')


def cog_vs_heading():
    # Two sources of boat, heading (compass heading): the RC42 compass and the VG100 GPS
    # unit.  Compass appears to be more reliable.
    #
    # Both return magnetic north, and not true north.  The declination is currently 15.6
    # (and this is stored in the logs as df.variation).

    threshold = 60
    mdf = bdf[(bdf.smooth_sog > 0.5) & (bdf.awa < threshold) & (bdf.awa > -threshold)]
                                        
    unwrap_true_heading = np.degrees(np.unwrap(np.radians(mdf.heading))) + mdf.variation.mean()

    # clever trick to unwrap a second signal so that it's difference to a base is
    # minimized.
    delta = np.fmod(mdf.cog - mdf.unwrap_true_heading, 360)
    delta[delta > 180] = delta[delta > 180] - 360
    delta[delta < -180] = delta[delta < -180] + 360
    mdf['unwrap_cog'] = mdf.unwrap_true_heading + delta

    plt.clf()
    mdf.unwrap_cog.plot()
    mdf.unwrap_true_heading.plot()


def plot_polars(df, fignum=None):

    sss = slice(76020, 107090, 5)
    sdf = df[sss]

    polar_fig = plt.figure(num=fignum)
    polar_fig.clf()
    ax = polar_fig.add_subplot(111, projection='polar')
    ax.set_rmax(10)
    ax.set_rlabel_position(-22.5)
    ax.grid(True)
    ax.set_theta_offset(math.pi / 2.0)
    ax.set_theta_direction(-1)  # Clockwise (not sure why this is not default)

    tws_knots = (sdf.tws * MS_2_KNOTS) # was meters per second
    tws_index = 2 * (tws_knots / 2).round()
    legend_list = []
    for speed in [6, 8, 10, 12]:
        fdf = sdf[tws_index == speed]
        angles = np.deg2rad(fdf.twa)
        speeds = fdf.spd * MS_2_KNOTS
        ax.plot(angles, speeds, linestyle = 'None', marker='.', markersize=3.0, alpha=0.5)
        legend_list.append(f'TWS {speed} knots.')

    plt.legend(legend_list, loc='best')
    plt.show()

    vmg = sdf.spd * MS_2_KNOTS * np.cos(np.radians(sdf.twa))

    quick_plot(sdf.index, (sdf.tws*MS_2_KNOTS, sdf.twa/27, sdf.spd * MS_2_KNOTS, sdf.awa/27, vmg),
               legend = "tws, twa/27, spd, awa/27, vmg".split(),
               fignum=2, clf=True, title='raw data')


def display_sensors(df):
    pdf = df[::10]
    plt.figure()
    plt.clf()
    pdf.rawa.plot(linestyle = 'None', marker='.', color='lightcoral')
    pdf.awa.plot(marker='.', color='red')
    (20 * pdf.rspd).plot(linestyle = 'None', marker='.', color='cyan')
    (20 * pdf.spd).plot(marker='.', color='blue')
    (10 * pdf.raws).plot(linestyle = 'None', marker='.', color='lime')
    (10 * pdf.aws).plot(marker='.', color='green')
    (20 * pdf.rsog).plot(linestyle = 'None', marker='.', color='magenta')
    (20 * pdf.sog).plot(marker='.', color='purple')
    (1 * pdf.rudder_position).plot(marker='.')
    pdf.rtwd.plot(marker='.')
    pdf.hdg.plot(marker='.')
    (20 * pdf.rtws).plot(marker='.')

    plt.legend(('rawa', 'awa',
                'rspd', 'spd 20x',
                'raws', 'aws x10',
                'rsog', 'sog x20',
                'rudder', 'rtwd',
                'hdg', 'rtws'
    ),
               loc='best')
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.clf()
    pdf.rcog.plot(linestyle = 'None', marker='.', color='lightcoral')
    pdf.cog.plot(marker='.', color='red')
    (20 * pdf.rsog).plot(linestyle = 'None', marker='.', color='magenta')
    (20 * pdf.sog).plot(marker='.', color='purple')
    plt.legend(('rcog', 'cog', 'rsog', 'sog'),
               loc='best')
    plt.grid(True)
    plt.show()


def merge_dicts(base_dict, update_dict):
    "Create new dict with updated key/values."
    res = copy.copy(base_dict)
    res.update(update_dict)
    return res


def sail_simulator():
    """
    Simulate sailing to create a set of sensor measurements that we can then analyze with
    ground truth.
    """
    # Where to get the trajectory??  How about tacking upwind for 30 seconds at a time?
    # Wind and currrent are random (small, slow perturbations)
    # feedback loop on rudder (or HDG directly) designed to maximize VMG (polars??)
    # some noise from waves (the boat is buffetted in velocity and orientation)
    pass


def true_wind(df):
    tdf = df.copy()

    # Some examples
    base = dict(cog=90, sog=2, hdg=90, awa=-60, spd=4)
    tdf = pd.DataFrame([
        base,
        merge_dicts(base, dict(cog=100)),
        merge_dicts(base, dict(cog=100, hdg=100))
    ])

    ccoeff = chebyshev_filter(0.1, 5)
    bcoeff = butterworth_filter(0.2, 5)
    acoeff = local_average_filter(31)
    slow_coeff = butterworth_filter(0.03, 5)

    coeff = bcoeff
    # Approach: No GPS, relative coords, "north" is direction of boat.  The only motion of
    # the boat is SPD times HDG.
    causal = False
    rawa = p.sign_angle(df.rawa)
    rtwa = p.sign_angle(df.rtwa)
    awa = p.smooth_angle(coeff, rawa, causal=causal)
    aws = p.smooth(coeff, df.raws, causal=causal)
    spd = p.smooth(coeff, df.rspd, causal=causal)
    hdg = p.smooth_angle(coeff, df.rhdg, causal=causal)
    aw_n = aws * north_d(awa)
    aw_e = aws * east_d(awa)
    tw_n = aw_n - spd
    twa = np.degrees(np.arctan2(aw_e, tw_n))
    tws = p.smooth(slow_coeff, np.sqrt(np.square(tw_n) + np.square(aw_e)))
    twd = p.compass_angle(smooth_angle(slow_coeff, twa + hdg + 15.3, causal=causal))

    quick_plot(df.index, (df.rtws, tws, rtwa/20, twa/20, df.rtwd/20, twd/20, aws, spd, awa/20),
               legend = "df.rtws, tws, rtwa/20, twa/20, df.rtwd/20, twd/20, aws, spd, awa/20".split(),
               fignum=2, clf=True, title='aws', s=slice(None, None, 10))

    # Approach: Use COG to compute motion, not HDG
    rawa = sign_angle(df.rawa)
    rtwa = sign_angle(df.rtwa)

    for o in np.linspace(1, 1.5, num=10):
        awa = smooth_angle(coeff, rawa, causal=causal)
        aws = 1.0 * smooth(coeff, df.raws, causal=causal)
        hdg = -4.0 + smooth_angle(coeff, df.rhdg, causal=causal) + 15.4
        sog = smooth(coeff, df.rsog, causal=causal)
        cog = smooth_angle(coeff, df.rcog, causal=causal)
        awd = awa + hdg
        aw_n  = aws * north_d(awd)
        aw_e  = aws * east_d(awd)
        vog_n = sog * north_d(cog)
        vog_e = sog * east_d(cog)
        tw_n = aw_n - vog_n
        tw_e = aw_e - vog_e
        if True:
            twd = compass_angle(smooth_angle(slow_coeff, np.degrees(np.arctan2(tw_e, tw_n))))
            tws = smooth(slow_coeff, np.sqrt(np.square(tw_n) + np.square(tw_e)))
        else:
            twd = compass_angle(np.degrees(np.arctan2(tw_e, tw_n)))
            tws = np.sqrt(np.square(tw_n) + np.square(tw_e))

        s = slice(85000, 105000)
        print(f'{o:.4f}, {twd[s].std():.5f}, {twd[s].mean()}, {tws[s].std():.5f}, {tws[s].mean()}')
        
    quick_plot(df.index, (tws, twd/36, aws, awa/36, awd/36, hdg/36, cog/36, sog, df.rudder_position/10),
               legend =  "tws, twd/36, aws, awa/36, awd/36, hdg/36, cog/36, sog df.rudder_position/10".split(),
               fignum=3, clf=True, title='aws', slice=slice(None, None, 10))

    quick_plot(df.index, (spd, hdg/36, cog/36, sog, df.rsog, df.rudder_position/10, df.turn_rate, df.zg100_turn_rate),
               legend =  "spd, hdg/36, cog/36, sog, df.rsog, df.rudder_position/10, df.turn_rate".split(),
               fignum=3, clf=True, title='aws', slice=slice(None, None, 10))
    
    quick_plot(df.index, (df.rtws, tws, df.rtwd/36, twd/36, aws, awa/36, awd/36, hdg/36, cog/36, sog),
               legend =  "df.rtws, tws, df.rtwd/36, twd/36, aws, awa/36, awd/36, hdg/36, cog/36, sog".split(),
               fignum=2, clf=True, title='aws', slice=slice(None, None, 10))

    quick_plot(df.index, (df.rtws, tws, df.rtwd/36, twd/36),
               legend = "(df.rtws, tws, df.rtwd/36, twd/36".split(),
               fignum=3, clf=True, title='aws', slice=slice(None, None, 10))
    

    quick_plot(df.index, (twd/36, aw_n, aw_e, sog, cog/36),
               legend = "(twd/36, aw_n, aw_e, sog, cog/36)".split(),
               fignum=3, clf=True, title='aws', slice=slice(None, None, 10))
    

    
    
    awd = tdf.awa + tdf.heading
    
    # Approach 1: True velocity is SOG times COG.
    vog_n = tdf.sog * north_d(tdf.cog)
    vog_e = tdf.sog * east_d(tdf.cog)

    
    
    vaw_world  = tdf.wind_speed * ( cos_d(awa_world) + 1j * sin_d(awa_world))

    vtw_world = vaw_world - vboat_world

    tdf['twa'] = twa_world = np.degrees(np.angle(vtw_world))
    tdf['tws'] = tws_world = np.abs(vtw_world)


def speed_multiplier(df):
    "Least squares best muiltiplier for speed to match SOG."
    # alpha * x = b, find alpha
    xTb = (df.speed_water_referenced * df.sog).sum()
    xTx = (df.speed_water_referenced * df.speed_water_referenced).sum()
    return (xTb / xTx)


def find_best_delay(df, graph=False):
    # Only use the time when the boat is moving significantly.
    runs = p.find_runs(np.asarray(df.sog > 1.0))
    first, last = sorted(runs, key=lambda a: a[1] - a[0])[-1]
    print(f"Boat is moving consistently from {first} to {last}.")
    if (last - first)/G.SAMPLES_PER_SECOND < (60 * 20):
        print(f"Warning that is not very long!")
        return(0)
    sdf = df[(first + 1):(last - 1)]

    # Unwrap,  because these are angles
    sig1 = np.unwrap(np.array(sdf.cog))
    sig2 = p.deg(p.match_wrap(p.rad(sig1), p.rad(np.array(sdf.hdg) + 15.4)))

    # High pass filter (by subtracting lowpass).  This gets rid of nuisance DC offsets and
    # emphasizes regions of rapid change...  which are the only regions where delay is
    # observable.
    coeff = p.butterworth_filter(cutoff=0.1, order=5)
    hp_sig1 = sig1 - p.smooth(coeff, sig1)
    hp_sig2 = sig2 - p.smooth(coeff, sig2)

    # Normalize (for correlation)
    nsig1 = (hp_sig1 - hp_sig1.mean()) / (np.std(hp_sig1) * np.sqrt(len(hp_sig1)))
    nsig2 = (hp_sig2 - hp_sig2.mean()) / (np.std(hp_sig2) * np.sqrt(len(hp_sig2)))

    # Compute correlation (dot product) for various leads and lags
    plus_minus_window = 50
    res = np.correlate(nsig1, nsig2[plus_minus_window:-plus_minus_window])
    delay = np.argmax(res) - plus_minus_window
    print(f"Found a delay of {delay}")

    if graph:
        quick_plot(df.index, (df.cog, df.hdg + 15.4), "cog, hdg".split(), fignum=1)
        quick_plot(sdf.index, (sig1, sig2), "cog, hdg".split(), fignum=2)
        quick_plot(sdf.index, (hp_sig1, hp_sig2), "cog, hdg".split(), fignum=3)
        quick_plot(sdf.index, (nsig1, nsig2), "cog, hdg".split(), fignum=4)
        quick_plot(None, [res], ["correlation"], fignum=5)
        quick_plot(None, (nsig1[delay:], nsig2[:-delay]), "cog hdg".split(), fignum=6)
        quick_plot(df.index[:-delay], (df.cog[delay:], df.hdg[:-delay]+15.4), "cog hdg".split(), fignum=7)
        quick_plot(df.index[:-delay], (df.rcog[delay:], df.rhdg[:-delay]+15.4), "cog hdg".split(), fignum=8)

    return delay


def sara_plot():
    udf = df[74920:109430].copy()
    udf['vmg'] = udf.sog * cos_d(udf.cog-udf.twd)

    plt.figure(1)
    plt.clf()
    legend=[]
    for i in [3, 4, 5, 6]:
        ndf = udf[(udf.tws < (i + 0.5)) & (udf.tws > (i - 0.5))]
        plt.plot(ndf.awa, ndf.vmg, marker='.', markersize=3.0, alpha=0.1)
        legend.append(f"TWS = {2*i}")
    plt.legend(legend, loc='best')


def test():
    sdf = df[5000:15000]
    
    # Unwrap,  because these are angles
    coeff = p.butterworth_filter(cutoff=0.01, order=1)
    sig1 = np.unwrap(np.array(sdf.twa))
    sawa = p.smooth_angle(coeff, sdf.awa)
    sawa = scipy.ndimage.median_filter(sdf.awa, int(30 / dt), origin=0)
    sig2 = p.deg(p.match_wrap(p.rad(sig1), p.rad(np.array(sawa))))
    
    quick_plot(sdf.index, (sig1, sdf.awa, sawa), "twa awa sawa".split(), fignum=2)

    # High pass filter (by subtracting lowpass).  This gets rid of nuisance DC offsets and
    # emphasizes regions of rapid change...  which are the only regions where delay is
    # observable.
    coeff = p.butterworth_filter(cutoff=0.005, order=1)
    hp_sig1 = sig1 - p.smooth(coeff, sig1)
    hp_sig2 = sig2 - p.smooth(coeff, sig2)
    quick_plot(sdf.index, (hp_sig1, hp_sig2), legend, fignum=3)

    # Normalize (for correlation)
    nsig1 = (hp_sig1 - hp_sig1.mean()) / (np.std(hp_sig1) * np.sqrt(len(hp_sig1)))
    nsig2 = (hp_sig2 - hp_sig2.mean()) / (np.std(hp_sig2) * np.sqrt(len(hp_sig2)))

    # Compute correlation (dot product) for various leads and lags
    plus_minus_window = 400
    res = np.correlate(nsig1, nsig2[plus_minus_window:-plus_minus_window])
    delay = np.argmax(res) - plus_minus_window
    print(f"Found a delay of {delay}")

    if graph:
        legend = "twa awa".split()
        quick_plot(sdf.index, (sig1, sig2), legend, fignum=2)
        quick_plot(sdf.index, (hp_sig1, hp_sig2), legend, fignum=3)
        quick_plot(sdf.index, (nsig1, nsig2), legend, fignum=4)
        quick_plot(None, [res], ["correlation"], fignum=5)
        quick_plot(None, (nsig1[:-delay], nsig1[delay:], nsig2[:-delay]), legend, fignum=6)
        quick_plot(None, (nsig1[delay:], nsig2[:-delay]), legend, fignum=6)
        quick_plot(None, (sig1[:-delay], sig1[delay:], sig2[:-delay]), legend, fignum=7)
        quick_plot(df.index[:-delay], (df.rcog[delay:], df.rhdg[:-delay]+15.4), "cog hdg".split(), fignum=8)


def draw_arrow(ax, begin, end, color='red'):
    delta = end - begin
    ax.arrow(begin[0], begin[1], delta[0], delta[1], head_width=0.2, length_includes_head=True, color=color)

# COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
#           '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

COLORS = 'red green blue orange turquoise brown darkmagenta orange'.split()


def old_boat_forces(row, fignum=None):
    
    ss = slice(tdf.iloc[0].name, tdf.iloc[-1].name)
    sdf = df.loc[ss]
    row = tdf.iloc[3]
    row2 = df.loc[row.name+13]
    fignum = 1
    boat_relative = False

    causal = True
    scoeff = p.butterworth_filter(0.3, 2)
    cdf = df.copy()
    cawa = p.smooth_angle(scoeff, df.awa, causal=causal)
    cdf['saws'] = p.smooth(scoeff, df.aws, causal=causal)
    
    cdf['sawa'], betas = exponential_filter_angle(np.asarray(df.rawa), 0.99, np.radians(6))
    cdf['saws'], _ = exponential_filter(np.asarray(df.raws), 0.97, 1)

    pdf = cdf[ss]
    pdf = cdf
    quick_plot(pdf.index, (pdf.rawa, cawa, pdf.awa, pdf.sawa),
               legend = "(pdf.rawa, cawa, pdf.awa, pdf.sawa)".split(),
               fignum=2, clf=True, title='aws')

    pdf = cdf
    quick_plot(pdf.index, (pdf.rawa, pdf.awa, pdf.sawa, pdf.rawa - pdf.sawa, 10*betas),
               legend = "(pdf.rawa, pdf.awa, pdf.sawa, diff beta)".split(),
               fignum=2, clf=True, title='aws')
    
    quick_plot(pdf.index, (pdf.tws, pdf.twa/20, pdf.aws, pdf.awa/20, pdf.spd, pdf.saws, pdf.sawa/10),
               legend = "(pdf.tws, pdf.twa/20, pdf.aws, pdf.awa/20, pdf.spd, saws, sawa/10)".split(),
               fignum=2, clf=True, title='aws')

    pdf = cdf[ss]    
    row = pdf.iloc[4]
    
    delta_t = 1
    length = 10.5


def boat_forces(row, fignum, boat_relative=True):
    hdg = row.hdg + 16
    delta_t = 1

    # either use the global coordinates, or the boat coordinates
    if boat_relative == True:
        # Display everything with boat hdg up.
        boat_to_figure = rotation_mat(0)
        world_to_figure = rotation_mat(-hdg)
    else:
        boat_to_figure = rotation_mat(hdg)
        world_to_figure = rotation_mat(0)

    fig, ax = new_axis(fignum, equal=True, clf=True)

    length = 10.5
    boat = BOAT_OUTLINE * length
    boat = np.dot(boat, boat_to_figure)
    ax.plot(boat[:, 0], boat[:, 1], alpha=0.5, color='darkgreen')

    vog = delta_t * row.sog * np.array([sin_d(row.cog), cos_d(row.cog)])
    vog = np.dot(vog, world_to_figure)
    draw_arrow(ax, (0, 0), vog, color='red')

    btv = delta_t * row.spd * np.array([sin_d(hdg), cos_d(hdg)])
    btv = np.dot(btv, world_to_figure)
    draw_arrow(ax, (0, 0), btv, color='darkgreen')

    vaw = delta_t * row.aws * np.array([sin_d(row.awa), cos_d(row.awa)])
    vaw = np.dot(vaw, boat_to_figure)
    draw_arrow(ax, vaw, (0, 0), color='orange')

    vtw = delta_t * row.tws * np.array([sin_d(row.twa), cos_d(row.twa)])
    vtw = np.dot(vtw, boat_to_figure)
    draw_arrow(ax, vtw, (0, 0), color='purple')

    vtwd = delta_t * row.tws * np.array([sin_d(row.twd), cos_d(row.twd)])
    vtwd = np.dot(vtwd, world_to_figure)
    draw_arrow(ax, vtwd, (0, 0), color='turquoise')
    
    draw_arrow(ax, vaw, vaw-btv, 'darkgreen')
    draw_arrow(ax, vaw, vaw-vog, 'red')


def mystery():
    plt.figure(3)
    cdf.twd.plot()
    cdf.rtwd.plot()
    
    quick_plot(cdf.index, (cdf.rawa, cdf.awa, cdf.zg100_roll))
    
    
    (cdf.twa + cdf.hdg).plot()

    coeff = p.butterworth_filter(cutoff=0.1, order=5)
    sig1 = np.unwrap(np.array(cdf.awa))    
    hp_sig1 = sig1 - p.smooth(coeff, sig1)
    hp_sig2 = cdf.zg100_roll - p.smooth(coeff, cdf.zg100_roll)

    # Normalize (for correlation)
    nsig1 = (hp_sig1 - hp_sig1.mean()) / (np.std(hp_sig1) * np.sqrt(len(hp_sig1)))
    nsig2 = (hp_sig2 - hp_sig2.mean()) / (np.std(hp_sig2) * np.sqrt(len(hp_sig2)))

    # Compute correlation (dot product) for various leads and lags
    plus_minus_window = 50
    res = np.correlate(nsig1, nsig2[plus_minus_window:-plus_minus_window])
    delay = np.argmax(res) - plus_minus_window
    print(f"Found a delay of {delay}")

    c.quick_plot(None, (nsig1[delay:], nsig2[:-delay]), "roll awa".split(), fignum=6)
    c.quick_plot(None, (hp_sig1[delay:], hp_sig2[:-delay]), "roll awa".split(), fignum=7)    
    c.quick_plot(None, (hp_sig1, hp_sig2), "roll awa".split(), fignum=7)    
    
    c.quick_plot(cdf.index, (hp_sig1, hp_sig2))
    


def plot_boat(df, time_slice, delay=40, fignum=None):
    delay = 40
    delay_slice = slice(time_slice.start + delay, time_slice.stop + delay, time_slice.step)
    sdf = df.loc[time_slice]
    ddf = df.loc[delay_slice]

    chart = c.create_chart(df, border=0.5)
    image = c.desaturate_image(chart.image)

    length = 10.5
    scaled_boat = BOAT_OUTLINE * length

    fig = plt.figure(fignum)
    fig.clf()
    ax = fig.subplots(1, 1)
    ax.imshow(image, extent=[0, chart.east - chart.west, 0, chart.north - chart.south])
    line, = ax.plot(chart.track[:, 0], chart.track[:, 1], alpha=0.5, color='teal')
    ax.grid(True)

    # TODO: scale for vectors (probably time...  assuming its uniform)
    for loc, (i, row), (j, drow) in zip(chart.track, sdf.iterrows(), ddf.iterrows()):
        hdg = row.hdg + 15.3
        rotated_boat = np.dot(scaled_boat, rotation_mat(hdg))
        shifted_boat = rotated_boat + loc
        ax.plot(shifted_boat[:, 0], shifted_boat[:, 1], alpha=0.5, color='darkgreen')
        ax.arrow(loc[0], loc[1], 5 * drow.sog * sin_d(drow.cog), 5 * drow.sog * cos_d(drow.cog), head_width=1, color='red')
        ax.arrow(loc[0], loc[1], 5 * row.spd * sin_d(hdg), 5 * row.spd * cos_d(hdg), head_width=1, color='blue')


def rotation_mat(degrees_from_north):
    # Code below can work if passed an array of angles, just be careful on the return.
    rad = np.radians(degrees_from_north)
    c, s = np.cos(rad), np.sin(rad)
    res = np.vstack((c, -s, s, c)).T
    if np.isscalar(degrees_from_north):
        return res.reshape(2, 2)
    else:
        return res.reshape(-1, 2, 2)


def make_updater(func, fig):
    def update(val):
        func(val)
        fig.canvas.draw_idle()
    return update


def trim_dock_time(df):
    pass


def find_cog():
    for i, df in zip(it.count(), dfs):
        print(i, df.filename[0], len(df))
        if 'cog' in df.columns:
            print("   ", df.cog.isna().sum())


def compute_speed_multiplier(df):
    df = bdf

    # First get rid of the near zero values
    non_zero = df[(df.speed_water_referenced > 0.5) & (df.sog > 0.5)]

    # best overall multiplier
    speed_multiplier(non_zero)

    # how about port vs stbd?

    # Roll sensor is likely not quite straight.
    roll_offset = non_zero.zg100_roll.mean()
    port = non_zero[non_zero.zg100_roll > roll_offset]
    stbd = non_zero[non_zero.zg100_roll < roll_offset]

    speed_multiplier(port)
    speed_multiplier(stbd)

    # Using CW wind angle (0 to 360), stbd is less than 180, port is greater than 180
    port = non_zero[non_zero.awa > 180]
    stbd = non_zero[non_zero.awa < 180]    

    speed_multiplier(port)
    speed_multiplier(stbd)
    # The overall multiplier appears to be 1.08

def determine_roll_sign(df):
    # First determine if we are close hauled on port or stbd.
    # Port should AWA should be about 340 ish?

    df = bdf.copy()
    # Create a signed apparent wind angle
    df['awa'] = df.awa.copy()
    df.loc[df.awa > 128, 'awa'] = df.loc[df.awa > 180, 'awa'] - 360

    # we are close hauled if the awa is less than 40, and speed is fast.
    threshold = 60
    close_hauled = df[(df.awa < threshold) & (df.awa > -threshold) & (df.speed_water_referenced > 2.0)]

    plt.clf()
    plt.plot(close_hauled.awa + 1.59, linestyle='None', marker='.')  # linestyle = 'None'
    plt.plot(close_hauled.zg100_roll + 3.889, linestyle='None', marker='.')

    close_hauled[['awa', 'zg100_roll']].corr()

    # Roll and awa, when close hauled are negatively correlated: -0.8.
    # Assuming awa is positive on stbd, then roll is negative on stbd.

    fig = plt.figure(1)
    fig.clf()
    ax = fig.add_subplot(111)
    ax.plot(df.awa, linestyle = 'None', marker='.')
    fig.suptitle("wind angle", fontsize=14, fontweight='bold')


SAIL_LOGS = [
    DictClass(log='2019-10-04_16:43.pd', doc='Sail to Everett for Foulweather Bluff Race.'),
    DictClass(log='2019-10-05_09:18.pd', doc='Foulweather Bluff Race'),
    DictClass(log='2019-10-11_17:38.pd', doc='Short practice, upwind tacks and downwind jibes.'),
    DictClass(log='2019-10-12_09:45.pd', doc='CYC PSSC Day 1', begin=19081, end=233893),
    DictClass(log='2019-10-18_13:51.pd', doc='Short, at dock.'),
    DictClass(log='2019-10-19_09:45.pd', doc='STYC Fall Regatta.'),
    DictClass(log='2019-10-26_09:40.pd', doc='Grand Prix Saturday.', begin=40503, end=87408),
    DictClass(log='2019-10-26_12:35.pd', doc='Short, at dock.'),
    DictClass(log='2019-11-07_12:46.pd', doc='Short, at dock.'),
    DictClass(log='2019-11-16_10:09.pd', doc='Snowbird #1.', begin=41076, end=111668),
    DictClass(log='2019-11-23_10:23.pd', doc='Practice.'),
    DictClass(log='2019-11-24_10:33.pd', doc='Practice.'),
    DictClass(log='2019-11-29_12:54.pd', doc='Longer.  At dock for data collection.'),
    DictClass(log='2019-12-06_11:25.pd', doc='Short, at dock.'),
    DictClass(log='2019-12-07_09:47.pd', doc='Snowbird #2.', begin=54316, end=109378),
    DictClass(log='2020-02-08_10:23.pd', doc='Snowbird #4 Part1.', begin=34000, end=139000, twd=True),
    DictClass(log='2020-02-08_13:36.pd', doc='Snowbird #4 Part2.', begin=0000, end=1009378, twd=True),    
]

def test():
    importlib.reload(p)
    DATA_DIRECTORY = '/Users/viola/canlogs'
    example = DictClass(log='2019-11-16_10:09.pd', begin=76890, end=105810)
    example = DictClass(log='2019-12-07_09:47.pd', begin=101881, end=109567)
    example = DictClass(log='2019-12-07_09:47.pd', doc='Snowbird #2.', begin=54316, end=109378)
    example = DictClass(log='2019-10-26_09:40.pd', doc='Grand Prix Saturday.', begin=40503, end=87408)
    example = DictClass(log='2019-10-12_09:45.pd', doc='CYC PSSC Day 1', begin=19081, end=233893)
    example = DictClass(log='2019-11-16_10:09.pd', doc='Snowbird #1.', begin=42548, end=111668)
    
    sail_logs = [example.log]

    dfs, bdf = p.read_sail_logs(sail_logs, skip_dock_only=False, trim=True, path=DATA_DIRECTORY, cutoff=0.3)
    df = dfs[0]
    chart = c.plot_track(df, 1)
    
    sdf = df.loc[example.begin : example.end]
    schart = c.plot_track(sdf, 3)
    


def onetime():
    DATA_DIRECTORY = '/Users/viola/canlogs'
    mbtiles_file = "/Users/viola/Downloads/MBTILES_06.mbtiles"
    map_width = 10000
    # Create the very large basemap
    create_base_map(mbtiles_file, map_width)


