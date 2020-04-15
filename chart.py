"""
Collection of tool to create sailing charts, display race tracks, and race data.

Most interesting part is the generation of GEO-registered charts, upon which lat/lon
positions can be scale accurately ploted.
"""
import os
import itertools as it
import time

import math

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets

import cv2

from utils import DictClass

import process as p
from global_variables import G

# PLOTTING HELPERS ###################################################################################

def draw_arrow(ax, begin, end, color='red'):
    "Draw an arrow on the AXIS."
    delta = end - begin
    ax.arrow(begin[0], begin[1], delta[0], delta[1], head_width=0.2, length_includes_head=True, color=color)


def new_axis(fignum=None, equal=False, clf=True):
    "Convenience to create an axis with optional CLF and EQUAL."
    fig = plt.figure(fignum)
    if clf:
        fig.clf()
    ax = fig.add_subplot(111)
    if equal:
        ax.axis('equal')
    return fig, ax


def quick_plot_ax(ax, index, data, legend=None, s=slice(None, None, None)):
    if index is None:
        index = range(len(data[0]))
    else:
        index = index[s]
    for d in data:
        ax.plot(index, d[s])
    if legend is not None:
        ax.legend(legend, loc='best')
    ax.grid(True)


def quick_plot(index, data, legend=None, fignum=None, clf=True, title=None, s=slice(None, None, None), ylim=None):
    """
    Super quick tool to display a set of associated data on a single axis.

    All data is assumed to share a single index (X axis).  All data is the same length.

    DATA is a list of Y axis data.
    LEGEND is a list of legend names

    So if you have your data in a DataFrame this works great.

    quick_plot(df.index, (df.one, df.two, df.three), ['one', 'two', 'three'])

    OR

    quick_plot(df.index, (df.one, df.two, df.three), "(df.one, df.two, df.three)".split())

    Note, that in the second case I just copied the text of the DATA param into the legend and let split work it out.
    """
    if isinstance(fignum, matplotlib.figure.Figure):
        fig = fignum
    else:
        fig = plt.figure(num=fignum)
    if clf:
        fig.clf()
    ax = fig.add_subplot(111)
    quick_plot_ax(ax, index, data, legend=legend, s=s)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if title is not None:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.tight_layout()


# Code to support plotting on charts ######################################################

def create_base_map(map_width=10000):
    """
    One time task to create a single chart of a SAILING AREA from the NOAA MBTILES file.
    The result is much smaller than the MBTILE and perhaps a bit faster to read (not
    clear).  Result single image is still quite large.
    """
    print(f"Creating base map for {G.LOCALE}")
    
    # Create an area a bit bigger than the data
    lat_max, lat_min = G.LAT_MAX_MIN
    lon_max, lon_min = G.LON_MAX_MIN

    # Setup gdalwarp args
    # Define the extent of the map in lon/lat
    te_arg = f"-te {lon_min:.7f} {lat_min:.7f} {lon_max:.7f} {lat_max:.7f}"
    t_srs_arg = f"-t_srs '{G.PROJ4}'"

    zoom = "-oo ZOOM_LEVEL=16"
    zoom = ""
    command = "gdalwarp"
    command += " " + zoom
    command += " " + t_srs_arg
    command += " " + te_arg
    command += " " + f"-te_srs EPSG:4326 -ts {map_width} 0 -r bilinear"
    command += " " + "-of vrt"
    command += f" {G.MBTILES_FILE} /tmp/chart.vrt"
    print(command)
    os.system("rm /tmp/chart.vrt")
    os.system(command)

    # command = "gdal_translate -co compress=LZW seattle.vrt seattle.tif"
    command = "gdal_translate"
    command += " -co COMPRESS=JPEG -co TILED=YES"
    command += f" /tmp/chart.vrt {G.BASE_MAP_PATH}"
    os.system(f"rm {G.BASE_MAP_PATH}")
    os.system(command)

    command = "gdaladdo --config COMPRESS_OVERVIEW JPEG --config INTERLEAVE_OVERVIEW PIXEL"
    command += " -r average"
    command += f" {G.BASE_MAP_PATH}"
    command += " 2 4"
    os.system(command)


def desaturate_image(im, factor=2):
    hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
    hsv[:, :, 1] = hsv[:, :, 1] // factor
    val_offset = ((factor - 1) * 255) // factor
    hsv[:, :, 2] = val_offset + (hsv[:, :, 2] // factor)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def create_chart(df, border=0.2):
    # Add/sub just a bit to make the map interpretable for small excursions
    lat_max, lat_min, lat_mid = p.max_min_mid(df.latitude, border) + (0.002, -0.002, 0.0)
    lon_max, lon_min, lon_mid = p.max_min_mid(df.longitude, border) + (0.002, -0.002, 0.0)

    # Find the extents of the map
    west, south = np.array(G.MAP(lon_min, lat_min))
    east, north = np.array(G.MAP(lon_max, lat_max))

    # Setup gdalwarp args
    # Define the extent of the map in lon/lat
    te = f"-te {lon_min:.7f} {lat_min:.7f} {lon_max:.7f} {lat_max:.7f}"
    t_srs = f"-t_srs '{G.PROJ4}'"
    # Size of the image should match the shape of the map
    if (east - west) > (north - south):
        ts = "-ts 2000 0"
    else:
        ts = "-ts 0 2000"
    chart_file = os.path.join('/tmp/chart.tif')

    zoom = "-oo ZOOM_LEVEL=16"
    zoom = ""
    command1 = f"gdalwarp {zoom} {te} {t_srs} -te_srs EPSG:4326 -r bilinear"
    command1 += " " + ts
    command1 += " " + G.BASE_MAP_PATH
    command1 += " " + chart_file

    os.system(f"rm {chart_file}")
    os.system(command1)
    image = cv2.imread(chart_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    track = np.vstack(G.MAP(np.asarray(df.longitude), np.asarray(df.latitude))).T - (west, south)

    return DictClass(image=image, west=west, east=east, north=north, south=south,
                     track=track)


def draw_track(df, chart, ax=None, subsample_rate=1, color='green', **args):
    """
    Convert the track from lat/lon to image coordinates and then draw.

    Optionally draw on a specific axis.
    """
    lon, lat = np.asarray(df.longitude), np.asarray(df.latitude)
    track = np.vstack(G.MAP(lon, lat)).T - (chart.west, chart.south)
    track = track[::subsample_rate]
    if ax is None:
        ax = chart.ax
    ax.plot(track[:, 0], track[:, 1], color=color, **args)


def plot_chart(df, fig_or_num=None, border=0.2):
    """
    Plot a track for race.  The background image comes from NOAA (though it has limited
    resolution if you sail long distances).
    """
    chart = create_chart(df, border=border)

    if isinstance(fig_or_num, matplotlib.figure.Figure):
        chart.fig = fig_or_num
    else:
        chart.fig = plt.figure(num=fig_or_num)
    chart.fig.clf()
    chart.ax = chart.fig.subplots(1, 1)
    chart.fig.tight_layout()
    image = desaturate_image(chart.image)
    chart.ax.imshow(image, extent=[0, chart.east - chart.west, 0, chart.north - chart.south])
    chart.ax.grid(True)
    return chart


# This was a challenge to get working well both in native python and in Jupyter.  
def plot_track(df, fignum=None, sliders=True, border=0.2, skip=None, delay=0.01):
    """
    Plot an interactive sail track on a map.  Provides sliders which can be used to trim
    the begin and end of the track.

    One potential use is to find the trim points so that only the race is
    displayed/analyzed.

    ch = plot_track(df)
    # Play with the UI, this happens in a different thread
    # At any point you can get the values of the sliders.
    print(ch.begin, ch.end)
    
    """
    chart = plot_chart(df, fig_or_num=fignum, border=border)
    if skip is None:
        skip = math.ceil(len(chart.track) / 2000)  # No more than 2000 points
    index = df.index[::skip]
    track = chart.track[::skip]

    fig = chart.fig
    line, = chart.ax.plot(track[:, 0], track[:, 1], color='red')

    if sliders:
        ax1 = fig.add_axes([0.05, 0.1, 0.03, 0.8], facecolor='lightgoldenrodyellow')
        ax2 = fig.add_axes([0.11, 0.1, 0.03, 0.8], facecolor='lightgoldenrodyellow')
        count = track.shape[0]
        s_beg = widgets.Slider(ax1, 'Begin', 0, count, valinit=0, orientation='vertical')
        s_end = widgets.Slider(ax2, 'End',   0, count-1, valinit=count-1, orientation='vertical')
        chart.begin, chart.end = index[0], index[-1]

        def update():
            track_begin, track_end = int(s_beg.val), int(s_end.val)
            chart.begin, chart.end = index[track_begin], index[track_end]
            line.set_data(track[track_begin:track_end, 0],
                          track[track_begin:track_end, 1])

            # fig.canvas.restore_region(axbackground)
            # line.axes.draw_artist(line)
            # chart.ax.figure.canvas.blit(chart.ax.bbox)
            fig.canvas.flush_events()

        s_beg.on_changed(lambda v: update())
        s_end.on_changed(lambda v: update())

    return chart


def show_boat_arrows(df, df_slice, dt_seconds=5, skip=2, current_scale=1):
    delay = 16
    dt = dt_seconds * G.SAMPLES_PER_SECOND
    scale = dt_seconds
    ss = slice(df_slice.start, df_slice.stop, dt)
    dss = slice(ss.start+delay, ss.stop+delay, dt)
    
    mdf = df.loc[ss]
    ddf = df.loc[dss]
    vog_n = scale * ddf.sog * p.north_d(ddf.cog)
    vog_e = scale * ddf.sog * p.east_d(ddf.cog)

    tw_n = scale * ddf.tws * p.north_d(ddf.twd)
    tw_e = scale * ddf.tws * p.east_d(ddf.twd)

    hdg = mdf.hdg + df.variation.mean()
    btv_n = scale * mdf.spd * p.north_d(hdg)
    btv_e = scale * mdf.spd * p.east_d(hdg)

    cur_n = current_scale * (np.asarray(vog_n) - np.asarray(btv_n))
    cur_e = current_scale * (np.asarray(vog_e) - np.asarray(btv_e))

    chart = plot_chart(mdf, 3, border=0.0)
    chart.mdf = mdf
    chart.ddf = ddf
    longitudes = np.asarray(mdf.longitude)
    latitudes = np.asarray(mdf.latitude)
    pos = np.vstack(G.MAP(longitudes, latitudes)).T - (chart.west, chart.south)

    color = 'blue'
    hwidth = scale/5
    for (east, north), ve, vn in it.islice(zip(pos, vog_e, vog_n), 0, None, skip):
        avog = chart.ax.arrow(east, north, ve, vn, head_width=hwidth, length_includes_head=True, color=color)

    color = 'green'
    for (east, north), ve, vn in it.islice(zip(pos, tw_e, tw_n), 0, None, skip):
        atw = chart.ax.arrow(east, north, ve, vn, head_width=hwidth, length_includes_head=True, color=color)

    color = 'red'
    for (east, north), ve, vn in it.islice(zip(pos, btv_e, btv_n), 0, None, skip):
        abtv = chart.ax.arrow(east, north, ve, vn, head_width=hwidth, length_includes_head=True, color=color)

    color = 'orange'
    for (east, north), ve, vn in it.islice(zip(pos, cur_e, cur_n), 0, None, skip):
        acurrent = chart.ax.arrow(east, north, ve, vn, head_width=hwidth, length_includes_head=True, color=color)

    chart.ax.legend([avog, atw, abtv, acurrent],
                    'VOG TWD BTV CURRENT'.split(),
                    loc='best')
        
    return chart
