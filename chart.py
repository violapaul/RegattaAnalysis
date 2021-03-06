"""
# Charting and Graphing

Collection of tools to create sailing charts, display race tracks, and plot instrument data versus time.

The first part of this notebook describes the the generation of GEO-registered charts, upon which lat/lon
positions can be scale accurately ploted.

The second describes the plotting of boat instrument data. 

We'll conclude with some details on how these two can be combined.

## Literate Notebook

Warning this is a [Literate Notebook](Literate_Notebook_Module.ipynb), i.e. the notebook contains the code for the charting module.  Do not edit the code in the module directly, edit the notebook and then regenerate the module code.

    convert_notebook.py Chart_Module.ipynb --module
"""

#### Cell #7 Type: module ######################################################

# Load some libraries

# Basics
import os
import math
import time  # used to compute elapsed times
# Helpful when plotting dates.
from datetime import datetime
from dateutil import tz

# Matplotlib is the engine for plotting graphs and images
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import matplotlib.style as mplstyle
# Try to make matplotlib faster.  See https://matplotlib.org/tutorials/introductory/usage.html#performance
mplstyle.use('fast')

# Numpy and Pandas are used to process our racing data.
import numpy as np
import pandas as pd

# OpenCV is a powerful tool for image manipulation.
import cv2  # why is it called cv2?  ... just is.

# These are libraries written for RegattaAnalysis
from global_variables import G  # global variables
from utils import DictClass, is_iterable
import utils

#### Cell #9 Type: module ######################################################

# Helper functions

def new_axis(fignum=None, equal=False, clf=True):
    "Convenience to create an axis with optional CLF and EQUAL."
    fig = plt.figure(fignum)
    if clf:
        fig.clf()
    ax = fig.add_subplot(111)
    if equal:
        ax.axis('equal')
    return fig, ax

#### Cell #13 Type: module #####################################################

# This is the heart of the chart extraction process.

def gdal_extract_chart(chart, source_path, chart_path, zoom_level=None):
    """
    Using GDAL, extract a chart from the source image.

    chart                  : a dict with lat_max, lat_min, lon_max, lon_min, for the extent of the map
    source_path, chart_path : path of the input and output files
    zoom_level              : optional, specify to limit resolution and speed up (13 will do that)
    pixels                  : max width/height of the image, other dimension is computed from extent
    srs                     : spatial reference of the resulting chart
    """
    # Find the extents of the map

    sw = map_project(DictClass(lat=chart.lat_min, lon=chart.lon_min))  # Southwest corner
    ne = map_project(DictClass(lat=chart.lat_max, lon=chart.lon_max))  # Northeast corner

    south, west = sw.north, sw.east
    north, east = ne.north, ne.east

    # Setup gdalwarp args
    # Define the extent of the chart in lon/lat
    te = f"-te {chart.lon_min:.7f} {chart.lat_min:.7f} {chart.lon_max:.7f} {chart.lat_max:.7f}"
    # Reference system of the extent
    te_srs = " -te_srs EPSG:4326 "  # WGS 84

    # Define the SRS of the result
    t_srs = f"-t_srs '{chart.proj}'"
    # Size the image to the longest axis,
    if (east - west) > (north - south):  # wide?
        ts = f"-ts {chart.pixels} 0"  # limit width
    else:
        ts = f"-ts 0 {chart.pixels}"   # limit height

    # Note the default is to pick the zoom level that is matched to the output resolution.
    # Smaller numbers are lower res, and faster.  13 will speed things up a bit.
    if zoom_level is None:
        zoom = ""
    else:
        zoom = f"-oo ZOOM_LEVEL={zoom_level}"  # A hint to determine the level of the pyramid to use.

    command = f"gdalwarp {zoom} {te} {te_srs} {t_srs} {ts} -r bilinear {source_path} {chart_path}"

    run_system_command(f"rm {chart_path}")  # remove output, since gdalwarp will not overwrite
    run_system_command(command)
    # Add the extent of the map in the projection
    return chart.union(dict(south=south, north=north, east=east, west=west,
                            path=chart_path, source=source_path))


def run_system_command(command, dry_run=False):
    "Run a shell command, time it, and log."
    G.logger.debug(f"Running command: {command}")
    if not dry_run:
        start = time.perf_counter()
        os.system(command)
        end = time.perf_counter()
        G.logger.debug(f"Command finished in {end-start:.3f} seconds.")

def map_project(p):
    """
    Project lat/lon pair into our prefered map projection.
    """
    east, north = G.MAP(p.lon, p.lat)  # gdal order is lon then lat, I prefer lat/lon
    return DictClass(north=north, east=east)

def region_from_marks(latlon_list, lat_border=0.2, lon_border=0.3):
    "Compute a map region from a set of latlon marks."
    lats = [p.lat for p in latlon_list]
    lons = [p.lon for p in latlon_list]    
    lat_max, lat_min = max_min_with_border(np.array(lats), lat_border)
    lon_max, lon_min = max_min_with_border(np.array(lons), lon_border)
    return DictClass(lat_max=lat_max, lat_min=lat_min,
                     lon_max=lon_max, lon_min=lon_min)

def extract_region(df, border=0.2, fudge = (0.015, -0.015)):
    """
    Extract the geographic region which covers the entire race track.  BORDER is an
    additional margin which ensures you do not bump up against the edge when graphing.
    """
    # Add just a bit of "fudge factor" to ensure that the extent is not too small, which
    # triggers some corner cases.

    # TODO: since the border is applied in lat/lon separately, its is not uniform.  Same
    # for FUDGE.
    lat_max, lat_min = max_min_with_border(df.latitude, border) + fudge
    lon_max, lon_min = max_min_with_border(df.longitude, border) + fudge

    return DictClass(lat_max=lat_max, lat_min=lat_min,
                     lon_max=lon_max, lon_min=lon_min)


def max_min_with_border(values, border=0.1):
    "Return the range of a series, with a buffer added which is border times the range."
    max = values.max()
    min = values.min()
    delta = (max - min)
    max = max + border * delta
    min = min - border * delta
    return np.array((max, min))


#### Cell #18 Type: module #####################################################

def desaturate_image(im, factor=2):
    "Desaturate the colors in an image, in preparation for plotting on that image."
    hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)  # HSV is hue, saturation, value (intensity)
    hsv[:, :, 1] = hsv[:, :, 1] // factor      # divide the value by factor
    val_offset = ((factor - 1) * 255) // factor
    hsv[:, :, 2] = val_offset + (hsv[:, :, 2] // factor)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

#### Cell #20 Type: module #####################################################

# Combining all features.

def plot_chart(df, fig=None, border=0.2, pixels=2000, color='green', fudge=(0.015, -0.015),  **plot_args):
    """
    Plot a track for race.  The background chart comes from NOAA tiled charts.
    """
    region = extract_region(df, border, fudge=fudge) 
    chart = create_chart(region, pixels=pixels)
    chart.fig, chart.ax = create_figure(fig)
    chart = draw_chart(chart, chart.ax)
    chart = draw_track(df, chart, color=color, **plot_args)
    return chart

def create_chart(region, pixels=2000):
    """
    Using the extent of the GPS race track, create a geolocated map image AND a
    reprojection of the track into the local north/east coordinates.

    Data is loaded into a 'chart' dict, which will collect info on this transformation and
    later plots.
    """
    # Extract the region of the race.
    chart = region.union(dict(proj=G.PROJ4, pixels=pixels))
    
    chart = gdal_extract_chart(chart, G.MBTILES_PATH, "/tmp/mbtile.tif")
    image = cv2.imread(chart.path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return chart.union(dict(image=image))

def create_figure(fig=None, figsize=(6, 6)):
    G.logger.debug(f"Creating figure {fig}")
    if isinstance(fig, matplotlib.figure.Figure):
        fig = fig
    else:
        fig = plt.figure(num=fig, figsize=figsize)
    fig.clf()
    ax = fig.add_subplot(111)   
    return fig, ax

def draw_chart(chart, ax=None, desaturate=True):
    if ax is None:
        ax = chart.ax
    if desaturate:
        image = desaturate_image(chart.image)
    else:
        image = chart.image
    ax.imshow(image, 
              extent=[chart.west, chart.east, chart.south, chart.north])
    ax.grid(True)
    return chart

def draw_track(df, chart, ax=None, color='green', **plot_args):
    """
    Convert the track from lat/lon to image coordinates and then draw.

    Optionally draw on a specific axis.
    """
    lon, lat = np.asarray(df.longitude), np.asarray(df.latitude)
    track = np.vstack(G.MAP(lon, lat)).T
    chart.track = track
    if ax is None:
        ax = chart.ax
    chart.line = ax.plot(chart.track[:, 0], chart.track[:, 1], color=color, **plot_args)[0]
    return chart

#### Cell #27 Type: module #####################################################


def quick_plot(index, data, legend=None, fignum=None, clf=True, title=None, s=slice(None, None, None), ylim=None):
    """
    Super quick tool to display multiple plots on a single axis.

    All data is assumed to share a single index (X axis).  All data is the same length.

    INDEX, which can be None, is the common index for all plots.
    DATA is a sequence of multiple Y axis data (e.g. list or numpy array)
         NOTE: data can be string.  If so it is assumed that it is a comma separated list of
         expressions (see example below).
    LEGEND is a list of legend names one for each data
    FIGNUM is the existing figure to use.
    CLF to clear before plotting, or just plot on what is already there
    YLIM to set the Y limits
    S an optional slice to limit the data to display (or reduce the size)

    For example: 

    quick_plot(df.index, (df.one, df.two, df.three), ['one', 'two', 'three'])
    quick_plot(df.index, (df.one, df.two, df.three), "df.one df.two df.three")
    quick_plot(df.index, "df.one, df.two, df.three")  # Spiffy all in one!
    """
    # Setup the figure and axis
    if isinstance(fignum, matplotlib.figure.Figure):
        fig = fignum
    else:
        fig = plt.figure(num=fignum)
    if clf:
        fig.clf()
    # Create a single axis that fills the figure
    ax = fig.add_subplot(111)
    # Do the plotting
    plot = quick_plot_ax(ax, index, data, legend=legend, s=s)
    # Decorate or adjust
    if ylim is not None:
        ax.set_ylim(*ylim)
    if title is not None:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.tight_layout()
    return plot

def find_nearest(array, value):
    "Find nearest value in array."
    idx = np.searchsorted(array, value, side="left")
    return idx, array[idx]

def to_local_timezone(dt):
    "Convert a datetime to a local datetime."
    HERE = tz.tzlocal()
    UTC = tz.gettz('UTC')
    gmt = dt.replace(tzinfo=UTC)
    return gmt.astimezone(HERE)

# I found that plotting data against a datetime was very slow.  One solution is to explicitly
# compute the datetime ticks.

def get_ticks(datetimes):
    "Explicitly compute the ticks for an array of datetimes."
    formatter = matplotlib.dates.DateFormatter('%H:%M', tz=G.TIMEZONE)
    # Pick a delta time that yields a reasonable number of ticks.
    for dt in [120, 90, 60, 30, 20, 10, 8, 6, 4, 2, 1]:
        loc = matplotlib.dates.MinuteLocator(byminute=None, interval=dt, tz=G.TIMEZONE)
        tick_values = loc.tick_values(datetimes.iloc[0], datetimes.iloc[-1])
        G.logger.debug(f"dt = {dt}, {str(tick_values)}")
        if len(tick_values) > 8:
            break
    tick_datetimes = [to_local_timezone(matplotlib.dates.num2date(v)) for v in tick_values]
    tick_positions = [find_nearest(datetimes, v)[0] for v in tick_datetimes]
    tick_labels = [formatter.format_data(v) for v in tick_values]
    return tick_positions, tick_labels

def quick_plot_ax(ax, index, data, legend=None, s=slice(None, None, None)):
    "Helper function.  See quick_plot for documentation of arguments."
    if isinstance(data, str):
        expressions = data.split(',')
        data = []
        for d in expressions:
            G.logger.debug(f"Evaluating expression {d}")
            data.append(eval(d))
        if legend is None:
            legend = expressions
    np_data = [np.asarray(d) for d in data]
    x = range(len(np_data[0][s]))
    
    def draw():
        if index is None:
            x = range(len(np_data[0][s]))
        else:
            x = index[s]
        if is_datetime(index):
            # If a datetime this can be slow.  Use a range instead, which is fast, and explicitly
            # compute the ticks.
            x = range(len(np_data[0][s]))
            tick_positions, tick_labels = get_ticks(index[s])
            ax.set_xticks(tick_positions) 
            ax.set_xticklabels(tick_labels)
        for d in np_data:
            ax.plot(x, d[s])
        if isinstance(legend, str):
            # When the loc is 'best' then it is very slow!
            ax.legend(legend.split(','), loc='upper right')
        elif is_iterable(legend):
            ax.legend(legend, loc='upper right')
        ax.grid(True)

    def update_func(begin, end):
        nonlocal s
        s = slice(begin, end)
        ax.clear()
        draw()

    def trim_func(*args):
        pass

    draw()

    return DictClass(trim_func=trim_func, update_func=update_func)


def is_datetime(series_like):
    "Does this series contain data which looks like a?"
    # The column types are weirdly obscure. Check the first value.
    np_datetime = isinstance(nth_value(series_like, 0), np.datetime64)
    pd_datetime = isinstance(nth_value(series_like, 0), pd.Timestamp)
    return np_datetime or pd_datetime

def nth_value(series_like, n):
    if isinstance(series_like, pd.Series):
        return series_like.iloc[n]
    elif isinstance(series_like, np.ndarray):
        return series_like[n]

#### Cell #31 Type: module #####################################################

# To link plot to the chart, we add two functions to the chart.  
#
# 1. to trim the track shown to match the plot.
# 2. to show a point at a particular time

def chart_update_functions(chart, skip=None):
    """
    Create two update functions.
    
    trim_func(begin, end) redraws the track trimming off the points before begin and after
    end.

    point_func(time) draws mark at the particular time along the track
    """

    # There can be a huge number of sampled points (at 10Hz).  This limits the total
    # number of samples.
    if skip is None:
        skip = math.ceil(len(chart.track) / 8000)  # No more than 2000 points
    track = chart.track[::skip]

    def trim_func(begin, end):
        G.logger.info(f"trim_func {track.shape} {begin} {end}")
        begin = max(0, begin)
        # end = min(track.shape[0])
        G.logger.info(f"trim_func {begin} {end}")        
        chart.begin, chart.end = begin, end
        b, e = int(begin/skip), int(end/skip)
        chart.line.set_data(track[b:e, 0],
                            track[b:e, 1])
        chart.fig.canvas.draw_idle()

    chart.point = None

    def point_func(time):
        point = chart.track[time]
        G.logger.info(f"Calling point_func update with {time}, {point}")
        if chart.point is None:
            chart.point = chart.ax.plot([point[0]], [point[1]], linestyle = 'None', marker='+', color='red')[0]
        else:
            chart.point.set_data([point[0]], [point[1]])
        chart.fig.canvas.draw_idle()

    chart.trim_func = trim_func
    chart.point_func = point_func

    return chart


#### Cell #32 Type: module #####################################################

# And now we pull it all together.

def chart_and_plot(df, index, data, data2=None):
    """
    Create a figure with one or two plots of data, along with a synced chart/track.

    - If you zoom/pan the top plot, then the track will display the region of interest.
    - If you click on either plot, then the point on the chart will be highlighted.
    """
    if data2 is not None:
        # Create three axes.
        fig = plt.figure(figsize=(8, 10))  # make this figure large
        ax1 = plt.subplot(311)
        ax2 = plt.subplot(312, sharex=ax1)
        ax_chart = plt.subplot(313)
    else:
        # Create two axes.
        fig = plt.figure(figsize=(8, 8))  # make this figure large
        ax1 = plt.subplot(211)
        ax_chart = plt.subplot(212)
        
    quick_plot_ax(ax1, index, data)
    if data2 is not None:
        quick_plot_ax(ax2, index, data2)

    # Chart in the third axis.
    region = extract_region(df) 
    chart = create_chart(region)    
    chart.fig = fig
    chart.ax = ax_chart        # Assign the axis
    chart = draw_chart(chart)
    chart = draw_track(df, chart, color='chartreuse')  # Draw track in light color
    chart = draw_track(df, chart, color='green')       # Draw again in darker
    # Note, chart.line and other attributes are overridden, purposefully.
    fig.tight_layout()

    # Create a set of chart update functions, 
    chart = chart_update_functions(chart)

    # Arrange it so that the plots match the chart, by redrawing the chart to highlight
    # the currently zoomed region in the plots, stored in xlim

    # Declare and register callbacks
    def on_xlim_change(event_ax):
        G.logger.info(f"xlim changed")
        lo, hi = [int(v) for v in ax1.get_xlim()]
        G.logger.info(f"updated xlim: {(lo, hi)}")
        chart.trim_func(lo, hi)

    ax1.callbacks.connect('xlim_changed', on_xlim_change)

    def on_click(event):
        G.logger.info(f"Click event: {event.xdata}")
        if event.xdata is not None:
            chart.point_func(int(event.xdata))

    fig.canvas.mpl_connect('button_press_event', on_click)
    return chart


#### Cell #34 Type: module #####################################################

# One of the critical tasks is to trim the data to remove time at the dock, or to slip into races.

def trim_track(df, fig_or_num=None, border=0.2, skip=None, delay=0.01):
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
    # Chart in the third axis.
    region = extract_region(df) 
    chart = create_chart(region)
    chart.fig, chart.ax = create_figure(fig_or_num)

    chart = draw_chart(chart)  
    chart = draw_track(df, chart, color='chartreuse')  # Draw track in light color
    chart = draw_track(df, chart, color='green')       # Draw again in darker   

    # Chart overlapps the widgets... sometimes.
    # chart.fig.tight_layout()

    # Create a set of chart update functions, 
    chart = chart_update_functions(chart)

    ax_beg = chart.fig.add_axes([0.05, 0.1, 0.03, 0.8], facecolor='lightgoldenrodyellow')
    ax_end = chart.fig.add_axes([0.11, 0.1, 0.03, 0.8], facecolor='lightgoldenrodyellow')
    count = chart.track.shape[0]
    chart.begin, chart.end = 0, count-1
    s_beg = widgets.Slider(ax_beg, 'Begin', 0, count, valinit=0, orientation='vertical', valfmt="%i")
    s_end = widgets.Slider(ax_end, 'End',   0, count-1, valinit=count-1, orientation='vertical', valfmt="%i")

    # Unfortunate "bug" that graph can be unresponsive if you do not keep a handle on the sliders.
    # https://github.com/matplotlib/matplotlib/issues/3105/
    chart.sliders = [s_beg, s_end]
    chart.is_trimmed = False
    chart = chart_update_functions(chart)

    def trim(val):
        G.logger.info(f"Calling plot_track update.")
        chart.is_trimmed = True
        chart.trim_func(int(s_beg.val), int(s_end.val))
        chart.fig.canvas.draw_idle()

    s_beg.on_changed(trim)
    s_end.on_changed(trim)

    return chart


#### Cell #41 Type: metadata ###################################################

#: {
#:   "metadata": {
#:     "timestamp": "2021-02-21T21:13:45.033077-08:00"
#:   }
#: }

#### Cell #42 Type: finish #####################################################

