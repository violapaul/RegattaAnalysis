"""
Make a chart for a race application to the Coast Guard.
"""

# Basics
import os
import datetime
import itertools as it

import arrow
import time
import numpy as np
import matplotlib.pyplot as plt

import numpy.ma as ma
from numba import jit

import netCDF4

# These are libraries written for RegattaAnalysis
from global_variables import G  # global variables
import race_logs                # load data from races
import chart
import utils
import process as p
from utils import DictClass

import metadata

import nbutils
from nbutils import display_markdown, display

from latlonalt import LatLonAlt as lla

#### Cell #3 Type: module ######################################################

G.init_seattle()

def region_from_latlon(pos, lat_border=0.2, lon_border=0.3):
    lats = [p.lat for p in pos]
    lons = [p.lon for p in pos]    
    lat_max, lat_min = chart.max_min_with_border(np.array(lats), lat_border)
    lon_max, lon_min = chart.max_min_with_border(np.array(lons), lon_border)
    return DictClass(lat_max=lat_max, lat_min=lat_min,
                     lon_max=lon_max, lon_min=lon_min)


def plot_positions(ch, positions):
    "Plot the STYC marks identified by name."
    lats = np.array([p.lat for p in positions])
    lons = np.array([p.lon for p in positions])
    marks = np.vstack(G.MAP(lons, lats)).T 
    # Add red x's to the chart ABOVE, at the location of the marks
    ch.ax.scatter(marks[:, 0], marks[:, 1], color='green', marker='x')


SAN_JUANS = dict(
    SW = lla(48.192903, -123.199949),
    N = lla(48.769520, -122.930888),
    SE = lla(48.136633, -122.639886)
)

RTS = {"vashon" :lla(47.32588833718183, -122.51286408543277),
       "gedney" : lla(48.00983499000558, -122.31568520939943),
       "point hudson" : lla.from_degrees_minutes((48, 07.456), (-122, 44.663))}

RACE = RTS

positions = RACE.values()
region = region_from_latlon(positions, 0.05, 0.15)

chartlet = chart.create_chart(region, 10000)
chartlet.fig, chartlet.ax = chart.create_figure(1)
chartlet = chart.draw_chart(chartlet, chartlet.ax, desaturate=False)

plot_positions(chartlet, [RTS["point hudson"]])
chartlet.fig.set_tight_layout(True)

chartlet.fig.savefig("/Users/viola/tmp/foo.jpg", orientation='portrait', dpi=600)


