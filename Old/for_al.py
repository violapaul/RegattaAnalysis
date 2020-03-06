import os

import copy
import math
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
from utils import DictClass

SAIL_LOGS = [
    dict(filename='2019-10-04_16:43.pd', doc='Sail to Everett for Foulweather Bluff Race.'),
    dict(filename='2019-10-05_09:18.pd', doc='Foulweather Bluff Race'),
    dict(filename='2019-10-11_17:38.pd', doc='Short practice, upwind tacks and downwind jibes.'),
    dict(filename='2019-10-12_09:45.pd', doc='CYC PSSC Day 1'),
    dict(filename='2019-10-18_13:51.pd', doc='Short, at dock.'),
    dict(filename='2019-10-19_09:45.pd', doc='STYC Fall Regatta.'),
    dict(filename='2019-10-26_09:40.pd', doc='Grand Prix Saturday.'),
    dict(filename='2019-10-26_12:35.pd', doc='Short, at dock.'),
    dict(filename='2019-11-07_12:46.pd', doc='Short, at dock.'),
    dict(filename='2019-11-16_10:09.pd', doc='Snowbird #1.'),
    dict(filename='2019-11-23_10:23.pd', doc='Practice.'),
    dict(filename='2019-11-24_10:33.pd', doc='Practice.'),
    dict(filename='2019-11-29_12:54.pd', doc='Longer.  At dock for data collection.')
    ]

SAMPLES_PER_SECOND = 10

def test():
    importlib.reload(p)
    importlib.reload(a)
    sail_logs = p.find_sail_logs(a.DATA_DIRECTORY)
    print(sail_logs)
    sail_logs = ['2019-11-16_10:09.pd']

    dfs, bdf = p.read_sail_logs(sail_logs, skip_dock_only=False, trim=True, path=a.DATA_DIRECTORY, cutoff=0.3)
    df = dfs[0]
    chart = a.plot_track(df, 1)

    rdf = df.loc[41844:111988]
    chart = a.plot_track(rdf, 1)

    sdf = df.loc[79140:105810]
    chart = a.plot_track(sdf, 1)

    tacks = find_tacks(sdf)

    num = 5
    ss = slice(tacks[num][0]-200, tacks[num+1][1]+200)
    tdf = df.loc[ss]
    chart = a.plot_track(tdf, 1)

    quick_plot(df.index, (df.hdg+15.4, df.cog), "hdg cog".split(), fignum=2)
    quick_plot(tdf.index, (tdf.hdg+15.4, tdf.cog), "hdg cog".split(), fignum=2)

    ss = slice(tacks[num][0]-200, tacks[num+1][1]+200, 50)
    a.plot_boat(tdf, ss)
