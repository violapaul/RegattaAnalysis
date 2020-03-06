import os

import copy
import math
import itertools as it
import importlib

import pandas as pd
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt

from numba import jit

from pyproj import Proj
import cv2

import boat_shape
from boat_shape import OUTLINE as BOAT_OUTLINE

import process as p
import chart as c
from utils import DictClass

pd.set_option('display.max_rows', 100)
pd.reset_option('precision')
pd.set_option('display.float_format', lambda x: '%.3f' % x)

np.set_printoptions(precision=4, suppress=True, linewidth = 180)

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
    DictClass(log='2019-11-16_10:09.pd', doc='Snowbird #1.', begin=41076, end=111668, twd=True),
    DictClass(log='2019-11-23_10:23.pd', doc='Practice.', twd=True),
    DictClass(log='2019-11-24_10:33.pd', doc='Practice.', twd=True),
    DictClass(log='2019-11-29_12:54.pd', doc='Longer.  At dock for data collection.'),
    DictClass(log='2019-12-06_11:25.pd', doc='Short, at dock.'),
    DictClass(log='2019-12-07_09:47.pd', doc='Snowbird #2.', begin=54316, end=109378, twd=True),
    DictClass(log='2020-01-18_10:50.pd', doc='STYC Iceberg.', begin=13325, end=108305)
]

def tmp1():
    DATA_DIRECTORY = '/Users/viola/canlogs'

    for ex in SAIL_LOGS:
        df = p.read_sail_log(ex.log, skip_dock_only=True, trim=True, path=DATA_DIRECTORY, cutoff=0.3)
        if df is not None:
            print(ex)
            if 'twd' in df.columns:
                print(df.columns)


def test():
    importlib.reload(p)
    DATA_DIRECTORY = '/Users/viola/canlogs'

    example = DictClass(log='2020-01-18_10:50.pd', doc='STYC Iceberg.', begin=13325, end=108305)
    sail_logs = [example.log]

    dfs, bdf = p.read_sail_logs(sail_logs, skip_dock_only=False, trim=True, path=DATA_DIRECTORY, cutoff=0.3)
    df = dfs[0]
    chart = c.plot_track(df, 1)

    begin, end = (69000, 71000)    
    begin, end = (37037, 43949)
    
    sdf = df.loc[begin : end]
    schart = c.plot_track(sdf, 3)
    c.draw_track(sdf, chart)
    

