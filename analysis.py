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
from global_variables import G
from utils import DictClass

pd.set_option('display.max_rows', 100)
pd.reset_option('precision')
pd.set_option('display.float_format', lambda x: '%.3f' % x)
np.set_printoptions(precision=4, suppress=True, linewidth = 180)

def find_cog_delay(df, graph=False):
    # Let's be sure the boat is moving for this analyis
    is_moving = np.asarray(df.sog > 0.8)
    is_mostly_moving = np.convolve(is_moving, np.ones(11), mode='same') > 6.0

    # Find the largest run where the boat is moving
    runs = p.find_runs(is_mostly_moving)
    sorted_runs = sorted(runs, key=lambda a: a[1] - a[0])
    first, last = sorted_runs[-1]
    # print(f"Boat is moving consistently from {first} to {last}.")
    if (last - first)/G.SAMPLES_PER_SECOND < (60 * 20):
        print(f"Warning that is not very long!")

    sdf = df.iloc[(first + 1):(last - 1)].copy()

    # Unwrap,  because these are angles
    sig1 = p.unwrap_d(sdf.rcog)
    true_hdg = np.array(sdf.rhdg) + sdf.variation.mean()
    sig2 = p.match_wrap_d(sig1, true_hdg)

    if graph is True:
        c.quick_plot(sdf.index, (sdf.cog, sig1, true_hdg, sig2),
                     "cog, cog_unwrap, hdg, hdg_unwrap".split())

    return find_signal_delay(sig1, sig2, lowcut=0.05, highcut=2, graph=graph)


def find_twa_delay(df, graph=False):
    # Let's be sure the boat is moving for this analyis
    is_moving = np.asarray(df.sog > 0.8)
    is_mostly_moving = np.convolve(is_moving, np.ones(11), mode='same') > 6.0

    # Find the largest run where the boat is moving
    runs = p.find_runs(is_mostly_moving)
    sorted_runs = sorted(runs, key=lambda a: a[1] - a[0])
    first, last = sorted_runs[-1]
    if (last - first)/G.SAMPLES_PER_SECOND < (60 * 20):
        print(f"Boat is moving consistently from {first} to {last}.")
        print(f"Warning that is not very long!")

    sdf = df.iloc[(first + 1):(last - 1)].copy()

    # Unwrap,  because these are angles
    sig1 = p.unwrap_d(sdf.rtwa)
    sig2 = p.match_wrap_d(sig1, np.array(sdf.rawa))

    if graph is True:
        c.quick_plot(sdf.index, (sdf.rawa, sig1, sdf.rtwa, sig2),
                     "awa, awa_unwrap, twa, twa_unwrap".split())

    return find_signal_delay(sig1, sig2, lowcut=0.05, highcut=1, graph=graph)


def find_signal_delay(sig1, sig2, lowcut=0.05, highcut=1, graph=False):
    # High pass filter (by subtracting lowpass).  This gets rid of nuisance DC offsets and
    # emphasizes regions of rapid change...  which are the only regions where delay is
    # observable.
    coeff = p.butterworth_bandpass(lowcut, highcut, order=5, )
    bp_sig1 = p.smooth(coeff, sig1, causal=False)
    bp_sig2 = p.smooth(coeff, sig2, causal=False)


    # Normalize (for correlation)
    nsig1 = (bp_sig1 - bp_sig1.mean()) / (np.std(bp_sig1) * np.sqrt(len(bp_sig1)))
    nsig2 = (bp_sig2 - bp_sig2.mean()) / (np.std(bp_sig2) * np.sqrt(len(bp_sig2)))

    if graph:
        scale = sig1.std()/(5*nsig1.std())
        mu = sig1.mean()
        c.quick_plot(None, (sig1-mu, scale * nsig1, sig2-mu, scale * nsig2))

    # Compute correlation (dot product) for various leads and lags
    plus_minus_window = 50
    res = np.correlate(nsig1, nsig2[plus_minus_window:-plus_minus_window])
    return np.argmax(res) - plus_minus_window, res
