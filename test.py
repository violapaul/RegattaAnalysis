
import matplotlib
matplotlib.get_backend()

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qgrid

# These are libraries written for RaceAnalysis
import global_variables
G = global_variables.init_seattle()
import race_logs
import process as p
import analysis as a
import chart as c


dfs, races, big_df = race_logs.read_dates(["2020-03-07"])
df = dfs[0]

plt.figure()
df.twd.plot()

import importlib
importlib.reload(c)

chart = c.plot_chart(df)
c.draw_track(df, chart, color='green')


ch = c.plot_track(df)


matplotlib.rcsetup.interactive_bk
matplotlib.rcsetup.non_interactive_bk
matplotlib.rcsetup.all_backends
