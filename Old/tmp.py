

import os
import itertools as it
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
import utils


import importlib
importlib.reload(race_logs)

# Info about the race logs are stored in a DataFrame.
log_info = race_logs.read_log_info()


race_logs.save_updated_log_info(log_info)

missing_files = race_logs.find_new_logs(log_info)
print(missing_files)


# Load these new log files
new_dfs = []
for file in missing_files:
    print(f"Loading {file}")
    ndf = race_logs.read_log(file, discard_columns=True, skip_dock_only=False, trim=True, path=G.LOGS_DIRECTORY, cutoff=0.3)
    new_dfs.append(ndf)

# As a convenience combine the new logs into one large DataFrame
bdf =  pd.concat(new_dfs, sort=True, ignore_index=True)


# Lets display each dataset on a map, to jog the memory

# Create a chart that can contain all the tracks.
chart = c.plot_chart(bdf)

# Plot each in a different color.
for df, color in zip(new_dfs, it.cycle("red green blue brown grey".split())):
    print(f"Displaying in {color}")
    c.draw_track(df, chart, color=color)

log_info


