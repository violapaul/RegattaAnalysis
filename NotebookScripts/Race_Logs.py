#!/usr/bin/env python
# coding: utf-8

# # Race Logs
# 
# This notebook is used to view, update, and manage the set of race/sail instrument logs collected on Peer Gynt using the Raspberry Pi.
# 
# Information about the race logs are collected into a Pandas dataframe (which is much like a database).  Why not use a database?  Because then we can share "know how" and some tools.
# 
# ## Goals
# 
# - View the list of all logs and inspect for accuracy.
#     - Correct errors
# - Update and add a new log
# - Add meta data (like race start/end)
# - View race data.
# 
# ## TODO
# 
# - Add additional metadata for each race.  Some extracted automatically?
#    - Tenet: this data should be user entered, not automatic.  Automatic goes in a separate table?
#    - Conditions.  Crew.  Settings for rig.
#    - Speed. Quality of maneuvers.
# - How to edit more complex and longer text fields.
# - How to handle multiple races in one log??
#    - Split into different files?
# - Make it faster to show the race track.
# - What if I want to permanently delete a log?  
# - How can I tell if the log is a duplicate?
# 

# In[1]:


import os
import itertools as it
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qgrid


# In[2]:


## %matplotlib inline


# In[3]:


# These are libraries written for RaceAnalysis
from global_variables import G
from nbutils import display_markdown, display
import race_logs
import process as p
import analysis as a
import chart as c
import metadata
import utils

import nbutils

# Initialize for Seattle.
G.init_seattle(logging_level="DEBUG")


# In[4]:


# Info about the race logs are stored in a DataFrame.
md = metadata.read_metadata()
log_info = metadata.summary_table(md.records)

# The data in this table can be editted using a QGrid Control.  Click on the column header to sort.  Click again 
# to sort in a different order.  Double click on a cell to edit.
w = qgrid.show_grid(log_info, show_toolbar=True)
w


# ## Trimming the data 
# 
# The logs start from the time we power up until we shutdown.  And this typically inclues 30-90 mins at the dock (or more).
# 
# The UI below (which is sort of unreliable right now) can be used to find the trim points.
# 
# On the left are two "sliders" (primitive, I know).  The first is used to determine the beginning of the data to show.  The second the end.  When you are done, the results are stored in `ch.begin` and `ch.end`.
# 
# Note, for some reason the UI freezes.  If so,  you can just re-run the command.  

# ## Quick Visualization Interface
# 
# Below we have added a bit of additional functionality to the qgrid interface:  When you select a row, that race track will be shown automatically.
# 
# Note, it takes a second (or two) between selecting a row and the display.  Its one of the only things that are a bit slow.

# In[7]:


# create a function that is called "back" when a row is selected

def show(args, _):
    # Args are a bit obscure
    row_num = args['new'][0]  # The newly selected row numbers, selected the first
    print(f"selected row {row_num}")
    # Need to used the changed df, in case of reordering, etc.
    file = w.get_changed_df().iloc[row_num].file
    print(f"displaying file: {file}")
    df = race_logs.read_log_file(file, discard_columns=True, skip_dock_only=False, trim=True, cutoff=0.3)
    chart = c.trim_track(df, fig_or_num=fig)
    fig.tight_layout()

fig = plt.figure(figsize=(6, 6))
w = qgrid.show_grid(log_info, show_toolbar=True)
display(w)

# Bind the callback
w.on('selection_changed', show)


# In[6]:


G.logger.debug("foo")


# In[ ]:




