#!/usr/bin/env python
# coding: utf-8

# # Race Review
# 
# This is a general page that will display info about a race day.

# In[1]:


# These are libraries written for RaceAnalysis
from global_variables import G
from nbutils import display_markdown, display
import race_logs
import metadata as m
import process as p
import analysis as a
import chart as ch
import utils

import nbutils

# Initialize for Seattle.
G.init_seattle(logging_level="INFO")


# In[2]:


import importlib
importlib.reload(m)
importlib.reload(race_logs)


# In[6]:


date = "2019-10-19"
date = '2020-04-26'
dfs, races, bigdf = race_logs.read_dates([date])
df = dfs[0]
race = races[0]
chart = ch.trim_track(df)


# In[4]:


m.display_race_summary(race)


# In[ ]:




