#!/usr/bin/env python
# coding: utf-8

# # SBYC Snowbird Feb 8th, 2020
# 
# 
# 

# West Point Wind
# 
# ![Screen%20Shot%202020-02-15%20at%202.37.40%20PM.png](attachment:Screen%20Shot%202020-02-15%20at%202.37.40%20PM.png)
# 
# Golden Gardens Wind
# 
# ![Screen%20Shot%202020-02-15%20at%202.35.48%20PM.png](attachment:Screen%20Shot%202020-02-15%20at%202.35.48%20PM.png)

# In[1]:



# Load some libraries
get_ipython().run_line_magic('matplotlib', 'notebook')

import importlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import process as p
import analysis as a
import chart as c
import tides
import global_variables as G
from utils import DictClass

from numba import jit


# In[2]:


pwd


# In[3]:


# Race data is stored in these files.
example = DictClass(log='2020-02-08_10:23.pd.gz', doc='Snowbird #4.', begin=32000, end=130010, twd=True)

print(f"Reading and analyzing {example.doc} from {example.log}")
raw_df = p.read_sail_log(example.log, discard_columns=True, skip_dock_only=False, trim=True, 
                     path=G.DATA_DIRECTORY, cutoff=0.3)

# Let's grab the times for the race.
raw_df['tide'] = tides.tides_at(raw_df.row_times)

# Trim the raw log, which typically runs from the dock to the dock.
df = raw_df.loc[example.begin : example.end]


# In[4]:


begin_time = df.row_times.iloc[0].strftime('%B %d, %Y, %r')
end_time = df.row_times.iloc[-1].strftime('%r')
print(f"Trimmed log runs from {begin_time} to {end_time}")


# In[5]:


chart = c.plot_chart(df)
c.draw_track(df, chart, color='green')


# In[37]:


# This a bit fancier and can be used to interactively trim the race.  
# Note, it freezes up sometimes.  If it does, you can re-evaluate the expression.
ch = c.plot_track(raw_df, skip=1000)


# In[27]:


# ch will hold the values of the sliders above.  After you are done trimming away the all but 
# the race, then you can take the ends and acess them like this
begin, end = ch.begin, ch.end
print(f"We have trimmed the race from {begin} to {end}")

# These numbers can be used to trim the raw log (as we did above), or to focus in on a key 
# section of the race.

wdf = df.loc[begin : end]


# In[51]:


# Or just save a previous begin/end time.
begin, end = 38000, 53000
begin, end = 39510, 53010
wdf = df.loc[begin : end]

chart = c.plot_chart(wdf)
c.draw_track(wdf, chart, color='green')


# In[50]:


plt.figure()
wdf.awa.plot()


# In[36]:


race_slice = slice(example.begin, example.end)
race_slice = slice(begin, end)
wdf = df.loc[race_slice]
print(wdf.row_times.min(), wdf.row_times.max())
ch = c.show_boat_arrows(df, race_slice, dt_seconds=10, skip=1)


# In[32]:


wdf.row_times


# In[ ]:


import tides

plt.figure()
ttt = tides.tides_at(odf.row_times)
plt.plot(odf.row_times, ttt)


# In[ ]:


df.variation


# In[ ]:


wdf.twd


# In[ ]:


c.quick_plot(df.row_times, (df.twd, 20*df.tws))


# In[ ]:


wdf.row_times


# In[ ]:




