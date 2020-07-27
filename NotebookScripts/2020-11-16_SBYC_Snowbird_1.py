#!/usr/bin/env python
# coding: utf-8

# # SBYC Snowbird 1, November 16th, 2020
# 
# 
# 

# West Point Wind
# 
# 
# 
# Golden Gardens Wind
# 
# 

# In[2]:


# Load some libraries
get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})

import numpy as np

# These are libraries written for RaceAnalysis
from global_variables import G
import race_logs
import process as p
import analysis as a
import chart as c

G.init_seattle()


# In[6]:


import importlib
importlib.reload(c)


# In[7]:


dfs, races, big_df = race_logs.read_dates(["2019-11-16"])
# dfs, races, big_df = race_logs.read_dates(["2019-11-16"])
df = dfs[0]
races


# In[8]:


chart = c.plot_chart(df)
c.draw_track(df, chart, color='green')


# In[11]:


chart.image.shape


# In[8]:


df.latitude.mean(), df.longitude.mean()


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




