#!/usr/bin/env python
# coding: utf-8

# # Tune Up Race vs Creative, 2020-04-16
# 
# [RaceQs video](https://youtu.be/9a5bLeZw8EM)
# 
# [RaceQs Link](https://raceqs.com/tv-beta/tv.htm#userId=1146016&divisionId=64190&updatedAt=2020-04-17T18:05:59Z&dt=2020-04-16T15:43:47-07:00..2020-04-16T17:39:12-07:00&boat=Creative)
# 
# ### Crew 
# 
# Peer Gynt: Sara and Paul.  Creative: Al and Shauna
# 
# ## Summary
# 
# 
# 
# ## Settings
# 
# 
# 
# ## Observations
# 
# 
# ## Conditions
# 
# 
# 
# ![im](Data/Images/2020-04-16_tides.png)
# ![im](Data/Images/2020-04-16_west_point_wind.png)
# ![im](Data/Images/2020-04-16_gg_wind.png)
# 
# 
# 
# 

# In[3]:


# Load some libraries
get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})

import numpy as np

# These are libraries written for RaceAnalysis
import global_variables
G = global_variables.init_seattle()
import race_logs
import process as p
import analysis as a
import chart as c


# In[10]:


# dfs, races, big_df = race_logs.read_dates(["2020-04-16"])
dfs, races, big_df = race_logs.read_dates(["2020-04-19"])
df = dfs[0]
print(races[0])


# In[23]:


ch = c.plot_track(df)


# In[25]:


print(ch.begin)
print(ss)


# In[26]:


# print(ch.begin, ch.end)
ss = slice(22973, 66245)
sdf = df.loc[ss]


# In[27]:


chart = c.plot_chart(sdf)
c.draw_track(sdf, chart, color='red')  


# In[28]:


c.quick_plot(None, (sdf.twd, sdf.stwd, sdf.boat_twd, sdf.hdg, sdf.awa), "twd stwd btwd hdg awa".split())


# In[ ]:





# In[7]:


c.quick_plot(None, (df.twd, df.stwd, df.boat_twd, df.awa), "twd stwd btwd awa".split())


# In[22]:


c.quick_plot(None, (20*df.tws-50, df.awa), "tws awa".split())

