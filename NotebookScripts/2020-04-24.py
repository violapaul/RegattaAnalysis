#!/usr/bin/env python
# coding: utf-8

# # Tune Up Race vs Creative, 2020-04-24
# 
# [RaceQs video](??)
# 
# [RaceQs Link](??)
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

# In[1]:


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


# In[2]:


# dfs, races, big_df = race_logs.read_dates(["2020-04-16"])
dfs, races, big_df = race_logs.read_dates(["2020-04-24"])
df = dfs[0]
print(races[0])


# In[3]:


ch = c.plot_chart(df)
c.draw_track(df, ch)


# In[4]:


ch = c.plot_track(df)


# In[5]:


sdf = df


# In[6]:


chart = c.plot_track(sdf)  


# In[7]:


sdf = df.loc[chart.begin:chart.end]
sdf.row_times


# In[ ]:


sdf.row_times


# In[ ]:


c.quick_plot(sdf.row_times, (sdf.twd, sdf.stwd, sdf.boat_twd, sdf.hdg, sdf.awa, 100*sdf.spd), "twd stwd btwd hdg awa".split())


# In[ ]:


import matplotlib.dates as mdates
chart.ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))


# In[ ]:


import pandas as pd
display(type(sdf.row_times.dtype))
display(sdf.row_times.dtype)
display(np.datetime64)
pd.DatetimeTZDtype()


# In[ ]:


isinstance(sdf.timestamp.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)


# In[ ]:


dir(sdf.row_times)


# In[ ]:


import datetime 
sdf.row_times.dtype == np.datetime64


# In[ ]:


np.datetime64


# In[ ]:


sdf.timestamp


# In[ ]:


c.quick_plot(sdf.row_times, (sdf.boat_twd+15, sdf.hdg, sdf.awa, 100*sdf.spd, 100*sdf.aws, 100*sdf.sog), "twd hdg awa spd aws sog".split())


# In[ ]:





# In[ ]:


sdf.row_times


# In[ ]:


c.quick_plot(sdf.timestamp, (sdf.spd, sdf.boat_tws, sdf.aws), "spd tws aws".split())


# In[ ]:


c.quick_plot(sdf.timestamp, (sdf.cog, sdf.hdg+15, sdf.boat_twd+15+50, sdf.twd+15+50), "cog hdg btwd".split())


# In[ ]:


c.quick_plot(sdf.row_times, (sdf.rhdg+15, sdf.hdg+15))


# In[ ]:


c.quick_plot(None, (df.twd, df.stwd, df.boat_twd, df.awa), "twd stwd btwd awa".split())


# In[ ]:


c.quick_plot(None, (20*df.tws-50, df.awa), "tws awa".split())

