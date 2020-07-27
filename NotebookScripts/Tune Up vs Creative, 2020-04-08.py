#!/usr/bin/env python
# coding: utf-8

# # Tune Up vs Creative, 2020-04-08
# 
# [RaceQs video](https://youtu.be/Pt6hNME7Uac)
# 
# [RaceQs Link](https://raceqs.com/tv-beta/tv.htm#userId=1073253&updatedAt=2020-04-13T23:02:27Z&dt=2020-04-12T11:35:37-07:00..2020-04-12T12:28:01-07:00&boat=Peer%20Gynt)
# 
# ### Crew 
# 
# Peer Gynt: Sara and Paul.  Creative: Al and Shauna
# 
# ## Summary
# 
# Quick upwind race from West Point to Meadow Point.  I know I was a bit confused about the start...  but Creative was clearly the better boat, consistently picking up boat lengths on every leg.  Nice job folks.
# 
# ## Settings
# 
# Generally we were going for depowered settings, don't know if we overdid it.  With waves perhaps we needed to go for more power?
# 
# - Shroud settings for 14-18 (base is 10-14).  Sara did the mast sighting before the "race".   
#   - While we were out she reported some fall of at the mast head (rather than sag in the middle).  We did not adjust.
#   - Afterward Sara mentioned a difference from PORT to STBD.  *More details?*
# - Jib cars at base (3 for us... but its a just a spot on the rail).
# - Backstay on (between 3 and 4 inches).  Alot for us (but not max).
# - Outhaul on hard (*red to red?*)
# - Main halyard at 10 (near max). Plus 6-9 inches of cunningham. 
#   - This is close to max for us.  We need more when the ba
# - Jib halyard at 9-10 (near max for us).
# - Main: Most of the time top telltale streaming.
# - Jib: Inside the shrouds.  Between 2-3 tapes red/green (counting from outside in).
# 
# 
# ## Observations
# 
# - Paul drove a bit more than half the time.  Sara's been driving lately and doing better.
#   - Paul likes to try to play the main (while driving),  but that is near impossible in those conditions.
# - Generally a lot of heel, but not too much weather helm.
#   - The depowered settings were meant to improve this, but perhpas we did too much?
#   - Did we need more power to punch through waves?
# - We generally did not sail higher than Creative.  
# 
# 
# ## Conditions
# 
# Sunny warm (50+).  Wind 10-15 (mostly 13ish).  Sea state 2-4 ft (big ebb tide combined with winds form the North).
# 
# 
# ![im](Data/Images/2020-14-12_tides.png)
# ![im](Data/Images/2020-14-12_West_Point_Wind.png)
# ![im](Data/Images/2020-14-12_Golden_Gardens_Wind.png)
# 
# ?? [Facebook RaceQs video]()
# 
# [RaceQs Link](https://raceqs.com/tv-beta/tv.htm#userId=1073253&updatedAt=2020-04-09T01:04:48Z&dt=2020-04-08T10:49:19-07:00..2020-04-08T15:07:03-07:00&boat=Peer%20Gynt)
# 
# ### Crew 
# 
# Peer Gynt: Sara and Paul.  Creative: Al and Shauna
# 
# ### Conditions
# 
# Sunny and warm.  Wind up to 15kts with seastate 2-4 feet.  
# 
# ## Summary
# 
# 
# ### Learnings
# 
# ### West Point Wind
# 
# ![im](Data/Images/2020-04-08-WestPointWind.png)
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


import importlib
# importlib.reload(race_logs)


# In[3]:


dfs, races, big_df = race_logs.read_dates(["2020-04-08"])
df = dfs[0]


# In[4]:


races[0]


# In[6]:


ch = c.plot_chart(df)
c.draw_track(df, ch)


# In[18]:


c.quick_plot(None, (df.twd, df.stwd, df.boat_twd), "twd stwd btwd".split())


# In[17]:


df.twd.describe()


# In[16]:


c.quick_plot(None, (df.twd, df.stwd, df.awa), "twd stwd awa".split())


# In[22]:


c.quick_plot(None, (20*df.tws-50, df.awa), "tws awa".split())


# In[31]:


# ss = slice(ch.begin, ch.end)
wdf = df.loc[ss]
c.quick_plot(None, (wdf.twd-200, wdf.boat_twd-200, wdf.stwd-200, wdf.awa))


# In[17]:


ch = c.show_boat_arrows(df, ss, dt_seconds=10, skip=3)


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




