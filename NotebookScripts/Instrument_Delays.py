#!/usr/bin/env python
# coding: utf-8

# # Instrument Delays
# 
# *It makes sense to read [Boat Instruments](Boat_Instruments.ipynb) if you do not yet know about the set of all instruments on a modern boat and how that data is processed.*
# 
# Various physical phenomena are measured using instruments on the boat and captured in different ways.  As we have already discussed, the raw signals can be quite noisy, and many of the signals are "damped" (i.e. filtered) before they are displayed (AWA, AWS, SPD, etc).  Other signals are computed from basic instruments (TWD, TWA, etc).  Filtering unavoidably introduces delays.
# 
# The most complex sailing instrument by far is the GPS, which provides speed and course over ground (SOG and COG).  GPS uses the ranges measured to 5-10 satellites to triangulate the position on the earth.  The process of tracking the satellites and estimating distance is complex (and it requires filtering).  From that satellite ranges, global positions and velocities are estimated, which in turn requires additional filtering.  
# 
# All this filtering adds up to some delay.  The question is **how much delay**?
# 
# ## Summary
# 
# In this notebook, we will compare COG and the HDG and search for a latency that best aligns the two values (being sure to first realize that HDG is magnetic north and not true north).  Note, on any given day, for any given race, we might run into some issues with this approach.  If the current is strong out of the north (flood here in Seattle) and we are sailing East to West, then the COG will always be south of the HDG.  Our hope is that this will wash out for an entire sailing session.  And it should certainly wash out across many days.  (As we will see, our comparison technique is also somewhat insensitive to the "current effect".)
# 
# ## Glossary
# 
# - AWA: apparent wind angle, the angle of the wind blowing at the top the mast (fast but noisy)
# - AWS: apparent wind speed, the speed at the mast head (fast and noisy)
# - SPD: boat speed **through the water** measured with the paddle wheel speedo in the hull (fast and noisy)
# - HDG: compass heading (on PG this is **magnetic northa and not true north**, though easily corrected using magnetic variation/declination).
# - TWS: true wind speed, the speed of the wind over the ground (computed from the above quantities using the "wind triangle").
# - TWD: true wind direction, the angle of the wind blowing over the ground (see "wind triangle").
# - TWA: true wind angle, the angle of the wind over the ground reported relative the orientation of the boat (same)
# - COG and SOG: course and speed over ground from the GPS (these are relative to true north not magnetic on PG).

# In[1]:


# Load some libraries
get_ipython().run_line_magic('matplotlib', 'notebook')

import matplotlib.pyplot as plt
import numpy as np
from numba import jit

# These are libraries written for RaceAnalysis
import global_variables
G = global_variables.init_seattle()
import race_logs
import process as p
import analysis as a
import chart as c


# In[9]:


import importlib
importlib.reload(race_logs)


# In[13]:


dfs, races, big_df = race_logs.read_dates(["2019-10-05", "2020-02-08"])
df = dfs[1]


# In[14]:


# mark to the finish
chart = c.plot_chart(df)
c.draw_track(df, chart, color='green')


# ## COG is unstable if the boat is not moving
# 
# A GPS unit has no direct mechanism for estimating orientation. COG is *estimated* from postion (lat/lon) over time.  For this to work, the boat must be moving.  We'll set a threshold of 1 m/s (2 knots).

# In[16]:


# Let's be sure the boat is moving for this analyis
is_moving = np.asarray(df.sog > 1.0)
is_mostly_moving = np.convolve(is_moving, np.ones(11), mode='same') > 6.0
c.quick_plot(df.index, (df.sog, is_moving, is_mostly_moving))


# In[20]:


# Find the largest run where the boat is moving
runs = p.find_runs(is_mostly_moving)
sorted_runs = sorted(runs, key=lambda a: a[1] - a[0])
first, last = sorted_runs[-1]
print(f"Boat is moving consistently from {first} to {last}.")
if (last - first)/G.SAMPLES_PER_SECOND < (60 * 20):
    print(f"Warning that is not very long!")
    
sdf = df.iloc[(first + 1):(last - 1)].copy()


# ## Comparing HDG with COG
# 
# Next step is to compare HDG with COG.
# 
# ### Angles are annoying
# 
# One annoying fact of angles is that 355 degrees is actually quite close to 5 degrees, though the numeric difference is large.  One way around this is to **unwrap** the angles.  This is process where the angles are adjust by adding/subtracting 360 so that large steps in the signal are minimized.  

# In[21]:


# Unwrap,  because these are angles
sdf['cog_unwrap'] = p.unwrap_d(sdf.rcog)
sdf['thdg'] = np.array(sdf.rhdg) + sdf.variation.mean()
sdf['thdg_unwrap'] = p.match_wrap_d(sdf.cog_unwrap, sdf.thdg)

c.quick_plot(sdf.index, (sdf.rcog, sdf.cog_unwrap, sdf.thdg, sdf.thdg_unwrap),
            "cog, cog_unwrap, hdg, hdg_unwrap".split())


# ### Unwrapping gets rid of the huge jumps
# 
# The raw signals (before unwrapping) have many large jumps.  The unwrapped versions make a lot more sense.

# In[22]:


#  Let's drill down and show COG vs. HDG for a short window.
start = 92000
wide_slice = slice(start, start + 1500)
wdf = sdf.loc[wide_slice]
c.quick_plot(wdf.index, (wdf.cog_unwrap, wdf.thdg_unwrap),
            "cog_unwrap, hdg_unwrap".split())


# ## COG and HDG are quite close.
# 
# Notice above that COG and HDG are actually pretty close (as they should be).  But there are clearly places where they are different.  There are two good reasons for COG and HDG to differ: 
# 
# 1. Current: the boat may be pushed by current in a different direction from its heading.
# 2. Leeway: when sailing upwind, all boats slip sideways slightly.
# 
# In addition, since HDG is a relative simple instrument (e.g. a compass) while GPS is quite complex, it is plausible that HDG measures a change in orientation before COG.
# 
# To compare these signals we will attempt to first normalize away the differences, and then find the delay that minimizes the remaining differences.
# 
# ### High pass
# 
# Its reasonable to assume that changes in boat orientation show up in both signals, and the most frequent examples are tacks and jibes.  The actual angles measured between tacks and jibes may well be a bit different.  These constant offsets (or slowly varying offsets) can be eliminated by throwing out the low frequencies (slowly changing stuff) and keeping the rapid changes.  This is called a [high pass filter](https://en.wikipedia.org/wiki/High-pass_filter).
# 
# The high pass rejects the absolute value of the signals (what some call DC) and leaves only the major wiggles.  This allows us to ignore that COG and HDG are affected by current (and can differ by 10 or more degrees). 
# 
# Note, this also removes any miscalibration because of compass alignment, since this is just a constant offset in HDG.
# 
# **Notice how much more similar the high pass versions are.**

# In[23]:


# Band-pass filter.  This gets rid of nuisance DC offsets and
# emphasizes regions of rapid change...  which are the only regions where delay is
# observable.
coeff = p.butterworth_bandpass(lowcut=0.05, highcut=1, order=5, )
sdf['hp_cog'] = hp_sig1 = p.smooth(coeff, sdf.cog_unwrap)
sdf['hp_hdg'] = hp_sig2 = p.smooth(coeff, sdf.thdg_unwrap)

wdf = sdf.loc[wide_slice]
mu = wdf.cog_unwrap.mean()-20
c.quick_plot(wdf.index, (wdf.hp_cog, wdf.hp_hdg, wdf.cog_unwrap-mu, wdf.thdg_unwrap-mu),
             "hp_cog, hp_hdg, cog, hdg".split())


# ## Normalizing the signals 
# 
# We can go further to normalize the signals so that they have the safe level of variability (this would allow you to compare signals that are similar but not identical).

# In[76]:


# Normalize (for correlation)
sdf = sdf
ncog = (sdf.hp_cog - sdf.hp_cog.mean()) / (np.std(sdf.hp_cog) * np.sqrt(len(sdf.hp_cog)))
nhdg = (sdf.hp_hdg - sdf.hp_hdg.mean()) / (np.std(sdf.hp_hdg) * np.sqrt(len(sdf.hp_hdg)))

sdf['ncog'] = ncog
sdf['nhdg'] = nhdg

wdf = sdf.loc[wide_slice]

c.quick_plot(wdf.index, (wdf.ncog, wdf.nhdg))


# ## Find the right delay using correlation
# 
# We can now (finally) compare the COG to delayed versions of HDG, to find the best match.  This is done with the [cross correlation](https://en.wikipedia.org/wiki/Cross-correlation).  See also [numpy.correlate](https://docs.scipy.org/doc/numpy/reference/generated/numpy.correlate.html).
# 
# The max correlation corresponds to the delay where the two signals have the highest match.

# In[77]:


# Compute correlation (dot product) for various leads and lags
plus_minus_window = 50
res = np.correlate(ncog, nhdg[plus_minus_window:-plus_minus_window])
delay = np.argmax(res) - plus_minus_window
print(f"Found a delay of {delay}")

plt.figure()
plt.plot(range(-plus_minus_window, plus_minus_window+1), res)


# In[78]:


wcog = np.array(wdf.ncog)
whdg = np.array(wdf.nhdg)

c.quick_plot(None, (wcog[delay:], whdg[:-delay]))


# In[12]:


logs_files = [log.log for log in G.SAIL_LOGS]
dfs, bdf = p.read_sail_logs(logs_files, path=G.DATA_DIRECTORY)


# In[80]:


for df in dfs:
    if df is not None:
        delay, res = a.find_cog_delay(df, )
        print(f"Delay for {df.filename} is {delay}")


# ## Delay estimating TWA
# 
# The TWA angle is computed on the boat using a combination of geometry and filtering.
# 
# The perception on the boat is that TWA is quite delayed relative to AWA (perhaps 5 or more seconds).
# 
# Note, TWA is not always logged (you need to have the autopilot on).  And I beleive we have adjusted the "damping" to various levels.

# In[23]:


tw_df = [df for df in dfs if df is not None and 'twa' in df]
[df.filename for df in tw_df]


# In[24]:


df = tw_df[-1]
c.quick_plot(df.index, (df.rawa, df.rtwa))


# In[79]:


importlib.reload(p)
importlib.reload(a)


# In[66]:


delay, res = a.find_twa_delay(df, graph=True)
for df in tw_df:
    if df is not None:
        delay, res = a.find_twa_delay(df, )
        print(f"Delay for {df.filename} is {delay/10.0}")


# ## Delay for TWA is surprsingly low!
# 
# As we explored in [Boat Instruments](Boat_Instruments.ipynb),  the calculation of TWA/TWD/TWS is actually quite clever, and in some cases it can **lead** changes in AWA. 

# In[1]:


df


# In[2]:


dfs


# In[6]:


df.columns


# In[29]:


tdf = df["row_times latitude longitude awa aws hdg spd sog cog".split()].loc[1005:1020]
tdf.row_times = tdf.row_times.dt.strftime('%r')
tdf


# In[30]:


tdf.to_pickle(os.path.join(G.DATA_DIRECTORY, "basic_example.pd"))


# In[25]:


import os


# In[ ]:




