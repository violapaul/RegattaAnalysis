#!/usr/bin/env python
# coding: utf-8

# # Signal Delays
# 
# It makes sense to read [Boat Instruments](Boat_Instruments.ipynb) if you do not yet know about the set of all instruments on a modern boat and how that data is processed.
# 
# Various physical phenomena are measured using instruments on the boat and captured in different ways.
# 
# As we have already discussed, the raw signals can be quite noisy, and many of the signals are "damped" (i.e. filtered) before they are displayed (AWA, AWS, SPD, etc).  Other signals are computed from basic instruments (TWD, TWA, etc).
# 
# The most complex instrument by far is the GPS, which provides speed and course over ground (SOG and COG).  GPS uses the ranges measured to 5-10 satellites to triangulate the position on the earth.  The process of tracking the satellites and estimating distance is complex (and it requires filtering).  From that positions and velocities are estimated, which in turn requires filtering.  
# 
# All this filtering adds up to some delay.  The question is **how much delay**?
# 
# ## Summary
# 
# We are going to take the COG and the HDG and search for a latency that best aligns the two values (being sure to first realize that HDG is magnetic north and not true north).  Note, on any given day, for any given race, we might run into some issues with this approach.  If the current is strong out of the north (flood here in Seattle) and we are sailing East to West, then the COG will always be south of the HDG.  Our hope is that this will wash out for an entire sailing session.  And it should certainly wash out across many days.
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

import matplotlib.pyplot as plt
import numpy as np

import process as p
import analysis as a
import chart as c
import global_variables as G
from utils import DictClass

from numba import jit


# In[2]:


example = G.get_log("2020-02-08")

print(f"Reading and analyzing {example.log}")
df = p.read_sail_log(example.log, discard_columns=True, skip_dock_only=False, trim=True, path=G.DATA_DIRECTORY, cutoff=0.3)


# In[4]:


# Snowbird #1 was a day when we battled with Creative on the upwind leg from the last 
# mark to the finish
chart = c.plot_chart(df)
df = df.loc[example.begin : example.end]
c.draw_track(df, chart, color='green')


# In[5]:


# Let's be sure the boat is moving for this analyis
is_moving = np.asarray(df.sog > 0.8)
is_mostly_moving = np.convolve(is_moving, np.ones(11), mode='same') > 6.0
c.quick_plot(df.index, (df.sog, is_moving, is_mostly_moving))


# In[6]:


# Find the largest fun where the boat is moving
runs = p.find_runs(is_mostly_moving)
sorted_runs = sorted(runs, key=lambda a: a[1] - a[0])
first, last = sorted_runs[-1]
print(f"Boat is moving consistently from {first} to {last}.")
if (last - first)/G.SAMPLES_PER_SECOND < (60 * 20):
    print(f"Warning that is not very long!")
    
sdf = df.iloc[(first + 1):(last - 1)].copy()


# In[7]:


import importlib
importlib.reload(p)
importlib.reload(a)


# ## How to do signal processing on angles??
# 
# Next step is to compare HDG with COG.  
# 
# One annoying fact of angles is that 355 degrees is actually quite close to 5 degrees, though the numeric difference is large.  One way around this is to **unwrap** the angles.  This is process where the angles are adjust by adding/subtracting 360 so that large steps in the signal are minimized.  

# In[22]:


# Unwrap,  because these are angles
sdf['cog_unwrap'] = p.unwrap_d(sdf.cog)
sdf['thdg'] = np.array(sdf.rhdg) + sdf.variation.mean()
sdf['thdg_unwrap'] = p.match_wrap_d(sdf.cog_unwrap, sdf.thdg)

c.quick_plot(sdf.index, (sdf.cog, sdf.cog_unwrap, sdf.thdg, sdf.thdg_unwrap),
            "cog, cog_unwrap, hdg, hdg_unwrap".split())

start = 98000
wide_slice = slice(start, start + 3500)
wdf = sdf.loc[wide_slice]
c.quick_plot(wdf.index, (wdf.cog_unwrap, wdf.thdg_unwrap),
            "cog_unwrap, hdg_unwrap".split())


# ## COG and HDG are quite close.
# 
# Notice above that COG and HDG are actually pretty close (as they should be).  But there are clearly places where they are different.  And in particular HDG leads COG.
# 
# To compare these signals we will attempt to first normalize them and then find the delay that minimizes the differences.

# In[23]:


# High pass filter (by subtracting lowpass).  This gets rid of nuisance DC offsets and
# emphasizes regions of rapid change...  which are the only regions where delay is
# observable.
coeff = p.butterworth_filter(cutoff=0.05, order=5)
sdf['hp_cog'] = hp_sig1 = sdf.cog_unwrap - p.smooth(coeff, sdf.cog_unwrap)
sdf['hp_hdg'] = hp_sig2 = sdf.thdg_unwrap - p.smooth(coeff, sdf.thdg_unwrap)

wdf = sdf.loc[wide_slice]
wdf = sdf
mu = wdf.cog_unwrap.mean()
c.quick_plot(wdf.index, (wdf.hp_cog, wdf.hp_hdg, wdf.cog_unwrap-mu, wdf.thdg_unwrap-mu),
             "hp_cog, hp_hdg, cog, hdg".split())


# ## High pass
# 
# We computed the high pass by first computed the low pass and then subtracting (... its one way to do it). 
# 
# The high pass rejects the absolute value of the signals (what some call DC) and leaves only the major wiggles.  This allows us to ignore that COG and HDG are affected by current (and can differ by 10 or more degrees). 
# 
# Note, this also removes any miscalibration (because of compass alignment).
# 
# **Notice how much more similar the high pass versions are.**

# In[24]:


# Normalize (for correlation)
sdf = sdf
ncog = (sdf.hp_cog - sdf.hp_cog.mean()) / (np.std(sdf.hp_cog) * np.sqrt(len(sdf.hp_cog)))
nhdg = (sdf.hp_hdg - sdf.hp_hdg.mean()) / (np.std(sdf.hp_hdg) * np.sqrt(len(sdf.hp_hdg)))

# Compute correlation (dot product) for various leads and lags
plus_minus_window = 50
res = np.correlate(ncog, nhdg[plus_minus_window:-plus_minus_window])
delay = np.argmax(res) - plus_minus_window
print(f"Found a delay of {delay}")

plt.figure()
plt.plot(range(-plus_minus_window, plus_minus_window+1), res)

logs_files = [log.log for log in G.SAIL_LOGS]
dfs, bdf = p.read_sail_logs(logs_files, path=G.DATA_DIRECTORY)

for df in dfs:
    if df is not None:
        delay, res = a.find_cog_delay(df)
        print(delay)

