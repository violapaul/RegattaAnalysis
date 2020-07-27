#!/usr/bin/env python
# coding: utf-8

# In[20]:


import os

import copy
import math
import itertools as it
import importlib

import pandas as pd
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets

from numba import jit

from pyproj import Proj
import cv2

import boat_shape
from boat_shape import OUTLINE as BOAT_OUTLINE

import process as p
import analysis as a
import chart as c
import global_variables as G
from utils import DictClass


# In[8]:


cd /Users/billey/scode/slogs


# In[99]:



SAMPLES_PER_SECOND = 10

# Tacks take 20 seconds or more
# Compute the approximately constant TWD through the tack
# Compute the VMG through the tack (using TWD)
# Find the beginning of the tack (rudder??)
# Measure the lost boat lengths.
# Measure the time before returning to full speed
# Measure the initial "tack angle" and final "tack angle"
def find_tacks(df, buffer=20):
    if False:
        skip = 1
        dt = skip / SAMPLES_PER_SECOND
        sdf = rdf.copy()
    else:
        # a tack is a transition from close hauled port to stbd (and vice versa).
        skip = 5
        dt = skip / SAMPLES_PER_SECOND
        sdf = df[::skip].copy()

    # remove noise
    # sdf['hdg_f'] = scipy.ndimage.median_filter(uhdg, int(10 / dt), origin=0)
    # sdf['awa_f'] = scipy.ndimage.median_filter(np.array(sdf.awa), int(10 / dt), origin=0)

    sdf['hdg_f'], _ = p.exponential_filter_angle(np.array(sdf.hdg), 0.995, 4)
    sdf['awa_f'], _ = p.exponential_filter_angle(np.array(sdf.awa), 0.995, 4)

    # find regions where boat is always close hauled for a period of time
    period = int(5 / dt)
    sdf['is_stbd_hauled_raw'] = scipy.ndimage.minimum_filter1d((sdf.awa_f > 15) & (sdf.awa_f < 45), period)
    sdf['is_port_hauled_raw'] = scipy.ndimage.minimum_filter1d((sdf.awa_f < -15) & (sdf.awa_f > -45), period)

    # expand so that we can find overlaps, i.e. tacks
    sdf['is_stbd_hauled'] = scipy.ndimage.maximum_filter1d(sdf.is_stbd_hauled_raw, 4 * period)
    sdf['is_port_hauled'] = scipy.ndimage.maximum_filter1d(sdf.is_port_hauled_raw, 4 * period)

    # if you are in the overlap between expanded stbd and port hauled then you are
    # tacking!
    sdf['is_tack'] = (sdf.is_stbd_hauled & sdf.is_port_hauled)

    if False:
        c.quick_plot(sdf.index, (sdf.awa, sdf.awa_f, 100*sdf.is_tack, 100*sdf.is_stbd_hauled, 100*sdf.is_port_hauled),
                     "(sdf.hdg, sdf.hdg_f, sdf.awa, sdf.awa_f)".split(),
                     fignum=1)

        c.quick_plot(sdf.index, (sdf.hdg, sdf.hdg_f, sdf.awa, sdf.awa_f, 100*sdf.is_tack),
                     "(sdf.hdg, sdf.hdg_f, sdf.awa, sdf.awa_f)".split(),
                     fignum=1)

    res = []
    for s, e in p.find_runs(np.array(sdf.is_tack)):
        s_extend = int(s-(buffer/dt))
        ##SB:e_extend = int(e+(buffer/dt))
        e_extend = int(e+4*(buffer/dt))   ## rescale tack window, finding more with each increment 
        hdg_before = np.median(sdf.hdg_f.iloc[s_extend:s])
        hdg_after = np.median(sdf.hdg_f.iloc[e:e_extend])
        tack_angle = np.abs(p.angle_diff(hdg_before, hdg_after))
        if True or tack_angle < 115 and tack_angle > 65:
            res.append(slice(sdf.index[s_extend], sdf.index[e_extend]))

    return res, sdf

def show_vmg(df, df_slice):

    df_slice = slice(79140, 105810)

    delay = 16
    dt = 1
    ss = slice(df_slice.start, df_slice.stop, dt)
    dss = slice(ss.start+delay, ss.stop+delay, dt)
    
    mdf = df[ss]
    ddf = df[dss]

    slow_coeff = p.butterworth_filter(0.03, 5)

    vmg = mdf.spd * a.cos_d(np.asarray(ddf.twa))
    ptwa = np.asarray(ddf.twa)
    ptwa[ptwa < 0] = - ptwa[ptwa < 0]
    ptwa = p.smooth(slow_coeff, ptwa, causal=False)

    pawa = np.asarray(ddf.awa)
    pawa[pawa < 0] = - pawa[pawa < 0]

    pvmg = polars_spline(ddf.tws, ptwa-10)

    a.quick_plot(mdf.index,
                 (ddf.tws, ptwa/10, pawa/10, mdf.spd, vmg, pvmg),
                 "tws ptwa, pawa, spd, vmg, polar".split(),
                 fignum=6)

    
    vog_n = dt * ddf.sog * a.north_d(ddf.cog)
    vog_e = dt * ddf.sog * a.east_d(ddf.cog)

    tw_n = dt * ddf.tws * a.north_d(ddf.twd)
    tw_e = dt * ddf.tws * a.east_d(ddf.twd)

    hdg = mdf.hdg + 13.2
    btv_n = dt * mdf.spd * a.north_d(hdg)
    btv_e = dt * mdf.spd * a.east_d(hdg)

    cur_n = 10.0 * (np.asarray(vog_n) - np.asarray(btv_n))
    cur_e = 10.0 * (np.asarray(vog_e) - np.asarray(btv_e))

    chart = a.plot_chart(mdf, 3)
    longitudes = np.asarray(mdf.longitude)
    latitudes = np.asarray(mdf.latitude)
    pos = np.vstack(a.MAP(longitudes, latitudes)).T - (chart.west, chart.south)

    color = 'blue'
    hwidth = dt/5
    for (east, north), ve, vn in it.islice(zip(pos, vog_e, vog_n), 0, None, 10):
        chart.ax.arrow(east, north, ve, vn, head_width=hwidth, length_includes_head=True, color=color)

    color = 'green'
    for (east, north), ve, vn in it.islice(zip(pos, tw_e, tw_n), 0, None, 10):
        chart.ax.arrow(east, north, ve, vn, head_width=hwidth, length_includes_head=True, color=color)

    color = 'red'
    for (east, north), ve, vn in it.islice(zip(pos, btv_e, btv_n), 0, None, 10):
        chart.ax.arrow(east, north, ve, vn, head_width=hwidth, length_includes_head=True, color=color)

    color = 'orange'
    for (east, north), ve, vn in it.islice(zip(pos, cur_e, cur_n), 0, None, 10):
        chart.ax.arrow(east, north, ve, vn, head_width=hwidth, length_includes_head=True, color=color)

def show_tacks():
    DATA_DIRECTORY = '/Users/billey/scode/slogs'
    #DATA_DIRECTORY = '/Users/viola/canlogs'
    example = DictClass(log='2019-11-16_10:09.pd', begin=41076, end=111668)
    example = DictClass(log='2019-10-26_09:40.pd', doc='Grand Prix Saturday.', begin=40503, end=87408)
    example = DictClass(log='2019-10-12_09:45.pd', doc='CYC PSSC Day 1', begin=19081, end=233893)
    example = DictClass(log='2019-11-16_10:09.pd', doc='Snowbird #1.', begin=42548, end=111668)
    sail_logs = [example.log]

    dfs, bdf = p.read_sail_logs(sail_logs, discard_columns=False, skip_dock_only=False, trim=True, path=DATA_DIRECTORY, cutoff=0.3)
    df = dfs[0]
    cdf = df.copy()

    chart = c.plot_chart(df, 3)
    c.draw_track(cdf, chart, color='green')
    chart = a.plot_track(cdf, 1, skip=1)
    
    rdf = df.loc[example.begin : example.end]

    chart = c.plot_track(rdf, 1)

    tacks, tdf = find_tacks(rdf)
    chart = c.plot_chart(rdf, 3)
    c.draw_track(rdf, chart, color='red')
    for tack_slice in tacks:
        tdf = rdf.loc[tack_slice]
        lon, lat = np.asarray(tdf.longitude), np.asarray(tdf.latitude)
        c.draw_track(tdf, chart, color='green')

    tdf = rdf.loc[tacks[5]]
    chart = c.plot_chart(rdf, 3)
    c.draw_track(rdf, chart, color='red')
    c.draw_track(tdf, chart, color='green')
    uhdg = np.degrees(np.unwrap(np.radians(tdf.hdg)))
    scale = G.MS_2_KNOTS
    c.quick_plot(tdf.index, (scale*tdf.spd, scale*tdf.aws, tdf.rudder),
                 legend = "(scale*tdf.spd, scale*tdf.aws, tdf.rudder)".split(),
                 fignum=2, clf=True)


    scale = 10 * G.MS_2_KNOTS
    c.quick_plot(tdf.index, (scale*tdf.spd, uhdg, tdf.awa, scale*tdf.aws, tdf.rudder),
                 legend = "(scale*tdf.spd, uhdg, tdf.awa, scale*tdf.aws, tdf.rudder, awa_f)".split(),
                 fignum=2, clf=True)

    vmg = tdf.spd * G.MS_2_KNOTS * p.cos_d(tdf.twa)

    c.quick_plot(tdf.index, (tdf.tws * G.MS_2_KNOTS, tdf.twa/27, tdf.spd * G.MS_2_KNOTS, tdf.awa/27, vmg),
                 legend = "tws, twa/27, spd, awa/27, vmg".split(),
                 fignum=1, clf=True, title='raw data')


    c.quick_plot(tdf.index, (tdf.tws * G.MS_2_KNOTS, tdf.twd/27, tdf.awa/27, tdf.twa/27),
                 legend = "(tdf.tws, tdf.twd/27, tdf.awa/27, tdf.twa/27".split(),
                 fignum=2, clf=True, title='raw data')
    
def quick_plot(index, data, num=None, clf=True, title=None, legend=None, s=slice(None, None, None)):
    fig = plt.figure(num=num)
    if clf:
        plt.clf()
    for d in data:
        plt.plot(index[s], d[s])
    if legend is not None:
        plt.legend(legend, loc='best')
    if title is not None:
        fig.suptitle(title, fontsize=14, fontweight='bold')

    


# In[100]:


DATA_DIRECTORY = '/Users/billey/scode/slogs'
#DATA_DIRECTORY = '/Users/viola/canlogs'
example = DictClass(log='2019-11-16_10:09.pd', begin=41076, end=111668)
example = DictClass(log='2019-10-26_09:40.pd', doc='Grand Prix Saturday.', begin=40503, end=87408)
example = DictClass(log='2019-10-12_09:45.pd', doc='CYC PSSC Day 1', begin=19081, end=233893)
example = DictClass(log='2019-11-16_10:09.pd', doc='Snowbird #1.', begin=42548, end=111668)
sail_logs = [example.log]

dfs, bdf = p.read_sail_logs(sail_logs, discard_columns=False, skip_dock_only=False, trim=True, path=DATA_DIRECTORY, cutoff=0.3)
df = dfs[0]
cdf = df.copy()

#chart = c.plot_chart(df, 3)
#c.draw_track(cdf, chart, color='green')
#chart = a.plot_track(cdf, 1, skip=1)

rdf = df.loc[example.begin : example.end]

#chart = c.plot_track(rdf, 1)

tacks, tdf = find_tacks(rdf)


# In[103]:


good_tacks=[1,5,6,9,11,14]  ## 1,5 got added by extending window to right
tacks


# In[109]:


### Let's check the rudder movement looks right on these tacks
for i in range(len(tacks)):
    a_tack=df.loc[tacks[i]]   ## trying the old loc trick!
    quick_plot(a_tack.index, 
               (a_tack.awa/5, a_tack.rudder), 
               legend='awa/5 rudder'.split(),
               title='Tack Number '+str(i), s=slice(None, None, 10))


# In[ ]:





# In[71]:


tacks[0]


# In[110]:


quick_plot(df.index, (df.awa/5, df.rudder), 
            legend='awa/5 rudder'.split(),
            title='Tack Number '+str(i), s=tacks[0])  
## same problem as without the loc trick


# In[112]:


for i in range(len(tacks)):
    a_tack=df.loc[tacks[i]] 
    quick_plot(a_tack.index, 
               (a_tack.awa/5, a_tack.rudder, a_tack.spd), 
               legend='awa/5 rudder spd'.split(),
               title='Tack Number '+str(i), s=slice(None, None, 10))


# In[113]:


for i in range(len(tacks)):
    a_tack=df.loc[tacks[i]] 
    quick_plot(a_tack.index, 
               (abs(a_tack.awa/10),  a_tack.spd, 3+a_tack.rudder/20), 
               legend='abs(awa/10)  spd rudder/20'.split(),
               title='Tack Number '+str(i), s=slice(None, None, 10))


# In[115]:


a_tack=df.loc[tacks[12]] 
quick_plot(a_tack.index, 
           (abs(a_tack.awa),  a_tack.spd, 3+a_tack.rudder),
           legend='abs(awa)  spd rudder'.split(),
           title='Tack Number '+str(12), s=slice(None, None, 10))


# In[26]:


df


# In[28]:


df.columns


# In[31]:


df.rudder.describe()


# In[68]:


quick_plot(df.index, 
           (df.awa/5, df.rudder), 
           legend='awa/5 rudder'.split(),
           title='stuff', s=slice(None, None, 10))


# In[ ]:





# In[65]:


### Let's check the rudder movement looks right on these tacks
for i in range(len(tacks)):
    atack=df[tacks[i]]

    quick_plot(atack.index, 
               (atack.awa/5, atack.rudder), 
               legend='awa/5 rudder'.split(),
               title='Tack Number '+str(i), s=slice(None, None, 10))


# In[49]:


type(tacks)
len(tacks)


# In[25]:


c.plot_chart(df, 3)


# In[122]:


wide_slice = slice(98000, 100000)
wdf = df.loc[wide_slice]


# In[132]:


# Let's drill down and look at 5 seconds (50 samples)
narrow_slice = slice(98400, 98450)
ndf = df.loc[narrow_slice]
plt.figure()
ndf.rawa.plot()


# In[118]:


ndf.rawa


# In[119]:


ndf.spd


# In[120]:


ndf.tws


# In[123]:





# In[127]:


narrow_slice = slice(98600, 98950)
ndf = df.loc[narrow_slice]


# In[134]:


ndf.rtwa


# In[133]:


ndf.spd.describe()


# In[ ]:




