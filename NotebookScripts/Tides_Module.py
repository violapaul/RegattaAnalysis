#!/usr/bin/env python
# coding: utf-8

# # Tides
# 
# This notebook is closely related to the Current_Module.  Please take a look there as well.
# 
# [Awesome Accessible Description of Tides](http://faculty.washington.edu/pmacc/LO/tides_background.html)
# 
# NOAA is the source of all information about Tides.
# 
# 
# 
# ## Approach
# 
# Loads a cached set of NOAA tide prediction data and then looks up the closest tide for
# any time.
# 
# 
# ## TODO
# 
# - Figure out the harmonics...  much more compact.
#   - This will help with analysis of currents as well.
# 
# 

# In[2]:


import pandas as pd
import numpy as np
import os
import os.path
import urllib


# In[3]:


# These are libraries written for RegattaAnalysis
from global_variables import G  # global variables

G.init_seattle()


# In[4]:


TIDES_PATH = os.path.join(G.DATA_DIRECTORY, 'Tides/seattle_tides2.pd')
TIDES_DF = pd.read_pickle(G.TIDES_PATH)


# In[5]:


TIDES_DF


# In[ ]:


def tides_at_closest(times):
    base_time = TIDES_DF.date_time.iloc[0]
    last_time = TIDES_DF.date_time.iloc[-1]    
    if (times < base_time).any():
        raise Exception("Tide time too early {0}".format(times.min()))
    if (times > last_time).any():
        raise Exception("Time time too late {0}".format(times.max()))
    # Compute the nearest ti
    tide_index = ((times - base_time).dt.total_seconds() / 360).round().astype(np.int)
    return TIDES_DF.iloc[tide_index].prediction.values


# In[ ]:




