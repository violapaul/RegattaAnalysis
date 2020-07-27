#!/usr/bin/env python
# coding: utf-8

# # Currents
# 
# ## Motivation
# 
# Current can be the determining factor in many races, particularly in regions with large tidal flows (like the Puget Sound and other popular sailing areas such as England and the Atlantic coast of France).  We'll use the PNW as a concrete example.
# 
# - Tides can be up to 14 ft, so there is a lot of flow.
# - Current in the center of a channel can be strong, up to 2kts.
#   - Most often its less than a 1kt; in narrow channels (Agate Pass) it can be 5kts.
# - There can be significant back eddy counterflows.
# - Current when running opposite to winds can increase waves.
# 
# In general, 0.2kts can make the difference between a win and a loss. 
# 
# ![im](Data/Images/current_tidal_current.png)
# 
# Above, an example of NOAA predicted tidal currents from the Puget Sound.  Below, a "tide print" which shows a much higher resolution picture of back eddies.  This first is from [Tidal Currents of the Puget Sound](https://www.starpath.com/catalog/books/1986.htm) the second from [Tide Prints](https://eos.ucs.uri.edu/seagrant_Linked_Documents/washu/washum77001.pdf) (which appears to be out of print, but provides the high image quality).
# 
# ![im](Data/Images/current_tide_print.png)
# 
# There are multiple sources of current data, and we will use them all:
# 
# - NOAA collects current data at multiple depths and multiple locations over a period of weeks. [Link](https://tidesandcurrents.noaa.gov/cdata/StationList?type=Current+Data&filter=historic&pid=36)
#   - This data is then used to predict currents in the future... most likely using references to the "Harmonic Constituents" used to predict the tides.
#   
# - The University of Washington has study tidal flows in the Puget Sound for many, many years.  The produced a scale model, the [Puget Sound Oceanographic Model](https://www.eopugetsound.org/articles/puget-sound-model-summary), and then used it to predict flows.  Using 1950's era tools (i.e. floating bits of foam and long exposure images) they captured these flows and published them in Tide Prints (above).  The direction of the flows (set) are considered accurate, but the magnitude (drift) are not.  There may also be issues with timing.  See this awesome [video](https://youtu.be/0-2y_j66xAE) to give you a sense of how this worked.
# 
# - As we sail we collect measurements of absolute position (SOG/COG) and the speed through the water (SPD/HDG).  The difference is principally current (though the boat also undergoes a drift to leeward called leeway).
# 
# 
# Some additional links:
# - Wikipedia on tides
#   - https://en.wikipedia.org/wiki/Theory_of_tides
#   - https://en.wikipedia.org/wiki/Tide#Constituents
#   
# - UW's more recent work understand flows in the Puget Sound.
#   - http://faculty.washington.edu/pmacc/LO/tides_background.html
#   - http://faculty.washington.edu/pmacc/LO/P_tracks_barber.html
#   - http://www.prism.washington.edu/story/Validating+the+circulation+model
#   - https://salish-sea.pnnl.gov/
#   - Got to figure this out... looks like a modern computer version of the "particle tracks" from the 
# 

# In[ ]:


# Load the tidal prints

import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.signal

import cv2

TIDE_PRINT_PATH = "Data/Currents/TidePrints"

tide_print_images = 


# In[ ]:




