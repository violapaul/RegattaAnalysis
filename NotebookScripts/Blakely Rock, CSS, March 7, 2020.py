#!/usr/bin/env python
# coding: utf-8

# #  Blakely Rock, CSS, March 7, 2020
# 
# [Facebook RaceQs video](https://www.facebook.com/groups/J105PNWFleet/permalink/1347056552144637/)
# 
# [RaceQs Link](https://raceqs.com/tv-beta/tv.htm#userId=1073253&divisionId=64030&updatedAt=2020-03-09T03:39:42Z&dt=2020-03-07T11:04:54-07:00..2020-03-07T15:37:34-07:00&boat=Peer%20Gynt)
# 
# ### Crew 
# 
# Laura Bow, Peter Mast/SpinTrim, Sara Pit/Strategist, Steve Mast/JibTrim, Dave Main/Tactician, Paul Driver
# 
# ### Conditions
# 
# Nice day, ending sunny and relatively warm (44F).  Overall solid and consistent wind from the South (10kts).  Sea state 1-2. Some rain in the first half.
# 
# ## Summary
# 
# **Speed** Good speed all around.  We matched well with the other boats upwind.  Perhaps we lost a bit to Insubordination on the first upwind tack. Downwind was great.  We did very well against the fleet, and beat both Corvo and Jaded when directly matched (and along the way we passed Creative).
# 
# After staring at RaceQs, we often sail higher on the upwind (2-3 degrees?) and lower on the downwind (trying to optimize VMG rather than speed).  On the downwind VMG is great.  Upwind, looks like a wash.
# 
# I had a bit of fun pumping the main on the downwind (though only for a short time).  We got in 2-3 foot rollers, and I pumped on each wave.  It felt like we gained boats on Corvo that way.  Rule 42.3.c.  
# 
# **Start** (pin was on the right side).  Long line.  Most fought for starboard tack at the pin end.  We started at the committee boat on port tack, with a good head of steam.  There was a chance we could have passed the whole fleet, but needed to tack to avoid Insubordination (it was close).  As driver, I did not drive up to close hauled fast enough, and perhaps I could have timed the start a few seconds better.  That first tack might have been a bit closer, giving us a better leebow, but it was tricky.  *But generally I liked this start.  We we were in a good position and always had clear air!*
# 
# **Transitions**: Overall we did great on tacks and gybes.  The (single) douse was a bit of a mess.  Driver's fault.  As we approached the mark we were fast approaching a symmetric spin boat, on a deeper approach angle.  They got to the zone first, and we needed to avoid.  We essentially got into a holding pattern waiting, and we actually drove down. **At that point we should have doused early while waiting.**  In the end we waited, and did not start the douse until we started rounding up.  We then had to drive back down, overshooting the mark by 5 boat lengths.  
# 
# We also learned a lesson on standardizing the language around tack timing:
# 
# - If possible Driver calls out tacking in 10 seconds
#     - No one does anything.  Just be ready.
#     - Now is the time to scream if we cannot tack (though it may not be optional).
# - 5 seconds before the tack Driver calls "prepare to tack"
#     - Jib trimmer jumps down and ensures the lines are ready.  Calls "prepared".
#     - No one else moves, but they mentally prepare.
# - Driver calls "tacking in 3-2-1 tacking", driving up.
#     - Others begin to move as late as possible, but be ready to move fast.
#     - Main can pull on (if there is time).
# 
# **Strategy** Overall great job. We were looking to get to the right, but the start precluded that.  We should probably have worked harder to get back to the right afterward.  Ultimately at the end of the long upwind leg we lost track of some of the fastest boats (we thought Moose and Insub were to the left, but they were actually to the right).  When we saw the lift at the end of the windward leg **the entire "fast" fleet was to the right**, and we lost 10+ lengths.
# 
# ### Learnings
# 
# **Plan the rounding AND the douse.**  You cannot be overly focused on the rounding without a complimentary douse plan.
# 
# **Find the rest of the fleet and understand how that affects strategy.**  Track the fast boats! If the fast boats are to your right, with big leverage, then you need to head right (unless you know better...  but in any case don't give Creative leverage!!).
# 
# **Strategy makes all the calls on tacks/gybes.** Except for emergencies.  No question, or comment, should every be interpretted as overruling strategy.
# 
# **VMG sailing is working for us, particularly downwind.**  Don't settle for speed, work on speed and angle (up and down).
# 
# 
# ### From Al Hughes on Facebook
# 
# *Fun day on the water today at CYC Blakely Rock Race. Good to see so many boats out on a good day with wind , sun and showers. The Insubordinates and Corvo showed good pace and direction from the start. Gary pushed his way up to round the Rock first, followed by Moose Unknown who seemed to be charging through us all. Creative and Jaded had a good battle back and forth up the Bainbridge shore, to round 3 & 4. On the long run to N, Moose stretched out on everyone by seemingly having both good speed and a nose for the shifts. Peer Gynt had a major rally in the second part of the run. It appeared that mark N was in a very strange location being quite close to mid-channel buoy. Rounding N was Moose, Insubordination, Peer Gynt, Jaded and Creative starting the beat home. No too long after we rounded N, lots of chatter on the vhf revealed the buoy was adrift. So race abandoned and we all got some good practice. Thanks for the good turnout. Last SBYC Iceberg next Saturday, then Scatchet Head on the 21st.*
# 
# ### West Point Wind
# 
# ![im](Data/Images/Screen%20Shot%202020-03-08%20at%208.20.08%20PM.png)
# 
# ### West Point Tides:
# 
# ![im](Data/Images/Screen%20Shot%202020-03-08%20at%208.19.51%20PM.png)
# 
# ### Golden Gardens Wind
# 
# ![im](Data/Images/Screen%20Shot%202020-03-08%20at%208.20.24%20PM.png)

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
importlib.reload(race_logs)


# In[3]:


dfs, races, big_df = race_logs.read_dates(["2020-03-07"])
# dfs, races, big_df = race_logs.read_dates(["2019-11-16"])
df = dfs[0]


# In[4]:


races[0]


# In[5]:


df


# In[6]:


ch = c.plot_chart(df)
c.draw_track(df, ch)


# In[7]:


c.quick_plot(None, (df.twd, df.stwd, df.boat_twd))


# In[8]:


ch = c.plot_track(df)


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




