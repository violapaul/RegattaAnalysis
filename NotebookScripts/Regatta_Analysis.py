#!/usr/bin/env python
# coding: utf-8

# # Regatta Analysis
# 
# Welcome to the Regatta Analysis Web "book".  This is the intro "chapter"... more follows in other "chapters". 
# 
# The goal of this project is to capture data from a sailboat during a race, or practice, so that we can go back later to understand what happened.  Particularly:
# 
# - What were the conditions?
# - How did they change during the race?
# - How fast were we sailing?
# - What angles were we sailing?
# - Could we have gone faster?  How?
# - Were our maneuvers efficient?  Where our tacks too fast, too slow, or just right?
# - Could we have made better strategic decisions? E.G. Based on current and conditions.
# - Could we have made better tactical decisions? E.G. Based on wind shifts.
# - Were weather and current predictions accurate?
# 
# One of my sailing heros is Arvel Gentry [LINK](http://www.gentrysailing.com/) (along with [Paul Cayard](https://en.wikipedia.org/wiki/Paul_Cayard) and [Frank Bethwaite](https://en.wikipedia.org/wiki/Frank_Bethwaite)).
# Arvel was an aerodynamics engineer and avid racer in San Diego (he seems to have had a big impact on the North Sails leaders as well).  It's possible that Arvel was the first to collect quantitative performance data from his boat during races in 1974! [LINK](http://www.gentrysailing.com/pdf-theory/Are-You-at-Optimum-Trim.pdf).
# 
# ![title](Data/Images/gentry_data_recorder.png)
# 
# Using this primitive device Arvel recorded boat speed and apparent wind speed.  And by adding notes during a practice run, he collected apparent wind angle and other conditions.  
# 
# **Our goals are the same.  Record data so that we can better understand what we did, and how we can do better.**
# 
# ## Table of Contents
# 
# The content in this project is split across multiple Jupter notebooks with associated python libraries as well.  Each notebook introduces a single concept that is valuable in the analysis of race data.
# 
# - This notebook will give a general overview of the data we collect, provides some examples of how that data can be viewed.
# 
# - [Race Logs](Race_Logs.ipynb) Describes our framework for organizing information about the logs captured on multiple days during multiple Regattas.  I also use this notebook to keep the table of info up to date.
# 
# - [Capturing Data from the Boat Using Canboat](Canboat_Datacapture.ipynb) Discusses how data is captured, transferred, processed, and then loaded into Python/Pandas.
# 
# - [Boat Instruments](Boat_Instruments.ipynb) 
# 
# - True Wind.
# 
# - How to find tacks, and analyze them.
# 
# - Tides and currents.
# 
# - Past weather and relating that to races.
# 
# - Polars, external data and measurement.
# 
# - And many more.
# 
# ## Python, Jupyter, Pandas, and Regatta Analysis
# 
# This "book" is written using Python, [Jupyter notebooks](https://jupyter.org/), and [Pandas](https://pandas.pydata.org/)
# 
# - Python is a powerful programming language that is also easy to use.  It is great for data analysis and visualization.  It has tremendous online support and huge set of useful libraries.  (Note, all programmers have their favorite languages, but Python is a super safe compromise.  No one wastes their time by learning Python!)
# 
# - A Jupyter notebook is a live web page that includes running Python code and supports data analysis with visualization
#    - To quote: The Jupyter Notebook is an open-source web application that allows you to create and share documents that contain live code, equations, visualizations and narrative text. Uses include: data cleaning and transformation, numerical simulation, statistical modeling, data visualization, machine learning, and much more.
# 
# - Pandas is a python library that includes great tools for data analysis (though I find its design undisciplined).
#    - To quote: **pandas** is a fast, powerful, flexible and easy to use open source data analysis and manipulation tool, built on top of the Python programming language"
# 
# The central object in Pandas is the DataFrame.  Its a table of data, with rows and columns.  After we are done with a bunch of massaging, the data from the boat will end up as a Dataframe.
# 
# ### Caveats
# 
# Jupyter is not the greatest way to make an interactive website.  It creates websites, sure, but there are specific interactive and dynamic javascript tools that might be better.  Why Jupyter?  Becasue all/most the code that is used to display and manipulate the data is right there in front of you.  Fancier, more responsive sites, require a lot of invisible programming (in Javascipt, etc) that would make it harder to customize and explore.  *Jupyter gives you some ability to explore, but it also gives you the ability to generate and modify.*
# 
# ## Some Examples
# 
# An example is worth a 1000 words.  Below is the type of data that we hope to get from the boat (though it is simplified from real data).

# In[1]:


# Import some Python libraries.  This will become familiar,  but for now just assume its necessary.
get_ipython().run_line_magic('matplotlib', 'notebook')

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qgrid

# These are libraries written for RaceAnalysis
import global_variables
G = global_variables.init_seattle()
import race_logs
import process as p
import analysis as a
import chart as c


# In[2]:


# Read some example data and load it into a DataFrame
df = pd.read_pickle(os.path.join(G.DATA_DIRECTORY, "basic_example.pd"))
# Display the DataFrame
df


# ## Each Row Tells the Story
# 
# The the table above, row contains the "current" value for each instrument.  Each row describes the instantaneous state of the boat, and it removes one of the more complex issues in the analysis of boat data.  On the boat, each instrument is separate and it sends out updates at a frequent, but not synchronous, rate.  
# 
# In other words, boat speed (SPD) is measured with a paddle wheel in the hull, and the values are sent asynchronously from the apparent wind angle (AWA) which is measured with a wind vane at the mast head.  Some instruments send rapid updates and others infrequent updates.  The onboard GPS sends full updates once per second (with GPS time and number of satellites, etc) and rapid updates 10x a second (only containing lat/lon).
# 
# The data processing pipeline will reorganize this asynchronous data into a single table, which is much more easily interpreted and analyzed.

# ## Glossary
# 
# There are some (mostly) standard names for instruments on the boat.  Here is a quick glossary that may be helpful if these are unfamiliar.
# 
# ### Instruments and their Measurements
# - AWA: apparent wind angle, the angle of the wind blowing at the top the mast (fast but noisy)
# - AWS: apparent wind speed, the speed at the mast head (fast and noisy)
# - SPD: boat speed **through the water** measured with the paddle wheel speedo in the hull (fast and noisy)
# - HDG: compass heading (on PG this is magnetic northa and not true north, though easily corrected using magnetic variation/declination).
# - COG and SOG: course and speed over ground from the GPS (these are relative to true north not magnetic on PG). These can differ from HDG/SPD because of current and leeway.
# 
# ### Computed Quantities
# - TWS: true wind speed, the speed of the wind over the ground (computed from the above quantities using the "wind triangle").
# - TWD: true wind direction, the angle of the wind blowing over the ground (see "wind triangle").
# - TWA: true wind angle, the angle of the wind over the ground reported relative the orientation of the boat (same)
# 
# ![im](Data/Images/out.png)
# 
# ### Other Quantities of Interest
# - CURRENT: Speed of water flow triggered by tides.
# - DEPTH: depth of water beneath the sensor.
# - TIDES: Principally used to understand depth, and predict currents

# ## Loading an Entire Race
# 
# The DataFrame above is super brief, and it shows just a few rows and a subset of the columns.  Below we will load an entire day on the water.

# In[3]:


# Info about all race logs are stored in a DataFrame.
log_info = race_logs.read_log_info()

# The data in this table can be editted using a QGrid Control.  Click on the column header to sort.  Click again 
# to sort in a different order.  Double click on a cell to edit.
w = qgrid.show_grid(log_info, show_toolbar=True)
display(w)


# In[4]:


# We can use fancy Pandas techniques to find one of the logs

# does the filename start with?
match = log_info.file.str.startswith("2019-11-16")

# This returns a set of bools
print(list(match))


# In[5]:


# Grab the first matching exmample
example = log_info[match].iloc[0]
example


# In[6]:


df = race_logs.read_log_file(example.file, discard_columns=True, skip_dock_only=False, trim=True, 
                            cutoff=0.3)

# Trim off the uninteresting pre/post race bits
df = df.loc[example.begin : example.end]

# Draw the track on a map
chart = c.plot_chart(df)
c.draw_track(df, chart, color='green')


# In[7]:


# Display a bit of the table (note, the notebook will only show a few of the rows and 
# columns, notice the "..." which appear)
df


# In[8]:


# Display the full list of columns
df.columns


# In[9]:


# We'll store information about the meanings of these columns in a DataFrame!
column_df = pd.read_pickle(os.path.join(G.DATA_DIRECTORY, "column_info.pd"))

# And display in an edittable grid.  Be sure to scroll around.
grid = qgrid.show_grid(column_df, show_toolbar=True)
grid


# In[10]:


# If you do update the table shown above, then this will save the changes (which are not saved by default)

if False:
    new_df = w.get_changed_df()
    new_df.to_pickle(os.path.join(G.DATA_DIRECTORY, "column_info.pd"))


# In[11]:


# As in the initial example, we can focus on the critical columns.
good_cols = "row_times latitude longitude awa aws hdg spd sog cog".split()
# Note, this split biz is just a way for me to quickly type a long list with out all the 
# punctuation.  Rather than ['a', 'b', 'c'] I type "a b c".split()
print(good_cols)
df[good_cols]


# In[12]:


# We can graph values versus time

# Recall that distance is stored in METERS (and METERS PER SECOND).
plt.figure()
df.spd.plot()


# In[13]:


# Or we can plot quantities and compare them

c.quick_plot(df.index, (df.spd, df.aws), ["spd", "aws"])


# # Conclusions
# 
# In 2020, we have many more automated tools than Arvel Gentry did in 1974.  Our goals remain the same.  Understand conditions and learn how to sail better.
