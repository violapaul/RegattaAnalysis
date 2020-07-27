#!/usr/bin/env python
# coding: utf-8

# # Race Metadata
# 
# ## Motivation
# 
# What were the conditions on a particular day?  The crew?  What sort of jib settings did we use?  The finishing position?  What were the shroud settings?  How did they perform?  
# 
# I started out by writing a email for each race, trying to including learnings, conditions, results.  I moved to creating a Google doc, easier to edit and update. And then I moved creating a Jupyter notebook for each race day, easier to included data from the actual race all in one place.
# 
# All of these are great.  But they take a lot of work repetitive work to setup.
# 
# But, I like structured data.  I like metadata which can be searched and cross-referenced.  Data should be easy to edit and update. I like "reasonable" renderings of that data.  I don't like to have data squirreled away in databases where it is hard to view and search (without running a fancy tool).  
# 
# The problems with this approach:
# 
# - Lots and lots of repeated work.  Each email/gdoc/notebook is a vague copy of the previous, updated with new info.  This copy/edit process is annoying.
#   - E.G. One step is to grab the weather/tides and this took a while by hand.
# - The data is locked in a human readable document, not in a machine readable representation.
# - No way to look at all the data in one place.  Where can we look to see trends or issues that are inconsistent?
# 
# The solution is to store all this metadata in a single datastructure which can then be created/edited/rendered more rapidly.
# 
# 
# ## Overview
# 
# Code to process race metadata and associate with race logs.
# 
# The data is stored in Yet Another Markup Language (YAML).
# 
# - Good reference to start: [YAML tutorial](https://rollout.io/blog/yaml-tutorial-everything-you-need-get-started/)
# - [The official reference](https://yaml.org/) Its written in YAML.
# 
# **This is a literate notebook.**
# 
# ## Caveats and concerns
# 
# - YAML, as edited by a human author, does not support a strong schema.  Its easy to mess things up, with typos, missing fields, incorrectly named fields, etc.
# 

# In[1]:


# notebook - literate notebook tag, not part of the module 
get_ipython().run_line_magic('matplotlib', 'notebook')
from IPython.display import display, Markdown, Latex


# In[2]:


import yaml
import logging

# A bit of magic here to ensure we have the best loader/dumper.  Specifying this is required when 
# calling load/dump (below).
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


# In[3]:


# notebook
example = """
date: 2020-04-16
name: >-
  Tune-up with Creative
purpose: tune_up
conditions: >-
  Beautiful day. Winds were 3 quickly building to 10ish. Flat
  seas. Upwind to Pt. Wells buoy, raised and raced home to the hamburger.
performance: >-
  Good height and speed vs. Creative on the way upwind. Perhaps a bit
  slow at first downwind, exploring to tradeoffs between depth and
  speed.  Best downwind speed when I was at the shrouds and Sara had a
  hand on the mainsheet.
learnings: >-
  Let the sails out for downwind: both main and kite.  Stand forward
  if possible.
raceqs_video: "https://youtu.be/9a5bLeZw8EM"
raceqs: "https://raceqs.com/tv-beta/tv.htm#userId=1146016&divisionId=64190&updatedAt=2020-04-17T18:05:59Z&dt=2020-04-16T15:43:47-07:00..2020-04-16T17:39:12-07:00&boat=Creative"
segments:
  - start: 
"""

race_metadata = yaml.load(example, Loader=Loader)
display(race_metadata)


# In[4]:


def is_url(s):
    return s.startswith("http")  # Not great, but OK for now

def make_title(key):
    """"
    YAML keys are python keywords (lowercase and separated by underscores).  This converts to a pretty 
    and printable string.
    """
    words = key.split("_")
    words = [w.capitalize() for w in words]
    return " ".join(words)

def display_url(link_text, url):
    "Displays a markdown URL."
    display(Markdown(f"[{link_text}]({url})"))
    
def display_section(name, text):
    "Displays a markdown section with text."
    display(Markdown(f"## {name}"))
    display(Markdown(text))

def display_summary(data):
    display(Markdown(f"# {data['name']}: {data['date']}"))

    for k in "conditions performance learnings".split():
        display_section(k.capitalize(), data[k])
    display(Markdown("## Links"))
    for k in "raceqs raceqs_video".split():
        display_url(make_title(k), data[k])


# In[5]:


# notebook
display_summary(race_metadata)


# In[6]:


def read_yaml(yaml_path):
    with open(yaml_path, 'r') as yaml_stream:
        race_yaml = list(yaml.load_all(yaml_stream, Loader=Loader))
    res = {}
    for record in race_yaml:
        res[record['date']] = record
    return res


# In[7]:


# Notebook

yaml_path = "Data/metadata.yml"
race_data = read_yaml(yaml_path)

display_summary(race_data['2020-04-19'])


# In[8]:


race_data


# In[ ]:


import functools

def schema_merge(schema_list):
    return functools.reduce(lambda a,b: dict(a, **b), schema_list)

def schema_extract_dict(race_dict):
    res = {}
    for key, val in race_dict.items():
        if type(val) == dict:
            res[key] = schema_extract(val)
        elif type(val) == list:
            res[key] = [schema_merge([schema_extract(v) for v in val])]
        else:
            res[key] = type(val)
    return res


# In[ ]:




