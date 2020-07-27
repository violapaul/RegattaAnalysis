#!/usr/bin/env python
# coding: utf-8

# # An Example Notebook
# 
# Just a simple exmaple of a literate programming notebook.

# In[1]:


# Let's import JSON, its handy!
import json


# In[2]:


# notebook

# Here is a cell that will not make it into the "module"

json_example = """
{"menu": {
  "id": "file",
  "value": "File",
  "popup": {
    "menuitem": [
      {"value": "New", "onclick": "CreateNewDoc()"},
      {"value": "Open", "onclick": "OpenDoc()"},
      {"value": "Close", "onclick": "CloseDoc()"}
    ]
  }
}}
"""

json_dict = json.loads(json_example)
json_dict


# In[3]:


# This cell is part of the module

def count_keys(json_dict):
    "Count the number of keys in a JSON dict."
    # This code is just an example... there are better ways!
    count = 0
    for k in json_dict:
        count += 1
    return count


# In[4]:


# notebook 

count_keys(json_dict)


# In[5]:


import sys
import notebook
import notebook.services.contents.filemanager


# In[5]:


dir(notebook)


# In[6]:


import notebook.services


# In[ ]:




