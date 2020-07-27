#!/usr/bin/env python
# coding: utf-8

# # Foo bar
# 
# - one
# - two
#   - three
#   
# $ \int x$
# 
# # Todo: Ideas for Using Data to Improve Racing Skills 
# 
# 
# - [ ] Current Maps of Shilshole Bay
# - [ ] Optimal Tacking
# - [ ] Jibe Angles
# - [ ] Jib Car Experiements
# - [ ] Shroud Tension Settings 
# - [x] Relating Twist to Sail Log Data
# - [ ] Compute Your Own Polars
# 
# 
# 
# # Current Maps
# 
# 
# ### Current Maps ###
# 
# # Todo: Ideas for Using Data to Improve Racing Skills 
# 
# 
# - [ ] Current Maps of Shilshole Bay
# - [ ] Optimal Tacking
# - [ ] Jibe Angles
# - [ ] Jib Car Experiements
# - [ ] Shroud Tension Settings 
# - [x] Relating Twist to Sail Log Data
# - [ ] Compute Your Own Polars
# 
# 
# 
# # Current Maps
# 
# The Puget Sound is a long network of ocean waterways bounded on the
# north by the Oympic Pennisula, the Straight de Juan de Fuca, Whidbey
# Island, and Deception Pass and proceeding south for 80 miles to the
# southern end of Olympia, the capitol of Washington State. It is part
# of the bigger Salish Sea Basin which includes the inland sea around
# Vancouver Island as well. The tidal flow of water from the Puget Sound
# drains in and out through Admirality Inlet and Deception Pass creating
# strong currents at certain predictable times. As sailors, we need to
# understand these currents well enough to take advantage of them when
# we can and minimze the negative impact of adverse current when we need
# to.
# 
# 
# 
# 
# 
# 
# [I'm an inline-style link](https://www.google.com)
# 
# If we sail 100 days in a year and collect data, we should be able to
# find a good model of the local current at most times each day and in
# most locations. There will be tricky spots known as singularities in
# the current flow pattern, but we will assume we won't be stuck in a
# tricky spot very long.  We want to know the flow patterns detectable
# over on order of 100 meters. With additional data, we could the
# improve accuracy of the model and focus in on strong current flow
# areas.
# 
# Goals: Given the nautical charts we have of Shilshole Bay, plot
# 3 dimension current flow vectors with a magnitude and direction at
# each location we sail sampling every 5 minutes.  The third dimension
# is determined by time.  We don't want to use GPS time or clock time,
# we want to use a measure of time with respect to the tides.  
# 
# There are two high tides and two low tides each day. One is noticibly
# bigger than the other and they alternate like this.
# 
# 
# 
# 
# 
# 
# 

# In[ ]:




