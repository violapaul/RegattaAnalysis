#!/usr/bin/env python
# coding: utf-8

# # Race Video Rendering
# 
# Read the chapter on [Garmin Fit Files](Garmin_Fit_files.ipynb) if you'd like additional info on FIT files.  They are mentioned below but are not critical to this chapter.
# 
# Below is a snapshot of a hypothetical annotated sailing video, with instrument gauges overlayed.
# 
# ![im](Data/Images/virb_edit_screenshot.png)
# 
# Garmin is a pioneer both in personal fitness, marine instruments, and action cameras.  Most Garmin devices produce FIT files which capture data from sensors, and this has led to a cool additional functionality: **you can overlay FIT data onto a video** (see image above).  This "dashboard" is created using a video editing tool from Garmin called [Virb Edit](https://buy.garmin.com/en-US/US/p/573412).  Virb is the brand name for Garmin cameras, and Virb Edit is used to edit video and produce final version.  Virb Edit can also be used to merge the video and the FIT data, producing awesome overlays.
# 
# Virb Edit calls FIT data "GMetrix" (for "Garmin Metrics"?).  There are a very wide range of "instrument" types.  And you can put together a pretty professional video.  Virb Edit can be used with any video source (above is video from a GoPro).  
# 
# The FIT file format is a mostly public standard, except for one **super annoying** gap: the marine data like that shown above (e.g. AWA, TWA, etc) is **not part of the standard** (more on that below).  This data actually came from a different Garmin boat (a J/99 called *One Life*).   Virb Edit can read this FIT file, and will overlay the encoded info on the video (even though it is unrelated to my boat).  Ultimately the lack of documentation has forced the development of an alternative solution that does not require FIT files.
# 
# **I beleive having annotated videos like that above can be a great training aid.**
# 
# - With a 360 camera you can see the most of the sail, including the top, to understand sail settings. - You can pan and zoom the video so you can see most anything around the boat, including marks and competitors.
# - You have synchronize access to instrument data and crew actions.  This allows you to understand what caused changes in performance.
# - We could potentially overlay virtual "sensors", which can show VMG and speed versus polars.
# 
# If you search the web for applications that can produce videos with gauge overlays you find:
# 
# - [DashWare](http://www.dashware.net/) which is a flexible application that can produce videos with quality gauge overlays of many kinds (it claims to read *many* types of files).  
#   - It is windows only.  
#   - It can export video in "green screen" which can then be blended with other videos.
#   - The goals of Dashware are generic, but it seems to mostly support auto racing.
#   - Does not support 360 video directly (more on what this means below).
#   - Looks fairly old and perhaps lightly supported.  There is some mention that these folks moved to GoPro.
#   - Does not seem to read the FIT files.  Some mention of this on the web, but not much.
#   
# - Adobe After Effects seems to have the ability to import and overlay mgJSON files which define the gauges.  There is a company that will automatically process GoPro video, extract the GPS and other data, and then produce the mgJSON files [GoPro Telemetry Extractor](https://goprotelemetryextractor.com/).
#   - Commercial software (not a huge issue, but annoying).
#   - I am sure mgJSON is well documented somewhere.  But it will require some deep diving.
# 
# - Garmin Virb Edit
#   - Very nice end to end integration with Garmin systems (which generate FIT files).
#     - If a Garmin camera is paired with a Garmin marine "plotter" it will record instrument data in addition to video.  (In a sense this replaces the Raspberry PI.)
#     - This data is easily read by Virb Edit and can be overlayed with a set of standard gauges.
#   - It can read FIT files (which can store lots of data) or GPX files (which just store lat/lon/alt).
#   
# Note, one critical feature is time alignment. Frequently video does not have timestamps.  In this case you need to align the instrument data with the video.  Both DashWare and Virb Edit have good alignment tools (though it is done by hand).
# 
# 
# ## Quick note on the Virb 360
# 
# Garmin makes a 360 camera called the [Garmin Virb 360](https://buy.garmin.com/en-US/US/p/562010).  While it seems a bit dated compared with GoPro Max and more expensive, it has many advantages:
# 
# - It is a 360 camera.
# - It seems quite rugged.
# - It has a well designed external power system built into its tripod mount (ensuring I can record for 6+ hours).  My GoPro does **not** (nor does the newer GoPro Max).
# - It has an idiot proof mechanism for recording video (a big red mechanical switch).  No more pushing buttons and hoping that video is recorded.  I've lost videos on the GoPro because it is tricky to get started.
# 
# If your boat has Garmin electronics the overall end to end experience is pretty good: pair the camera with your plotter, record video, load into Virb Edit, and create videos with overlayed gauges.  It might be nice to include other data, but there is no obvious way to include information about VMG or polars. Checkout this [video from One Life](https://youtu.be/VXX5R1Jaxgg).
# 
# Originally I was hoping to create a FIT file from my boat instrument data and *sneak* it into Virb Edit.  But no luck.  More info here: [Garmin Fit Files](Garmin_Fit_files.ipynb).
# 
# And then I had a "better idea".

# ## Programmatic Video Editting
# 
# I currently have a somewhat automated scheme for processing boat instrument data (described elsewhere).  The corresponding video process is annoyingly manual (and very primitive).
# 
# - Start by anotating the Pandas data (race name, crew, conditions, trimming to race start/end).
# - Automatically clip the video to the race, add titles and initial annotation.
# - Automatically overlay instrument data in sync.
# - It would be even cooler if "critical" portions of the race were automatically collected.
#   - Each tack and jibe (perhpas 60 seconds centered?)
#   - Mark roundings.
#   - Places where we did particularly well or poorly?
# 
# Summary of required functionality:
# 
# - Where does the video come from?
#   - Raw: Straight from the Virb 360 (30 minute MP4 video blocks) with a singl FIT file.
#   - Processed: From Virb Edit, after merging and **stabilization** (see below, this is likely critical).
# - Extract the start time of the video
#   - Raw: The events denoting raw video capture times are in the FIT file.
#   - Processed: Virb Edit removes this info (can assume the start is the start of the first block).
# - Process, overlay and combine video.
#   - Moviepy??
# 

# # What is 360 Video?
# 
# Turns out that 360 video is must a video file (in our case mostly MP4 files) that have been stiched (or transformed) into a [equirectangular](https://en.wikipedia.org/wiki/Equirectangular_projection) format.  
# 
# Most 360 camera capture two fisheye images, and these are then stitched into a single image.  Note, both of these images are from [Paul Bourke's Website](http://paulbourke.net/dome/).
# 
# ![im](Data/Images/two_spherical.jpg)
# 
# ![im](Data/Images/equirectangular.jpg)
# 
# If the tags are set correctly on 360 videos (or images) then many viewers will automatically operate in 360 mode.  This mode remaps a small portion of the equirectangular image onto the screen of the user (being careful to minimize distortions).
# 
# The default mode for the Virb 360 is to capture the two spherical images and stitch them into a single equirectangular video.  There is also a raw mode that directly captures the spherical images at higher resolution (using more data, and ominously has warning about the camera overheating).
# 
# ## Tricky bits of 360 video
# 
# There are two things that are important for a good 360 experience:
# 
# - Video stabilization
#   - Action cameras can shake and this disturbs the video.
#   - There are two ways to stabilize: i) using gyros in the camera (Virb) and ii) tracking points in the video. 
#   - See [video](https://www.youtube.com/watch?v=uY1P2SrF0TQ) for how this can work on the Virb.
# - Straightening the horizon
#   - If the horizon is not level, then as the view pans left and right the visual motion include up and down motion *and* rotation.  Its weird.
#   - There are two ways to fix this: i) using acceleromters in the camera to measure the direction of gravity (Virb) and ii) by hand.
#   - See [video](https://youtu.be/GZYaGR6KRe8?t=23) for examples of how this can go wrong and how to fix it by hand.
# 
# This is an example of a rotated horizon (from the video above).
# 
# ![im](Data/Images/bad_horizon_360.png)
# 
# 
# 
# 
# ## References
# 
# - [360 Video Projection](https://en.wikipedia.org/wiki/360_video_projection)
# - [Cube mapping](https://en.wikipedia.org/wiki/Cube_mapping)
#   - Alternative to equirectangular (more uniform use of pixels)
# - [Google Equal Angle Cube Map](https://blog.google/products/google-vr/bringing-pixels-front-and-center-vr-video/)
#   - Perhpas better than cube map.
# 

# ## Tooling
# 
# #### Exiftool
# 
# > ExifTool is a platform-independent Perl library plus a command-line application for reading, writing and editing meta information in a wide variety of files. ExifTool supports many different metadata formats including EXIF, GPS, IPTC, XMP, JFIF, GeoTIFF, ICC Profile, Photoshop IRB, FlashPix, AFCP and ID3, as well as the maker notes of many digital cameras
# 
# [LINK](https://exiftool.org/)
# 
# - Important for very large files.
# 
#     exiftool -api largefilesupport=1 foo.mp4 
#     
# - To ensure 360 video is interpreted correctly
# 
# `exiftool -ProjectionType="equirectangular" -Spherical="true" baz.mp4 `
#     
# - The date *tags* file can be different from file date.
# 
#     exiftool V0130025.MP4  | grep -i date
# 
#     File Modification Date/Time     : 2020:04:09 21:41:24-07:00
#     File Access Date/Time           : 2020:04:10 22:04:06-07:00
#     File Inode Change Date/Time     : 2020:04:10 17:46:22-07:00
#     Create Date                     : 2020:04:09 21:41:39
#     Modify Date                     : 2020:04:09 21:41:39
#     Track Create Date               : 2020:04:09 21:41:39
#     Track Modify Date               : 2020:04:09 21:41:39
#     Media Create Date               : 2020:04:09 21:41:39
#     Media Modify Date               : 2020:04:09 21:41:39
#     
# #### Google Spatial Media Tools
# 
# [LINK](https://github.com/google/spatial-media)
# 
# These appear to do much the same as `exiftool` but have the (dis-)advantage of making a copy of the file **and** must be run in Python 2.7.
# 
#     [~/Sailboat/Packages/spatial-media]$ python spatialmedia -h
#     usage: spatialmedia [options] [files...]
# 
#     By default prints out spatial media metadata from specified files.
# 
#     positional arguments:
#       file                  input/output files
# 
#     optional arguments:
#       -h, --help            show this help message and exit
#       -i, --inject          injects spatial media metadata into the first file
#                             specified (.mp4 or .mov) and saves the result to the
#                             second file specified
# 
#     Spherical Video:
#       -s STEREO-MODE, --stereo STEREO-MODE
#                             stereo mode (none | top-bottom | left-right)
#       -c CROP, --crop CROP  crop region. Must specify 6 integers in the form of
#                             "w:h:f_w:f_h:x:y" where w=CroppedAreaImageWidthPixels
#                             h=CroppedAreaImageHeightPixels f_w=FullPanoWidthPixels
#                             f_h=FullPanoHeightPixels x=CroppedAreaLeftPixels
#                             y=CroppedAreaTopPixels
# 
#     Spatial Audio:
#       -a, --spatial-audio   spatial audio. First-order periphonic ambisonics with
#                             ACN channel ordering and SN3D normalization
# 
# 

# #### FFMPEG
# 
# 

# ### Programmatic Video

# 
# ### Video Overlays
# 
# In order to create awesome overlayed videos you need:  i) create a FIT file from the logs currently collected on the boat; ii) get that FIT file to load correctly into Virb Edit; iii) create widgets (gauges) which can display the required info.
# 
# There are two points of reference that have informed my design goals.
# 
# 
# #### Reference 2: [FlightData Manager](https://sites.google.com/site/pud2gpxkmlcsv/)
# 
# FDM is a very nice package, that includes a software tool to post-process drone data to create FIT files which can be loaded into Virb Edit and will produce custom overlays.  FDM also includes a set of custom gauges that display drone info very well.
# 
# ![im](Data/Images/flight_data_manager_example.jpg)
# 
# The author of FDM, Kenth Fuglsang Jensen, had many of the same goals that I have.  He collects video from his Parrot drone (which creates PUD files for telemetry).  He then needed to convert this to a FIT file.  Many of the fields in the drone data are missing or unsupported by Virb Edit (e.g. WIFI strength, and others).
# 
# Kenth did this in several steps.
# 
# - Figure out how to read the PUD files (we don't need this!).
# - Through trial and error finding which messages and fields are supported by Virb Edit and which are not.
# - Create new widgets (and templates) which can display this new information in useful ways.
# 
# **So I am building a FlightData Manger for sailboats.**

# # Random Stuff
# 
# 
# The Garmin Virb 360 camera captures videos in blocks (much like a GoPro). The blocks are 30 mins long (and about 18 GB).
# 
# There is often a single FIT file for the entire set of videos. I have found this both to be true for a continuous recording and when I recorded two videos in close succession (a minute apart).
# 
# The beginning and ending of each video is most likely encoded in this way:
# 
# Definition,10,camera_event,timestamp,1,,camera_file_uuid,128,,timestamp_ms,1,,camera_event_type,1,,camera_orientation,1
# 
# Data,10,camera_event,timestamp,"11",s,camera_file_uuid,"VIRBactioncamera360_Video_3840_2160_29.9700_3967827018_38f2ace9_1_13_2020-04-09-21-41-03.fit",,timestamp_ms,"257",ms,camera_event_type,"0",,camera_orientation,"0",
# 
# Data,10,camera_event,timestamp,"11",s,camera_file_uuid,"VIRBactioncamera360_Video_3840_2160_29.9700_3967827018_38f2ace9_1_13_2020-04-09-21-41-03.fit",,timestamp_ms,"257",ms,camera_event_type,"4",,camera_orientation,"0",
# 
# 
# Data,10,camera_event,timestamp,"22",s,camera_file_uuid,"VIRBactioncamera360_Video_3840_2160_29.9700_3967827018_38f2ace9_1_13_2020-04-09-21-41-03.fit",,timestamp_ms,"68",ms,camera_event_type,"2",,camera_orientation,"0",
# 
# Data,10,camera_event,timestamp,"22",s,camera_file_uuid,"VIRBactioncamera360_Video_3840_2160_29.9700_3967827018_38f2ace9_1_13_2020-04-09-21-41-03.fit",,timestamp_ms,"68",ms,camera_event_type,"6",,camera_orientation,"0",
# 
# Data,10,camera_event,timestamp,"28",s,camera_file_uuid,"VIRBactioncamera360_Video_3840_2160_29.9700_3967827018_38f2acfa_1_14_2020-04-09-21-41-03.fit",,timestamp_ms,"728",ms,camera_event_type,"0",,camera_orientation,"0",,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
# Data,10,camera_event,timestamp,"28",s,camera_file_uuid,"VIRBactioncamera360_Video_3840_2160_29.9700_3967827018_38f2acfa_1_14_2020-04-09-21-41-03.fit",,timestamp_ms,"728",ms,camera_event_type,"4",,camera_orientation,"0",,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
# Data,10,camera_event,timestamp,"42",s,camera_file_uuid,"VIRBactioncamera360_Video_3840_2160_29.9700_3967827018_38f2acfa_1_14_2020-04-09-21-41-03.fit",,timestamp_ms,"74",ms,camera_event_type,"2",,camera_orientation,"0",,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
# Data,10,camera_event,timestamp,"42",s,camera_file_uuid,"VIRBactioncamera360_Video_3840_2160_29.9700_3967827018_38f2acfa_1_14_2020-04-09-21-41-03.fit",,timestamp_ms,"74",ms,camera_event_type,"6",,camera_orientation,"0",,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
# 
# 

# In[1]:


# Load some libraries
get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})

import itertools as it
import pandas as pd
import numpy as np

# These are libraries written for RaceAnalysis
import global_variables
# We are using San Diego for these examples!
# G = global_variables.init_seattle()
G = global_variables.init_san_diego()
import race_logs
import process as p
import analysis as a
import chart as c

# This is the python-fitparse library
from fitparse import FitFile

ff = FitFile('Data/Virb360Fit/2020-04-09-21-41-03.fit')


# In[36]:


messages = it.islice(ff.get_messages('file_id'), 0, 10, None)

file_id = next(messages).get_values()

print(file_id)

dt = file_id['time_created']
print(dt)

print(dt.tzname())



# In[28]:



messages = list(it.islice(ff.get_messages('camera_event'), 0, 100, None))

dfs = []
rows = [m.get_values() for m in messages]
dfs.append(pd.DataFrame(rows))

df = dfs[0]

df
    


# In[15]:


dd


# In[16]:


help(dd)


# In[ ]:




