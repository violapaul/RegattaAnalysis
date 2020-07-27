#!/usr/bin/env python
# coding: utf-8

# # Preparing Boat Instrument Data for Analysis
# 
# **Summary**: Notebook contains a discussion of the data which is captured on boat and how it is processed to make it accessible in Python.  
# 
# 
# ## The processing pipeline
# 
# There are lots of steps between data catpure and analysis.  As a summary, most steps are listed below.  
# 
# - Convert NMEA 2K to USB using [Actisense NGT-1](https://www.actisense.com/products/ngt-1-nmea-2000-to-pc-interface/)
# - Capture the raw log data on the RPi, using `actisense-serial`.
# - Copy the logs onto a USB memory stick.
# - Copy the memory stick logs off the RPi and onto my laptop.
# - Convert the raw logs with `analyzer` to JSON.
# - Read the JSON into Python, normalize, and convert.
# - Reorganize the data into a Pandas DataFrame (tabular format) that is useful for analysis.
# - Save the tables using Pandas.
# 
# ## Background 
# 
# Peer Gynt has a "modern" B&G set of instruments and displays (circa 2015).  These instruments all sit on a digital databus called NMEA 2000. NMEA 2000 (or 2K for short) is an adaptation of the CAN bus used on many/all cars today.    
# 
# The data on the bus is captured using a Raspberry Pi (RPi), running the [Canboat](https://github.com/canboat/canboat) suite of software. The raw bus data is captured and logged to disk with the Canboat command line program called [actisense-serial](https://github.com/canboat/canboat/wiki/actisense-serial).
# 
# ### Deep Background (Optional)
# 
# In addition to NMEA2K, there is an older standard called [NMEA 0183](https://en.wikipedia.org/wiki/NMEA_0183) which was based on a serial standard called RS-422.  0183 is slower, harder to hookup, and less standardized.  In comparision, 2K is easier to hookup: the connectors are standardized and the cables also carry (limited) power. In theory you can easily connect devices from different manufacturers.  You can buy a bridge from older 0183 devices to the newer 2K bus. 
# 
# The biggest problem with with 2k is that it is propietary.  The messages are sent in an industry standard format,  but to get access to that standard you need to buy a license, sign a lot of forms, and then swear to never redistributed the information.  Interestingly, 0183 is old enough that the standard is much more public (though it is also more limited).  The secret nature of 2k makes it harder to create new devices, and this keeps the prices up for devices made by the larger manufacturers.
# 
# Luckily, some folks sat down and created [Canboat](https://github.com/canboat/canboat).  This is free software that knows how to read 2k messages and convert them to readable JSON.  The source code is available, and it is quite simple.  The Canboat folks have never signed the abovementioned legal agreements, so all of their work is based on "reverse engineering". They are very upfront that while their efforts are perfectly legal, their approach may result in errors and mistranslations.
# 
# To run Canboat you need to plug a small computer into the 2K bus.  I use a Raspberry Pi 3 by CanaKit [Amazon Link] (https://www.amazon.com/dp/B01C6EQNNK/ref=cm_sw_em_r_mt_dp_U_-QvuEbN65D6RP) (there are many other ways to buy something similar).  You also need to buy an adapter from 2k to the RPi.  I use [NGT-1 NMEA 2000® to PC](
# https://www.actisense.com/products/ngt-1-nmea-2000-to-pc-interface/).  This particular adapter is explicitly supported by Canboat.  There may be other ways to go,  but I felt good using a high quality interface, since I did not want to corrupt data on the 2K bus while sailing.
# 
# Note, Canboat is also used by a much more complex and ambitious project called [SignalK](http://signalk.org/).  SignalK has the goal of letting people create their own marine devices and information displays.  Its all very cool, but very complex...  or rather it "easy" to set up, but then doing anything beyond what is built in can be a lot harder.  By using Canboat I've cut out the middle man.  
# 
# In the longer term SignalK may well be the right way to build smart algorithms that can run on the boat.  I'm not sure.
# 
# #### LInks
# 
# - [Basic NMEA 2K Video](https://www.youtube.com/watch?v=U4jAxINtF5w) if you have never heard of NMEA 2k.
# - [NMEA 2K whitepaper](https://www.nmea.org/Assets/20090423%20rtcm%20white%20paper%20nmea%202000.pdf)
# - [Actisense's Guide to NMEA 2K](https://www.actisense.com/wp-content/uploads/2020/01/Complete-Guide-to-Building-an-NMEA-2000-Network-issue-1.1.pdf)
# - [NMEA 2k wikipedia](https://en.wikipedia.org/wiki/NMEA_2000)
# 
# ## Design Tenets
# 
# Before I describe the process used to collect and process boat instrument data, I will lay out the basic design tenet.  Every design should have tenets.  These are high level requirements that potentially impact a large portion of the design.
# 
# - Keep the original raw data. If there is a bug in the processing pipeline then the data can be re-processed.
#   - Keep at least one copy of all data.
# - Process all data, but flag useful vs useless.
# - Only copy changes, to make things fast
# - It should be easy to navigate data
#   - Filenames should make sense and be organized
#   - Additional Databases that cannot be easily hand viewed and edited are bad.
# - The Raspberry Pi compute is ephemeral.  Assume it can be lost!
# 
# 
# ### Creating the log files.
# 
# When the RPi is booted up, a "service" is started using [systemclt](https://www.digitalocean.com/community/tutorials/how-to-use-systemctl-to-manage-systemd-services-and-units) which reads data from actisense device and writes a log file and a paired `err` file which captures all errors (if the RPi is sitting on your desk at home it will create empty log files with an associated large set of errors).  The code for this **simple** service is in` RaspberryPi/Canboat`, the logs are written to the `logs` directory on the RPi.
# 
# Currently the log files are given names like: `actisense_2020_03_07_06_17_05_27362.log`  Which contains the "datetime" (`year_month_day_hour_minute_second`) and an additional random number (27362 in this case).  
# 
# Note, the RPi does not contain a battery backed clock, so the **datetime is not correct**.  This has lots of disadvantages: the RPi does not advance the clock when it is off *and* the clock is not very accurate, loosing seconds per day.  The clock is generally "monotonic" (it should only increase) and the clock is automatically corrected when the RPi can connect to the internet (so there are big jumps when I boot it up at home).  The fake datetime plus the short random number ensures that the filename is unique, even if the clock goes crazy.
# 
# ### Log file contents
# 
# The log files contain raw canbus data, which is not yet interpreted.  We never actually look at these files, but here is a sample of 8 lines (documentation for this format is [here](https://github.com/canboat/canboat/wiki/analyzer)).
# 
# Format: `timestamp, priority, message type (pgn), source, destination, number of data bytes, hexadecimal bytes`
# 
# ```
# 2020-03-07T02:17:07.216Z,6,129539,127,255,8,00,d3,46,00,5a,00,ff,7f
# 2020-03-07T02:17:07.216Z,7,127258,127,255,8,00,f6,ff,ff,5a,0a,ff,ff
# 2020-03-07T02:17:07.216Z,3,127257,128,255,8,00,ff,7f,99,fe,83,fd,ff
# 2020-03-07T02:17:07.216Z,2,130306,14,255,8,00,20,01,c3,79,fa,ff,ff
# 2020-03-07T02:17:07.216Z,2,127245,15,255,8,ff,ff,ff,7f,13,05,ff,ff
# 2020-03-07T02:17:07.216Z,2,127251,4,255,8,ff,c7,a7,00,00,ff,ff,ff
# 2020-03-07T02:17:07.216Z,2,127245,3,255,8,ff,ff,ff,7f,ff,7f,ff,ff
# 2020-03-07T02:17:07.216Z,3,129799,8,255,19,00,42,ef,00,00,42,ef,00,31,36,00,00,00,00,19,00,00,00,00
# ```
# 
# Note, the underlying [canbus](https://en.wikipedia.org/wiki/CAN_bus) sends packets which are only 8 bytes long.  One of the log messages above is longer, and this is accomplished on the 2K bus using multiple packets.  Additionally canbus packets include a checksum (which can be used to detect corrupted packets).  Corrupt packets are discarded and do not make it into the log.  
# 
# Each and every device on the bus sends data packets, some send more than 10 packets a second.  We see around 200 packets a second during normal operations.  I have some evidence (from Windows software that comes with the Actisense) that the bus is busy (perhaps using 20-30% of available capacity).
# 
# At 200 messages a second, the logs are large: 1200 messages a minute, or about a Megabyte a minute.  **A 5 hour sail results in 300M file.**  The log files can be compressed by 8-10x using `gzip`.
# 
# ### Getting the Log files off the RPi
# 
# Another `systemctl` service is constantly looking for the presence of a USB memory stick (`RaspberryPi/UsbCopy`).  If it is found, then the contents of the logs directory are copied from the RPi into the `logs` directory of the memory stick.  The `rsync` program is used both to avoid repeatedly copying the same file and to ensure that these are pefect copies.
# 
# At this point there are two copies of each log file (RPi and USB).

# ### Annoyances
# 
# At this point let me list a few annoyances that must be addressed:
# 
# - The log files are large.
# - The log files have strange names unrelated to the actual date acquired.
# - There are a number of smaller, or empty, files that have been captured during false starts.
# 
# 
# 
# 
# 
# 
# After the race I copy the raw log off of the RPi and then convert it to JSON using another Canboat program called [analyzer](https://github.com/canboat/canboat/wiki/analyzer).  This JSON file is then read into Pyhton/Pandas for analysis.
# 
# I do not currently compute on the boat during a race.  That is for later.

# In[ ]:


# Data in NMEA 2k is sent in PGN's.  

PGNS = pd.DataFrame([
    (126992, 'System Time'),
    (127245, 'Rudder'),
    (127250, 'Vessel Heading'),
    (127251, 'Rate of Turn'),
    (127257, 'Attitude (pitch, roll)'),
    (127258, 'Magnetic Variation'),
    (128259, 'Speed'),
    (128267, 'Water Depth'),
    (129025, 'Position, Rapid Update'),
    (129026, 'COG & SOG, Rapid Update'),
    (129029, 'GNSS Position Data'),
    (129539, 'GNSS DOPs'),
    (129540, 'GNSS Sats in View'),
    (130306, 'Wind Data')
], columns=['id', 'Description'])

PGN_WHITELIST = PGNS.loc[:, 'id'].tolist()

