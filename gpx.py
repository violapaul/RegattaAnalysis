"""
Handy routines for dealing with GPS.

Should put more here!

All functions, where possible, should be able to consume and produce both scalars and
numpy arrays.
"""

import math
import numpy as np

def degrees_to_dm(lat_or_lon):
    "Convert degrees to degrees and minutes (dm for short)."
    sign = np.sign(lat_or_lon)
    lat_or_lon = sign * lat_or_lon
    degrees = np.floor(lat_or_lon)
    minutes = 60 * (lat_or_lon - degrees)
    return sign * degrees, minutes

def degrees_to_dms(lat_or_lon):
    "Convert degrees to degrees, minutes, seconds (dms for short)."
    sign = np.sign(lat_or_lon)
    lat_or_lon = sign * lat_or_lon
    degrees = np.floor(lat_or_lon)
    minutes = 60 * (lat_or_lon - degrees)
    seconds = 60 * (minutes - math.floor(minutes))
    return sign * degrees, math.floor(minutes), seconds

def dm_to_degrees(degrees, minutes):
    "Convert degrees and minutes to degrees."
    sign = np.sign(degrees)
    return degrees + sign * minutes/60.0

def dms_to_degrees(degrees, minutes, seconds):
    "Convert degrees, minutes, seconds to degrees."
    sign = np.sign(degrees)
    return degrees + sign * minutes/60.0 + sign * seconds/(60*60.0)


# FIT packs lat and long into signed 16 bits ints.  The unit is called the "semi-circle"
SEMICIRCLES_TO_DEGREES = 180/2**31

def semi_to_degrees(semicircles):
    "Convert the Garmin unit called semicircle to degrees."
    return SEMICIRCLES_TO_DEGREES * semicircles

def degrees_to_semi(degrees):
    "Degrees to the Garmin unit."
    np.int16(degrees / SEMICIRCLES_TO_DEGREES)
