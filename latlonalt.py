"""
Handy routines for dealing with GPS.

Should put more here!

All functions, where possible, should be able to consume and produce both scalars and
numpy arrays.
"""

import numpy as np
import itertools as it

def degrees_to_dm(lat_or_lon):
    "Convert degrees to degrees and minutes (dm for short)."
    sign = np.sign(lat_or_lon)
    lat_or_lon = sign * lat_or_lon
    degrees = np.floor(lat_or_lon)
    minutes = 60 * (lat_or_lon - degrees)
    return np.int(sign * degrees), minutes

def degrees_to_dms(lat_or_lon):
    "Convert degrees to degrees, minutes, seconds (dms for short)."
    sign = np.sign(lat_or_lon)
    lat_or_lon = sign * lat_or_lon
    degrees = np.floor(lat_or_lon)
    minutes = 60 * (lat_or_lon - degrees)
    seconds = 60 * (minutes - np.floor(minutes))
    return np.int(sign * degrees), np.int(np.floor(minutes)), seconds

def dm_to_degrees(degrees, minutes):
    "Convert degrees and minutes to degrees."
    sign = np.sign(degrees)
    return degrees + sign * minutes/60.0

def dms_to_degrees(degrees, minutes, seconds):
    "Convert degrees, minutes, seconds to degrees."
    sign = np.sign(degrees)
    return degrees + sign * minutes/60.0 + sign * seconds/(60*60.0)

# Garmin FIT files pack lat (of lon) into signed 16 bits ints.  The unit is called the
# "semi-circle"
SEMICIRCLES_TO_DEGREES = 180/2**31

def semi_to_degrees(semicircles):
    "Convert the Garmin unit called semicircle to degrees."
    return SEMICIRCLES_TO_DEGREES * semicircles

def degrees_to_semi(degrees):
    "Degrees to the Garmin unit."
    np.int16(degrees / SEMICIRCLES_TO_DEGREES)

def iterable(val):
    "Small local helper. Ensures that val is iterable"
    if isinstance(val, np.ndarray):
        return val
    else:
        return [val]

class LatLonAlt(object):
    """
    Class that manages and converts information about lat/lon/alt locations.  In
    particular the class can convert from the convenient decimal degrees representation to
    degrees/minutes (aka dm) and degrees/minutes/seconds (aka dms).  Degrees is a
    singleton, dm is a tuple (degrees, minutes) and dms is a tuple (degrees, minutes,
    seconds).

    All functions have been written to handle NUMPY arrays as well as primitive numbers.
    """
    def __init__(self, lat=None, lon=None, alt=None, datetime=None):
        if lat is None or lon is None:
            raise Exception("We do not support LatLonAlt with empty lat or lon.")
        self.lat = lat
        self.lon = lon
        self.alt = alt if alt else np.full_like(lat, 0)
        self.datetime = datetime if datetime else np.empty_like(self.lat, dtype='datetime64[ns]') 

    @staticmethod
    def from_degrees_minutes(lat_dm=None, lon_dm=None, alt=None, datetime=None):
        if lat_dm is None or lon_dm is None:
            raise Exception("We do not support LatLonAlt with empty lat or lon.")
        return LatLonAlt(lat=dm_to_degrees(*lat_dm),
                         lon=dm_to_degrees(*lon_dm),
                         alt=alt,
                         datetime=datetime)

    @staticmethod
    def from_degrees_minutes_seconds(lat_dms=None, lon_dms=None, alt=None, datetime=None):
        if lat_dms is None or lon_dms is None:
            raise Exception("We do not support LatLonAlt with empty lat or lon.")
        return LatLonAlt(lat=dms_to_degrees(lat_dms),
                         lon=dms_to_degrees(lon_dms),
                         alt=alt,
                         datetime=datetime)

    def __repr__(self):
        return "\n".join(it.islice(self.d_string(), 0, 10))

    def __str__(self):
        return self.__repr__()

    def lat_dm(self):
        "Return latitude in tuple (degrees, minutes)."
        return degrees_to_dm(self.lat)

    def lat_dms(self):
        "Return latitude in tuple (degrees, minutes, seconds)"        
        return degrees_to_dms(self.lat)

    def lon_dm(self):
        "Return longitude in tuple (degrees, minutes)"        
        return degrees_to_dm(self.lon)

    def lon_dms(self):
        "Return longitude in tuple (degrees, minutes, seconds)"        
        return degrees_to_dms(self.lon)

    def dm_string(self):
        "String representation of lat/lon in degrees/minutes."
        for lat_dm, lon_dm in zip(iterable(degrees_to_dm(self.lat)),
                                  iterable(degrees_to_dm(self.lon))):
            if lon_dm[0] < 0:
                lon = -lon_dm[0]
                east_west = 'W'
            else:
                lon = lon_dm[0]                
                east_west = 'E'
            yield f"{lat_dm[0]} {lat_dm[1]:.6f}N {lon} {lon_dm[1]:.6f}{east_west}"

    def dms_string(self):
        "String representation of lat/lon in degrees/minutes/seconds."        
        for lat_dms, lon_dms in zip(iterable(degrees_to_dms(self.lat)),
                                    iterable(degrees_to_dms(self.lon))):
            if lon_dms[0] < 0:
                lon = -lon_dms[0]
                east_west = 'W'
            else:
                lon = lon_dms[0]                
                east_west = 'E'
            yield f"{lat_dms[0]:d} {lat_dms[1]:d} {lat_dms[2]}N {lon} {lon_dms[1]} {lon_dms[2]:.6f}{east_west}"

    def d_string(self):
        "String representation of lat/lon in decimal degrees."                
        for lat, lon in zip(iterable(self.lat), iterable(self.lon)):
            if lon < 0:
                lon = -lon
                east_west = 'W'
            else:
                east_west = 'E'
            yield f"{lat:.6f}N {lon:.6f}{east_west}"
