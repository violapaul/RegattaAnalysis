
import math
import numpy as np

def degrees_to_degrees_minutes(lat_or_lon):
    sign = np.sign(lat_or_lon)
    lat_or_lon = sign * lat_or_lon
    degrees = math.floor(lat_or_lon)
    minutes = 60 * (lat_or_lon - degrees)
    return sign * degrees, minutes

def degrees_minutes_to_degrees(degrees, minutes):
    sign = np.sign(degrees)
    return degrees + sign * minutes/60.0

