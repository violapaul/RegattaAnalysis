
import itertools as it
import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from fitparse import FitFile

pd.set_option('display.max_rows', 100)
pd.reset_option('precision')
pd.set_option('display.float_format', lambda x: '%.3f' % x)

np.set_printoptions(precision=4, suppress=True, linewidth = 180)


################################################################
class DictClass(dict):
    """
    Class that constructs like a dict, but acesss fields like a class.  Makes things much
    more compact.  So:
    foo = DictClass(one=1, two=2, bar=10)
    foo.bar (rather than foo['bar'])
            """
    def __init__(self, **args):
        for key in args.keys():
            self[key] = args[key]

    def __setattr__(self, attr, val):
        self[attr] = val
        
    def __getattr__(self, attr):
        return self[attr]

    def __str__(self):
        res = ""
        for k in sorted(self.keys()):
            v = self[k]
            if isinstance(v, float):
                res += f"{k}:{v:.2f}, "
            else:
                res += f"{k}:{v}, "
        return res


def degrees_to_dm(lat_or_lon):
    sign = np.sign(lat_or_lon)
    lat_or_lon = sign * lat_or_lon
    degrees = math.floor(lat_or_lon)
    minutes = 60 * (lat_or_lon - degrees)
    return sign * degrees, minutes

def degrees_to_dms(lat_or_lon):
    sign = np.sign(lat_or_lon)
    lat_or_lon = sign * lat_or_lon
    degrees = math.floor(lat_or_lon)
    minutes = 60 * (lat_or_lon - degrees)
    seconds = 60 * (minutes - math.floor(minutes))
    return sign * degrees, math.floor(minutes), seconds

def dm_to_degrees(degrees, minutes):
    sign = np.sign(degrees)
    return degrees + sign * minutes/60.0

def dms_to_degrees(degrees, minutes, seconds):
    sign = np.sign(degrees)
    return degrees + sign * minutes/60.0 + sign * seconds/(60*60.0)

SEMICIRCLES_TO_DEGREES = 180/2**31
def semi_to_degrees(semicircles):
    return SEMICIRCLES_TO_DEGREES * semicircles

def degrees_to_semi(degrees):
    return int(degrees / SEMICIRCLES_TO_DEGREES)


lat_dms = (47, 36, 21)
lon_dms = (-122, 24, 30.4)

lat_degrees = dms_to_degrees(*lat_dms)
lon_degrees = dms_to_degrees(*lon_dms)

lat_semi = degrees_to_semi(lat_degrees)
lon_semi = degrees_to_semi(lon_degrees)


def test_ll():
    slat = 567200491
    lat = semi_to_degrees(slat)
    lat_deg_min = degrees_to_degrees_minutes(lat)
    

def grab_until(elements, mesg_num, max=1000):
    while True:
        count = 0
        while count < max:
            mesg = next(elements)
            if mesg.mesg_num == mesg_num:
                return mesg
    return None


def skip_grab(num, elements):
    total = 0
    while True:
        count = 0
        while count < num:
            mesg = next(elements)
            if mesg.mesg_num == 20:
                count += 1
                total += 1
        dd = mesg.get_values()
        if 'position_lat' in dd:
            slat = dd['position_lat']
            slon = dd['position_long']
            if total % 100 == 0:
                print(total, math.fabs(lat_semi - slat), math.fabs(lon_semi - slon))
            if math.fabs(lat_semi - slat) < 200 and math.fabs(lon_semi - slon) < 200:
                return elements


def cache_elements(elements, count):
    return list(it.islice(elements, 0, count, 1))
        
def print_timestamps(elements, start=0, stop=None, skip=None):
    for e in it.islice(records, start, stop, skip):
        dd = e.get_values()
        if dd.has_key('timestamp'):
            print(dd['timestamp'])


def extract_fields(element):
    dd = element.get_values()
    u3 = dd['unknown_3']
    for i in range(20):
        dd[f"u3_{i:02d}"] = u3[i]
    del dd['unknown_3']
    return dd

def print_fields(element):
    dd = element.get_values()
    u3 = dd['unknown_3']

    values = []
    for i in range(10):
        values.append(u3[2*i] + 256 * u3[2*i + 1])

    print("Raw    ", end='')
    for i in range(10):
        print(f"    {values[i]:6d}", end='')
    print("")

    print("R/100  ", end='')
    for i in range(10):
        print(f"    {values[i]/100.0:4.2f}", end='')
    print("")


    print("R/1000 ", end='')
    for i in range(10):
        print(f"    {values[i]/1000.0:4.2f}", end='')
    print("")
    

    print("R m/s  ", end='')
    for i in range(10):
        v = 1.94 * values[i]/1000.0
        print(f"    {v:4.2f}", end='')
    print("")

def quick_df(elements):
    rows = [extract_fields(e) for e in elements if e.mesg_num == 176]
    return pd.DataFrame(rows)


def quick_df(elements):
    rows = [extract_fields(e) for e in elements if e.mesg_num == 176]
    return pd.DataFrame(rows)


for c in df.columns:
    print(c, df[c].unique())


file = '/Users/viola/Sailboat/Packages/FIT_SDK/java/2020-03-22-12-23-34.fit'
fitfile = FitFile(file)
elements = it.islice(fitfile.get_messages(), 0, None, None)
skip_grab(1, elements)



df = fitfile_to_pandas(fitfile.get_messages('record'), stop=100000)




def fitfile_to_pandas(records, start=0, stop=None, skip=None):
    rows = []
    # Get all data messages that are of type record
    for record in it.islice(records, start, stop, skip):
        # Go through all the data entries in this record
        row = {}
        for record_data in record:
            # Print the records name and value (and units if it has any)
            if record_data.units:
                name = record_data.name + "_" + record_data.units
            else:
                name = record_data.name
            row[name] = record_data.value
        rows.append(row)
        if (len(rows) % 1000) == 0:
            print(len(rows))
    df = pd.DataFrame(rows)
    df = df[df.isna().sum(1) == 0]
    df['longitude'] = semi_to_degrees(df.position_long_semicircles)
    df['latitude'] = semi_to_degrees(df.position_lat_semicircles)
    # df['row_times'] = df.timestamp.dt.tz_localize('UTC').dt.tz_convert('US/Pacific')
    return df

import chart as c

ch = c.plot_chart(df)
c.draw_track(df, ch)

def test():
    fitfile = FitFile('/Users/viola/canlogs/GARMIN_WATCH/4359549845.fit')
    fitfile = FitFile('/Users/viola/canlogs/GARMIN_WATCH/4324683973.fit')
    fit_df = fitfile_to_pandas(fitfile)
    chart.plot_track(fit_df)

