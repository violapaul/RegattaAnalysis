
#### Cell #0 Type: markdown ####################################################

#: # Speed Calibration
#: 
#: What is the relationship between boat speed (SPD) and GPS speed over ground (SOG).

#### Cell #1 Type: module ######################################################

# These are libraries written for RaceAnalysis
from global_variables import G
from nbutils import display_markdown, display
import race_logs
import metadata as m
import process as p
import analysis as a
import chart
import utils

import nbutils

# Initialize for Seattle.
G.init_seattle(logging_level="INFO")

import matplotlib.pyplot as plt

#### Cell #2 Type: module ######################################################

# Let's start out by looking at a particular day, when we were on motor and we performed a 
# maneuver, looping back over are course.

date = '2020-04-19'
df, race = race_logs.read_date(date)
ch = chart.trim_track(df)

# The interface below can be used to trim the track.  Notice the loop near 1800 North, and 680 West.

#### Cell #3 Type: module ######################################################

ch.begin, ch.end

#### Cell #4 Type: module ######################################################

import importlib
importlib.reload(chart)

#### Cell #5 Type: module ######################################################

# This loop can be trimmed out as follows:

ss = slice(52863, 55806)
tdf = df.iloc[ss]

_ = chart.plot_chart(tdf, border=0.5)


#### Cell #6 Type: module ######################################################

# And we can plot the boat's two measures of speed during that time.

chart.quick_plot(tdf.index, (tdf.sog, tdf.spd), "sog spd".split())

#### Cell #7 Type: markdown ####################################################

#: ### Not Equal
#: 
#: As we can see, the SOG does not equal SPD.  There are two main causes for this difference.
#: 
#: - Current:  The boat is moving through water which is itself moving.
#: - Calibration:  Speed is not correctly reported by the SPD sensor (a paddle wheel).
#: 
#: If we were sailing, which we were not, there would have been a third:
#: 
#: - Leeway: The tendency of the boat to slip sideway "to leeward" when sailing upwind.  This is not measured by the SPD sensor.
#: 
#: First observe that outside of the turns, the SPD is relatively constant.  We were moving through the water on motor with no changes in throttle.
#: 
#: The turns themselves are likely complex, and we should consider removing them.


#### Cell #8 Type: module ######################################################

# The turns can be easily seen in the COG and HDG.

# Note, HDG is magnetic.  And boat instruments report variation as a convenience. I believe it is simply a lookup in a table.  I do not believe the table is is particularly accurate and it is old, but its a start.
variation = tdf.variation.mean()

chart.quick_plot(tdf.index, (tdf.cog, tdf.hdg+variation), "cog hdg".split())

#### Cell #9 Type: module ######################################################

# There is a clear delay in the signals.  We can estimate that delay, as we do below.

delay, _ = a.find_signal_delay(tdf.cog, tdf.hdg+variation)
print(delay)

# note in similar analyses we have found about 16 samples of delay, or 1.6 seconds!

delay = 16
index = tdf.index[:-delay]
cog = tdf.iloc[delay:].cog
sog = tdf.iloc[delay:].sog
hdg = tdf.iloc[:-delay].hdg+variation
spd = tdf.iloc[:-delay].spd

chart.quick_plot(index, (cog, hdg))

#### Cell #10 Type: module #####################################################

# To simplify the analysis, we will decompose the boats motion north and east components.

vog_n = sog * p.north_d(cog)
vog_e = sog * p.east_d(cog)

btv_n = spd * p.north_d(hdg)
btv_e = spd * p.east_d(hdg)

chart.quick_plot(index, (vog_n, vog_e, btv_n, btv_e), "vog_n, vog_e, btv_n, btv_e")

#### Cell #11 Type: module #####################################################

# One simple way to estimate current is to take the average difference between the components.

cur_n = (vog_n - btv_n).mean()
cur_e = (vog_e - btv_e).mean()

print(cur_n, cur_e)

chart.quick_plot(index, (vog_n, vog_e, btv_n+cur_n, btv_e+cur_e), "vog_n, vog_e, btv_n, btv_e")

#### Cell #12 Type: markdown ###################################################

#: ### Improved, but far from perfect
#: 
#: We likely need to estimate both the current **and** the calibration.  
#: 
#: For the time being we will assume that the calibration is a single multiplier.  
#: 
#: Later, we can consider adding an offset, or perhaps some other non-linearities.  It is also possible that SPD is dependent on heel (reading higher on PORT or STBD tacks).

#### Cell #13 Type: module #####################################################

# We can directly optimize the current and multiplier in an effort to reduce the error (or mismatch) between the SOG/COG and SPD/HDG.

from scipy.optimize import minimize
import numpy as np

tdf = df.iloc[ss]
tdf = tdf[tdf.spd > 2.7]

def create_error_func(tdf):
    delay = 16
    index = tdf.index[:-delay]
    cog = tdf.iloc[delay:].cog
    sog = tdf.iloc[delay:].sog

    hdg = tdf.iloc[:-delay].hdg+variation
    spd = tdf.iloc[:-delay].spd

    vog_n = sog * p.north_d(cog)
    vog_e = sog * p.east_d(cog)

    btv_n = spd * p.north_d(hdg)
    btv_e = spd * p.east_d(hdg)

    def error(params):
        cur_n = params[0]
        cur_e = params[1]
        multiplier = params[2]
        # These are the residuals 
        r_n = vog_n - (multiplier * btv_n + cur_n)
        r_e = vog_e - (multiplier * btv_e + cur_e)
        return np.sum(np.square(r_n)) + np.sum(np.square(r_e))
    
    return error

error = create_error_func(tdf)
# Initial parameters
m0 = np.array([0, 0, 1.0])
print(error(m0))
res = minimize(error, m0, options=dict(disp=True))
                   # options={'xatol': 1e-5, 'disp': True})
    
params = res.x
cur_n = params[0]
cur_e = params[1]
multiplier = params[2]


#### Cell #14 Type: module #####################################################

print(params)
print(error(params))
print(error(m0))
print(cur_n, cur_e, multiplier)

#### Cell #15 Type: module #####################################################

chart.quick_plot(None, (vog_n, btv_n, multiplier*btv_n+cur_n, vog_e, btv_e, multiplier*btv_e+cur_e), "vog_n, btv_n, mbtv_n, vog_e, btv_e, mbtv_e")

#### Cell #16 Type: markdown ###################################################

#: ### Much better. North and East Residuals are Different?
#: 
#: Note the blue and green (north) curves are a much better match.  But the red/brown (east) are still off.
#: 
#: East/West speed seems to be underestimated, not sure why. The boat is mostly pointed north/south except in the turns.  Perhaps the boat is slipping sideways even under motor.

#### Cell #17 Type: module #####################################################

# We can re-run this analysis for the entire day's sail.  The problem is that current is almost certainly 
# not constant throughout the day.

# To re-emphasize the complexity in the process, let's look at a day when the wind was
# very light.

date = '2020-05-16'
df, race = race_logs.read_date(date)

chart.quick_plot(None, (df.spd, df.sog))

# We can clearly see that SPD can read zero for periods of time, most often when the true
# boatspeed is less than a 1 kt (or 0.5 m/s).

display(f"Mean speed {df.spd.mean()}")

# We still get basically similar results, though once again we assume constant current
# during the entire race.
error = create_error_func(df)
# Initial parameters
m0 = np.array([0, 0, 1.0])
print(error(m0))
res = minimize(error, m0, options=dict(disp=True))

params = res.x
print(params)
print(error(params))
print(error(m0))

#### Cell #19 Type: module #####################################################

# In this experiment, we introduce two multipliers: one for SPD < 1.0 m/s and a second for
# SPD greater.

def create_error_func2(tdf):
    delay = 16
    index = tdf.index[:-delay]
    cog = tdf.iloc[delay:].cog
    sog = tdf.iloc[delay:].sog

    hdg = tdf.iloc[:-delay].hdg+variation
    spd = tdf.iloc[:-delay].spd

    vog_n = sog * p.north_d(cog)
    vog_e = sog * p.east_d(cog)

    btv_n = spd * p.north_d(hdg)
    btv_e = spd * p.east_d(hdg)
    btv_switch = spd < 1.0
    print(btv_switch.sum(), (1-btv_switch).sum())

    def error(params):
        cur_n = params[0]
        cur_e = params[1]
        m0 = params[2]
        m1 = params[3]
        multiplier = (btv_switch * m0) + ((1-btv_switch) * m1)
        # These are the residuals 
        r_n = vog_n - (multiplier * btv_n + cur_n)
        r_e = vog_e - (multiplier * btv_e + cur_e)
        return np.sum(np.square(r_n)) + np.sum(np.square(r_e))
    
    return error


error = create_error_func2(df)
# Initial parameters
m0 = np.array([0, 0, 1.0, 1.0])
res = minimize(error, m0, options=dict(disp=True))

params = res.x
print(params)
print(error(params))
print(error(m0))

# Not surprisingly, the multiplier is higher for lo

#### Cell #20 Type: module #####################################################

# This is a good date, with both wind and light spots.
date = '2020-05-09'
df, race = race_logs.read_date(date)
display((race.title, race.date))

chart.plot_chart(df)
chart.quick_plot(None, (df.spd, df.sog))

# Assumptions
# current will be estimated every 60 seconds, or 600 samples.
# SOG/COG are the observables
# SPD/HDG are the inputs
# We will estimate
# - Current
# - SPD multiplier



#### Cell #21 Type: module #####################################################

btv_switch = (spd < 1.0)

#### Cell #22 Type: module #####################################################

(btv_switch * 1) + ((1-btv_switch) * 2)

#### Cell #23 Type: module #####################################################


# Estimating current from boat sensors is not unlikely estimating true wind, while it is
# simple in theory, it is complex in pactice.  The sensors involved a noisy and they
# require calibration (and that calibration may be more complex than a simple
# multiplier)..  There are physical phenomena, like leeway, which are not directly
# measured and can only be estimated.


#### Cell #24 Type: metadata ###################################################

#: {
#:   "metadata": {
#:     "kernelspec": {
#:       "display_name": "Python [conda env:sail] *",
#:       "language": "python",
#:       "name": "conda-env-sail-py"
#:     },
#:     "language_info": {
#:       "codemirror_mode": {
#:         "name": "ipython",
#:         "version": 3
#:       },
#:       "file_extension": ".py",
#:       "mimetype": "text/x-python",
#:       "name": "python",
#:       "nbconvert_exporter": "python",
#:       "pygments_lexer": "ipython3",
#:       "version": "3.7.0"
#:     },
#:     "timestamp": "2020-06-11T18:12:48.099720-07:00"
#:   },
#:   "nbformat": 4,
#:   "nbformat_minor": 2
#: }

#### Cell #25 Type: finish #####################################################

