import matplotlib.pyplot as plt
import numpy as np

import process as p
import analysis as a
import chart as c
import global_variables as G
from utils import DictClass

from numba import jit


# In[2]:


sail_logs = [
    DictClass(log='2019-11-16_10:09.pd', doc='Snowbird #1.', begin=41076, end=111668, twd=True),
    DictClass(log='2019-11-23_10:23.pd', doc='Practice.', twd=True),
    DictClass(log='2019-11-24_10:33.pd', doc='Practice.', twd=True),
    DictClass(log='2019-12-07_09:47.pd', doc='Snowbird #2.', begin=54316, end=109378, twd=True)
]

example = sail_logs[0]

print(f"Reading and analyzing {example.log}")
df = p.read_sail_log(example.log, discard_columns=True, skip_dock_only=False, trim=True, path=G.DATA_DIRECTORY, cutoff=0.3)


# In[3]:


# Snowbird #1 was a day when we battled with Creative on the upwind from the last
# mark to the finish
chart = c.plot_chart(df)
race_df = df.loc[example.begin : example.end]
c.draw_track(race_df, chart, color='green')


# In[4]:


# Let's just look at a bit of data, 100 seconds from the middle, centered on a tack.
seconds = 100
samples_per_second = 10
delay = seconds * samples_per_second
start = 99000

wide_slice = slice(start, start + delay)
wdf = df.loc[wide_slice]
plt.figure()
# RAWA is the raw values of AWA that are logged
wdf.rawa.plot()


# In[5]:


# Let's drill down and look at 5 seconds (50 samples)
narrow_slice = slice(start, start + 5 * 10)
ndf = df.loc[narrow_slice]
plt.figure()
ndf.rawa.plot()


# ## Noise, Signal, Damping, and Filtering
# 
# Noise is pure randomness, there is no structure and no clear physical explanation.
# 
# But in the above signal there is clear evidence of rapid oscillation.  Most likely this is **not** noise.  Remember that the wind sensor is at the top of the mast, 50 feet above the water, where small changes in boat heel can lead to large and rapid movement side to side.  As the boat cuts through rhythmic waves, it is plausible that regular oscilation in wind angle would result.
# 
# But is the useful information?  Probably not.  Neither the boat, nor the wind, is rotating 9 degrees in 0.5 seconds.  So these variations might as well be noise, and we are better of "filtering" them away.
# 
# The true AWA is hidden in this noise; if you were to display the raw wind sensor readings the numbers would be jumping around like crazy, almost impossible to read.  (On many boats you can adjust display settings to display the raw instrument readings directly.  Its not very useful.)
# 
# On our B&G boat, these issues are addressed by "damping", which is left a bit mysterious in the Triton and Zeus B&G documentation.  Damping is a number from 0 to 10, with higher numbers providing more filtering and less noise, while also introducing significant delay.  Further information on signal processing is provided in the documentation for some of the more advanced B&G sail computers: H5000 and WTP3.  Most likely the onboard units are using exponential filtering, perhaps a non-linear version.
# 
# Note, that what is logged for AWA is not what is shown on the display, because mast head sensor unit broadcasts raw AWA; the displays compute a filtered/damped version, which is not logged.
# 
# The logged values for true wind angle (TWA) are damped. On Peer Gynt, TWA (and TWS and TWD) are computed and broadcast by the autopilot computer (if the autopilot is off, then they missing from the logs).  These signals are damped **before** they are sent to other devices, including our logging device.
# 
# Since TWA is closely related to AWA, let's plot them both to get a sense of how damping works.

# In[6]:


plt.figure()
# RAWA is the raw values of AWA that are logged
wdf.rawa.plot()
# And the "raw" TWA that is logged is quite different and much smoother.
wdf.rtwa.plot()


# For more information on sailing instruments the book *Sail Smart: Understand Your Instruments to Sail Faster, Make the Right Calls & Win Races* by Mark Chisnell provides quite a bit of helpful info on boat instruments and measurements. [link](https://www.amazon.com/dp/B07F3CKRLH/ref=cm_sw_em_r_mt_dp_U_Px-gEbZJ52RTX).  The book is short, and for some it will repeat a lot of the basics, but it is helpful none the less.
# 
# Apparent wind conflates the "true wind" and the boat movement.  If the boat were stationary (e.g. at the dock) the apparent wind and the true wind are the same (both in speed and direction).  But if the boat is moving it introduces another "wind" which is equal and opposite to the boat's speed and direction.  True wind is an attempt to subtract the boat's contribution.
# 
# Note, AWA, AWS, and SPD are directly measured by sensor units on the boat.  True wind must be computed.
# 
# True wind is more consistent and drives the strategy that you use to race around the course.  Note TWA is reported relative to the boat, and TWD is reported relative to true north. We will see that they are otherwise the same.
# 
# We can compute the TWA from AWA/AWS/SPD using the code below.  My favorite way to do this is to separate the two components of the apparent wind: i) the wind along the boat (which I think of as "boat north" pointing to the BOW) and ii) the wind perpedicular to the boat, "boat east" or PORT.  The SPD is then subtracted from local north (since the boat is assumed to be moving forward in the local north coordinates). All of this can be transformed into global north by adding the boats heading (which we will not do here).  Note, heading (HDG) is typically given in magnetic coordinates, not world, so be careful to add in the magnetic variation.

# In[7]:


# Note, TWA is computed from AWA, AWS, and boat speed (SPD)
#
# This is "local" north, not global
tw_north = p.cos_d(df.rawa) * df.raws - df.rspd
tw_east = p.sin_d(df.rawa) * df.raws
df['predicted_twa'] = np.degrees(np.arctan2(tw_east, tw_north))
wdf = df.loc[wide_slice]
c.quick_plot(wdf.index, (wdf.rawa, wdf.predicted_twa, wdf.rtwa), "rawa predicted_twa rtwa".split())


# There is general agreement between boat computed TWA and the "predicted TWA" using our formulas.  But there is a huge difference in noise.  The boat TWA is clearly filtered to remove noise.  
# 
# I am very impressed (or perhaps surprised) that the TWA computed on the boat seems to be **ahead** of the predicted TWA at the rise. Smoothing always introduced some delay. (More below. The TWA actually seems to move before AWA, which and seems to line up with heading.)
# 
# How is smoothing computed?
# 
# Is the smoothing done to AWA, AWS, and SPD before combination, or only after?
# 
# I started out by implementing a number of classical smoothing filters, my favorite being the [Butterworth](https://en.wikipedia.org/wiki/Butterworth_filter).  The Butterworth is a [causal](https://en.wikipedia.org/wiki/Causal_filter) optimal linear filter, but there is no point where a single filter can both smooth out the local noise and track the large jumps.  Butterworth also introduces quite a bit of delay.

# In[8]:


coeff = p.butterworth_filter(cutoff=0.3, order=5)

df['butter_causal_awa'] = p.smooth_angle(coeff, df.rawa, causal=True)
wdf = df.loc[wide_slice]

c.quick_plot(wdf.index, (wdf.rawa, wdf.butter_causal_awa, wdf.rtwa), "awa sawa rtwa".split())


# Notice that the Butterworth signal is both more wiggly **and** very delayed (perhaps 4 seconds). We can reduce the noise (remove wiggles) but that increases delay.  Delay can be reduced, but the noise grows in response.

# In[10]:


coeff = p.butterworth_filter(cutoff=0.7, order=5)

df['butter_causal_awa'] = p.smooth_angle(coeff, df.rawa, causal=True)
wdf = df.loc[wide_slice]

c.quick_plot(wdf.index, (wdf.rawa, wdf.butter_causal_awa, wdf.rtwa), "awa sawa twa".split())


# To eliminate the delay there is a non-causal trick one can use, which is to first filter forward in time and then backward.  But this **cannot** be what is used on the boat in real-time.  My motivation is to understand what the boat does, and to create algorithms that might be used on the boat.
# 
# After digging through various writings on boat instrument processing I discovered an offhand comment regarding non-linear filtering.  The basic idea is to implement a non-linear version of a simple recursive filter 
# 
# $$ o_{t} = \alpha o_{t-1} + (1 - \alpha) i_t $$
# 
# where $o_t$ is the output and $i_t$ is the input (this is often called the *exponential filter*).  Larger values of $\alpha$ (up to 1.0) yield more smoothing.  
# 
# This sort of filter is very easy to implement, even on a tiny old computer,  but it is also primitive, providing poor filtering of noise.  If you want smoother filtering, then the signal is delayed (perhaps worse than a Butterworth).  Below is the basic recursive filter.

# In[12]:


# Ignore the mysterious parameter of 10,000 for now.  Explained below.
df['sawa_9'], _ = p.exponential_filter_angle(np.array(df.rawa), 0.9, 10000)
alpha = 0.97
df['sawa_alpha'], _ = p.exponential_filter_angle(np.array(df.rawa), alpha, 10000)
wdf = df.loc[wide_slice]
c.quick_plot(wdf.index, (wdf.rawa, wdf.sawa_9, wdf.sawa_alpha, wdf.rtwa), "awa sawa_9 sawa_97 twa".split())


# Above we see two runs of the exponential filter; higher alpha yields a smoother result.  TWA is included to show an "appropriate" amount of smoothing.
# 
# The large value of alpha is pretty good for filtering, but it introduces a **huge** delay!  The smaller value tracks more closely (with shorter delay), but it is too wiggly **AND** delayed.
# 
# Like the Butterworth, the exponential filter is linear.  The non-linear version introduces a tracking error threshold.  
# - If the difference between the filtered output and the raw input is larger than threshold then dynamically reduce the value of alpha.  
# - If the tracking error is less than threshold than dynamically increase the threshold (gradually back to original).
# 
# There are a couple of magic numbers here, like how much to decrease or increase the value of alpha (which are hard to set).  Code is below.  Note, I could not figure out a good way to do this with Numpy magic (which makes things fast in Python).  The result is a very slow Python loop.  I sped this up with the [Numba](http://numba.pydata.org/) package for compiling Python functions (see the `jit` decorator).
# 
# The code for filtering angles is a bit more complex (angles wrap around at 360).  Below is the simpler (non-angle) version.
# 
# Why is this a good filter?  All filters are trying to separate the noise from the underlying signal. It is plausible that the underlying signal is mostly changing slowly (between tacks), with additional large steps (at tacks).  This is the classic square wave.  If the filter had advanced knowledge of the regime (slow or step) then you could use two values of alpha, say 0.99 for the smooth regions and 0.5 for the steps.  The non-linear exponential filter moves between the regimes by looking at the output of the filter.  If the tracking is poor then the signal must be in the step regime, so alpha is reduced.  If tracking is good then alpha is increased.
# 
# The Butterworth is stuck processing both regimes using the same parameters.
# 
# Note, AWA looks a lot like a square wave.  Tacks and gybes introduce big changes, but otherwise you sail on a consistent wind angle.  Tacks and gybes also introduce changes in AWS and SPD as well, though less dramatically.
# 
# I suppose this sort of processing could be justified by a two state hidden markov model (HMM) which could work along similar lines.  

# In[13]:


@jit(nopython=True)
def exponential_filter(sig, alpha, max_error):
    """
    Apply a non-linear exponential filter, where alpha is the decay (higher alpha is longer decay and higher smoothing).  If the error is
    greater than max_error, then alpha is repeatedly reduced  (faster decay).
    """
    beta = beta0 = 1 - alpha
    res = np.zeros(sig.shape)
    betas = np.zeros(sig.shape)
    res[0] = sig[0]
    betas[0] = beta
    for i in range(1, len(sig)):
        res[i] = (1 - beta) * res[i-1] + beta * sig[i]
        if np.abs(res[i] - sig[i]) > max_error:
            beta = min(1.5 * beta, 0.5)
        else:
            beta = (beta + beta0) / 2
        betas[i] = beta
    return res, betas


# In[15]:


# Now the last parameter can be defined, it is the max error.  This is max_tracking_error threshold.
theta = 6  # the threshold, which should be approximate the noise 
alpha = 0.97
df['sawa_97'], _ = p.exponential_filter_angle(np.array(df.rawa), alpha, 10000)
df['sawa_alpha_theta'], _ = p.exponential_filter_angle(np.array(df.rawa), alpha, theta)

wdf = df.loc[wide_slice]
diff = wdf.sawa_alpha_theta - wdf.sawa_alpha

c.quick_plot(wdf.index, (wdf.rawa, wdf.sawa_alpha, wdf.sawa_alpha_theta, wdf.rtwa, diff), 
                        "awa sawa_alpha sawa_alpha_theta twa diff".split())
# Second plot is a bit cleaner because it leaves out the RAWA
c.quick_plot(wdf.index, (wdf.sawa_alpha, wdf.sawa_alpha_theta, wdf.rtwa, diff), 
                        "sawa_alpha sawa_alpha_theta twa diff".split())


# In the above figure DIFF shows the difference between the lineary and non-linear exponential filter. We can see that THETA only plays a role in the regions of the tack (where a rapid change in AWA is triggered).  In regions on consistent sailing, there is no difference between the two signals.
# 
# **How is theta determined?** In this case by eye.  After selecting a threshold, like 6, you can examine the tracking error to see if the signal strays far from the raw.

# In[16]:


# Now the last parameter can be defined, it is the max error.  This is max_tracking_error threshold.
theta = 6  # the threshold, which should be approximate the noise 
alpha = 0.97
df['sawa_alpha_theta'], _ = p.exponential_filter_angle(np.array(df.rawa), alpha, theta)

wdf = df.loc[wide_slice]
diff = wdf.sawa_alpha_theta - wdf.sawa_alpha

err = wdf.rawa - wdf.sawa_alpha_theta
c.quick_plot(wdf.index, (wdf.rawa, wdf.sawa_alpha_theta, err), 
                        "rawa sawa err".split())

print(f"The ERR mean {np.mean(err):.3f} std {np.std(err):.3f}")


# We can see that the ERR signal looks a lot like random noise (or perhaps it is the noise introduced by the motion of the mast).  And the noise has a magnitude of about of about 6.  

# In[17]:


# Note different signals require different settings of alpha, because they have different 
# dynamics and noise.
theta = 6  
alpha = 0.97
df['sawa'], _ = p.exponential_filter_angle(np.array(df.rawa), alpha, theta)

# Less noise and smaller theta values.
df['saws'], _ = p.exponential_filter(np.array(df.raws), 0.93, 0.5)
df['sspd'], _ = p.exponential_filter(np.array(df.rspd), 0.8, 0.5)

wdf = df.loc[wide_slice]

c.quick_plot(wdf.index, (wdf.rawa, wdf.sawa, wdf.twa), "rawa, sawa twa".split())
c.quick_plot(wdf.index, (wdf.raws, wdf.saws, wdf.rspd, wdf.sspd), "raws saws rspd ssped".split())


# Let's return to the calculation of TWA/TWS from raw instrument readings.
# 
# We have three options for computing a smoother version:
# 
# 1. Smooth the TWA after calculation from raw signals.
# 1. Compute TWA from the smoothed signals.
# 1. Do both.

# In[18]:


# First compute the baseline TWA
tw_north = p.cos_d(df.rawa) * df.raws - df.rspd
tw_east = p.sin_d(df.rawa) * df.raws
df['predicted_twa'] = np.degrees(np.arctan2(tw_east, tw_north))

# And then TWA from smoothed signals
tw_north = p.cos_d(df.sawa) * df.saws - df.sspd
tw_east = p.sin_d(df.sawa) * df.saws
df['predicted_stwa'] = np.degrees(np.arctan2(tw_east, tw_north))

# And then the smoothed TWA from the baseline
df['spredicted_twa'], _  = p.exponential_filter_angle(np.array(df.predicted_twa), alpha, theta)

# And then both
df['spredicted_stwa'], _  = p.exponential_filter_angle(np.array(df.predicted_stwa), alpha, theta)


# In[40]:


# Narrow focus to a set of tacks... keeps things simple
tacks_slice = slice(85000, 110000)
wdf = df.loc[tacks_slice]
c.quick_plot(wdf.index, (wdf.rawa, wdf.sawa, wdf.rtwa, wdf.predicted_stwa, wdf.spredicted_twa, wdf.spredicted_stwa),   
            "rawa sawa rtwa p_stwa sp_twa sp_stwa".split())
c.quick_plot(wdf.index, (wdf.rtwa, wdf.predicted_stwa, wdf.spredicted_twa, wdf.spredicted_stwa),   
            "rtwa p_stwa sp_twa sp_stwa".split())


# In[58]:


# Let's drill down into one tack.
wdf = df.loc[wide_slice]

c.quick_plot(wdf.index, (wdf.rudder, wdf.sawa, wdf.rtwa, wdf.spredicted_stwa, wdf.hdg-175),   
            "rudder sawa rtwa sp_stwa hdg".split())

# Note that TWA seems to be **leading** AWA!
# Note how TWA (from the boat) and HDG have very similar shapes (inverted)!


# The figure above is very interesting.
# 
# We've added RUDDER and HDG to help identify the beginning of the tack (in a sense the tack cannot begin **before** the rudder is turned).  As we expect, after the rudder is turned the HDG responds almost immediately (less than a second). 
# 
# But AWA seems to lag the change in HDG.  In a simple model this makes no sense.  Forget, for a moment, that this is a sailboat.  Instead imagine this is a wind sensor on a moving platform.  If you were to rotate the wind sensor toward the wind,  then the AWA should decrease (so should the TWA).  
# 
# But I do not believe this is an artifact.
# 
# **MY conjecture is that a change in HEEL introduces apparent wind on the mast head.**  As the boat rolls from starboard heel to port heel, the mast head **moves to windward**.  Motion to windward should increase the AWA.
# 
# We can investigate this by first smoothing heel (though this time with a non-causal Butterworth for now), and then differentiating (to get roll rate). Finally we can multiply by the mast height to get the velocity at the mast head.
# 
# Note, this boat does **not** have a high quality, high frequency HEEL sensor.  Currently the external GPS has both a compass, and a roll/pitch sensor.  Roll is heel,  but it is only report 1x a second (which may not be sufficient for this analysis).

# In[37]:


# Compute a smooth version of heel.  Note the cutoff is set to 0.3 Hz.  Since the signal is sampled at 1 Hz 
# Nyquist says we cannot hope measure any signal higher than 0.5 Hz.
coeff = p.butterworth_filter(cutoff=0.3, order=5)
# Note causal is false, which will filter the signal but not introduce any delay.
sheel = p.smooth_angle(coeff, wdf.zg100_roll, causal=False)

# It is not straightforward to apply the exponential filter defined above.  The signal is reported at 10Hz, but 
# the measure is at 1Hz.  This introduce at artifact that the signal appears to be constructed from step edges.
sheel2, _ = p.exponential_filter_angle(np.array(wdf.zg100_roll), 0.9, 10000)

# Compute the degrees per second rate of change of heel
sample_rate = 10
heel_rate = sample_rate * np.diff(sheel, prepend=sheel[0])

# Convert to masthead velocity
feet_per_meters = 0.3048
mast_height = 50 * feet_per_meters
mast_vel = mast_height * np.radians(heel_rate)



c.quick_plot(wdf.index, (wdf.zg100_roll, sheel, sheel2, heel_rate, mast_vel), 
            "heel sheel sheel2 heel_rate mast_vel".split())


# Interestingly we can clearly see a double peak in the roll rate centered arount the tack.  While this does not happen in every case it is not uncommon.  In a slow tack, as the boat passes head to wind there is a period where the heel is constant.

# In[38]:


# Break apparent wind into the component aligned with the boat "north" and the component aligned with roll, "east".
aw_n = wdf.raws * np.cos(np.radians(wdf.rawa))
aw_e = wdf.raws * np.sin(np.radians(wdf.rawa))

correction = mast_vel

# Corrected 
caw_e = aw_e - correction

c.quick_plot(wdf.index, (aw_n, aw_e, mast_vel, caw_e, 0.1*(wdf.hdg-170), wdf.rudder, sheel, wdf.zg100_roll), 
            "awn awe mast_vel corrected hdg rud sheel heel".split())

cawa = np.degrees(np.arctan2(caw_e, aw_n))
caws = np.sqrt(np.square(aw_n) + np.square(caw_e))

scawa, _ = p.exponential_filter_angle(np.array(cawa), alpha, theta)
c.quick_plot(wdf.index, (cawa, scawa, wdf.rawa, wdf.sawa),
                        "cawa scawa rawa sawa".split())


# The corrected AWA now **leads** raw AWA by over a second, and there are fewer artifacts.

# In[54]:


# Repeat for the entire signal


# Compute a smooth version of heel.  Note the cutoff is set to 0.3 Hz.  Since the signal is sampled at 1 Hz 
# Nyquist says we cannot hope measure any signal higher than 0.5 Hz.
coeff = p.butterworth_filter(cutoff=0.3, order=5)
# Note causal is false, which will filter the signal but not introduce any delay.
sheel = p.smooth_angle(coeff, df.zg100_roll, causal=False)

# Compute the degrees per second rate of change of heel
sample_rate = 10
heel_rate = sample_rate * np.diff(sheel, prepend=sheel[0])

# Convert to masthead velocity
feet_per_meters = 0.3048
mast_height = 50 * feet_per_meters
mast_vel = mast_height * np.radians(heel_rate)

# Break apparent wind into the component aligned with the boat "north" and the component aligned with roll, "east".
aw_n = df.raws * np.cos(np.radians(df.rawa))
aw_e = df.raws * np.sin(np.radians(df.rawa))

# Corrected 
caw_e = aw_e - mast_vel

df['cawa'] = np.degrees(np.arctan2(caw_e, aw_n))
df['caws'] = np.sqrt(np.square(aw_n) + np.square(caw_e))

df['scawa'], _ = p.exponential_filter_angle(np.array(df.cawa), alpha, theta)
df['scaws'], _ = p.exponential_filter(np.array(df.caws), 0.93, 0.5)

tw_north = p.cos_d(df.cawa) * df.caws - df.rspd
tw_east = p.sin_d(df.cawa) * df.caws

df['predicted_twa'] = np.degrees(np.arctan2(tw_east, tw_north))
# increase smoothness... seems to be a better fit
alpha = 0.97
df['spredicted_twa'], _  = p.exponential_filter_angle(np.array(df.predicted_twa), alpha, theta)

wdf = df.loc[tacks_slice]
c.quick_plot(wdf.index, (wdf.rawa, wdf.predicted_twa, wdf.rtwa), "rawa predicted_twa rtwa".split())

c.quick_plot(wdf.index, (wdf.cawa, wdf.scawa, wdf.rawa, wdf.sawa),
                        "cawa scawa rawa sawa".split())


# In[59]:


wdf = df.loc[wide_slice]

c.quick_plot(wdf.index, (wdf.rudder, wdf.scawa, wdf.rtwa, wdf.spredicted_twa, wdf.hdg-175),   
            "rudder scawa rtwa sp_twa hdg".split())


# We can clearly see that the fit is much better than before there are still some issues.
# 
# 1. The fit to TWA is not great.
# 1. TWA is both smoother and it **leads** our predicted TWA.
# 
# Another observation is worth making. **Notice that TWA and HDG have a very similar shape.**  Though it might be hard to see in this graph.  We can investigate this be trying to predict TWA from HDG alone.  

# In[56]:


def least_square_fit(target, signal):
    "Compute the least squares fit of target from signal."
    a = np.vstack((signal, np.ones((len(signal))))).T
    b = np.asarray(target)
    fit = np.linalg.lstsq(a, b, rcond=None)[0]
    predictions = a.dot(fit)
    return fit, predictions


# In[60]:


# Let's try to predict the TWA from HDG alone ?!?!!?
wdf = df.loc[wide_slice]
# For a short time slice is TWA a linear function of HDG?
fit, pred = least_square_fit(wdf.rtwa, wdf.rhdg)
c.quick_plot(wdf.index, (wdf.rtwa, pred, wdf.rawa, wdf.rudder), "rtwa predicted_from_hdg awa rudder".split())


# In[67]:


print(f"Best fit for TWA is {fit[1]:.3f} {fit[0]:+.3f} * HDG")
abs_error = np.abs(wdf.rtwa - pred)
print(f"Average error is {np.mean(abs_error):.3f} degrees")


# Both the fit and the graph show that the HDG and TWA are closely related (the blue and orange curves above).
# 
# Why is it possible to predict TWA from HDG alone?  It makes sense from a physical standpoint: TWA = TWD - HDG and TWD is varying very slowly over short periods of time (5000/10 = 100 seconds). During this period TWD must have been approximately 179 degrees.  
# 
# To review, the rudder is turned at 99524 and the HDG (and TWA) respond very soon after (less than a second).  The AWA is still above 20 until 99572 (5 seconds!), but at this point TWA has dropped from 45 down to 14 degrees!!
# 
# 
# ### Is the Boat really Smart??
# 
# What is going on here?  Why does the change in TWA come ahead of the change in AWA?  Why does it seem to track HDG almost perfectly?
# 
# Perhaps the above assumptions about filtering/damping are too simplistic.
# 
# If I were to estimate true wind with no real limitations, I would directly estimate the physical properties that were simple and varied slowly.  That would be TWD/TWS (true wind direction and speed).  This is a quantity of the **world**, and while it varies with time, these processes do not depend on the boat.  So for example, some boats tack a lot other very rarely, others have great drivers that keep a consistant TWA and others vary a lot.  Estimating TWA requires a filtering process that can be robust to these differences.  TWD depends only on the physics of the world.
# 
# TWA can then be computed from TWD, by subtracting heading (not the other way around!).
# 
# This explains why TWA can lead AWA.  Why wait for the AWA, which is noisy, to stabilize when the HDG is measured very accurately, with low noise (with a compass)?  TWD is likely consistent through the tack, so as HDG changes TWA does as well.
# 
# 

# In[80]:


# Let's verify that TWD and TWA are directly related by HDG + variation
variation = df.variation.mean()

# Note, it looks like the calculation takes 2 samples (pretty fast!)  Delay the logged HDG
# to take this into account.
true_hdg = p.delay(df.rhdg + variation, 2)

twd_from_twa = np.mod(df.rtwa + true_hdg, 360)
diff = df.rtwd - twd_from_twa
# Trim off the first 10 seconds.  Take a while to stabilize
abs_diff = np.abs(diff)[100:]

print(f"The average absolute difference is {abs_diff.mean()}")
print(f"The 99.9th percentile of the abs difference is {np.percentile(abs_diff, 99.9)}")
print(f"The max abs difference is {np.percentile(abs_diff, 100)}")

mu = df.rtwd.mean()
c.quick_plot(df.index, (df.rtwd-100, twd_from_twa-100, diff*20),
             "twd twd_twa 20*diff".split())

# We could do this backward 

ehdg = np.mod(df.rtwd - df.rtwa, 360)
diff = p.sign_angle(ehdg - true_hdg, 360)

c.quick_plot(df.index, (true_hdg, ehdg, 20*diff),
             "rue_hdg, ehdg diff".split())







# ### How to estimate TWD/TWS directly?
# 
# I started out considering a Kalman filter, which both estimates the quantity and its covariance.  The classic Kalman filter would proceed by iterating two steps: predict and update.
# 
# - **Predict** takes the current estimate and computes a new estimate.  This is the place to incorporate our sense of how fast wind direction may change.
# - **Update** takes the noisy observations of AWA, AWS, SPD, and HDG to compute a residual (the difference between what the model predicts and what we observed).  The estimates are then updated to reduce the residuals.
# 
# A simpler approach is an iterative filter which constantly updates its estimate of TWD/TWS to more accurately predict AWA/AWS.  This filter gradually adjusts the current estimate, rather than simply computing TWD/TWS.
# 
# The updates are quite similar. Compute the derivative of the residuals and then make small changes to TWD/TWS to reduce error.  Smaller changes yield a filter which changes more slowly and ignore more noise in the measurements.

# These are the equations for computing the observations from the hidden state.  Note they are non-linear.
# 
#     twa = twd - (hdg + var)
# 
#     tw_n = tws * p.cos_d(twa)
#     tw_e = tws * p.sin_d(twa)
# 
#     aw_n = tw_n + spd
#     aw_e = tw_e
# 
#     awa = np.degrees(np.arctan2(aw_e, aw_n))
#     aws = np.sqrt(np.square(aw_n) + np.square(aw_e))
#     
# We could get rid of a some of the non-linearity by working directly with vectors, rather than angles and magnitudes.  We can cheat abit by assuming that the observations are `aw_n` and `aw_e`.  
# 
#     twa = twd - (hdg + var)
# 
#     tw_n = tws * p.cos_d(twa)
#     tw_e = tws * p.sin_d(twa)
# 
#     aw_n = tw_n + spd
#     aw_e = tw_e
#     
# This has the mild advantage that for low speeds the awa is not particularly well defined.  This appears naturally using `aw_n` and `aw_e` (the vector is just short).
# 
# We can go further (too far!) and always estimate TW in boat coordinates, rotating back when needed (and assuming HDG is low noise).  
# 
#     aw_n = tw_n + spd
#     aw_e = tw_e
# 
# But is this last step of simplification any better than what we had before?  No.
# 
# ### Update equations
# 
# The goal is to determine how to update TWA and TWS given observations.  We will do this by taking the derivative of the observation equations.  These derivatives are used to update the states to reduce the residuals.
# 
# Short hand:
# 
#     twa = twd - (hdg + var)
#     c   = cos(twa)
#     s   = sin(twa)
# 
# These are the predicted observations:
# 
#     forward_aw_n = spd + c * tws
#     forward_aw_e =       s * tws
#     
#     residual = (observed - forward)
#     
# Given residuals, we can update the parameters using the derivatives of the predictions
#     
#     d_aw_n = [    c,   tws * -s]
#     d_aw_e = [    s,   tws *  c]
# 
#     delta_tws = r_n * c         + r_e * s
#     delta_twa = r_n * tws * -s  + r_e * tws * c
# 
# Intuitions?  If the observed aw_n is larged than what is predicted, then increase TWS if the current estimate of TWD says the boat is pointed into the true wind. 
# 

# In[121]:


@jit(nopython=True)
def estimate_true_wind_helper(epsilon, aws, awa, hdg, spd, tws, twd, variation):
    twd = np.radians(twd) + np.zeros(awa.shape)
    tws = tws + np.zeros(awa.shape)
    res_n = np.zeros(awa.shape)
    res_e = np.zeros(awa.shape)
    aw_n = aws * np.cos(np.radians(awa))
    aw_e = aws * np.sin(np.radians(awa))

    rhdg = np.radians(hdg)
    variation = np.radians(variation)

    for i in range(1, len(aws)):
        twa = twd[i-1] - (rhdg[i] + variation)
        c = np.cos(twa)
        s = np.sin(twa)
        f_aw_n = spd[i] + c * tws[i-1]
        f_aw_e =          s * tws[i-1]

        res_n[i] = (aw_n[i] - f_aw_n)
        res_e[i] = (aw_e[i] - f_aw_e)

        delta_tws = 30 * res_n[i] * c + res_e[i] * s
        delta_twd = res_n[i] * tws[i-1] * -s + res_e[i] * tws[i-1] * c

        tws[i] = epsilon * delta_tws + tws[i-1]
        twd[i] = epsilon * delta_twd + twd[i-1]

    return np.degrees(twd), tws, res_n, res_e

def estimate_true_wind(epsilon, df, awa_mult=1.0, aws_mult=1.0, spd_mult=1.0):
    return estimate_true_wind_helper(epsilon,
                                     aws = aws_mult * np.asarray(df.raws),
                                     awa = awa_mult * np.asarray(df.rawa),
                                     hdg = np.asarray(df.rhdg),
                                     spd = spd_mult * np.asarray(df.rspd),
                                     tws = df.tws.iloc[0],
                                     twd = df.twd.iloc[0],
                                     variation = df.variation.mean())

def estimate_true_wind_smooth(epsilon, df, awa_mult=1.0, aws_mult=1.0, spd_mult=1.0):
    return estimate_true_wind_helper(epsilon,
                                     aws = aws_mult * np.asarray(df.saws),
                                     awa = awa_mult * np.asarray(df.sawa),
                                     hdg = np.asarray(df.rhdg),
                                     spd = spd_mult * np.asarray(df.sspd),
                                     tws = df.tws.iloc[0],
                                     twd = df.twd.iloc[0],
                                     variation = df.variation.mean())


# In[131]:


sdf = df.loc[example.begin : example.end]

epsilon = 0.00025
if True:
    print(epsilon)
    (twd, tws, res_n, res_e) = estimate_true_wind(epsilon, sdf, 1.0, 1.0, 1.0)    
    print(np.mean(res_e), np.std(res_e))
    print(np.mean(res_n), np.std(res_n))

    (twd, tws, res_n, res_e) = estimate_true_wind_smooth(epsilon, sdf, 1.0, 1.0, 1.0)    
    print(np.mean(res_e), np.std(res_e))
    print(np.mean(res_n), np.std(res_n))

c.quick_plot(sdf.index, (sdf.twd-180, twd-180, 10*sdf.tws, 10*tws),
                 "twd etwd tws etws awa".split())


# In[ ]:


def estimate_true_wind_corrected(epsilon, df, awa_mult=1.0, aws_mult=1.0, spd_mult=1.0):
    return estimate_true_wind_helper(epsilon,
                                     aws = aws_mult * np.asarray(df.scaws),
                                     awa = awa_mult * np.asarray(df.scawa),
                                     hdg = np.asarray(df.rhdg),
                                     spd = spd_mult * np.asarray(df.sspd),
                                     tws = df.tws.iloc[0],
                                     twd = df.twd.iloc[0],
                                     variation = df.variation.mean())


# In[ ]:


sdf = df.loc[example.begin : example.end]

(twd, tws, res_n, res_e) = estimate_true_wind(0.00008, sdf, 1.0, 1.0, 1.0)    

print(np.mean(res_e), np.std(res_e))
print(np.mean(res_n), np.std(res_n))


# In[ ]:




