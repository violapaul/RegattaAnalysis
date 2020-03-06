"""
Contains generic tools for processing and transforming boat data.
"""
import math

from numba import jit

import numpy as np
import scipy
import scipy.signal

# GPS ################################################################

def gps_min_seconds(lat, lon):
    "Return the minutes and seconds for lat and lon."
    def helper(val):
        degrees = math.floor(val)
        decimal = val - degrees
        minutes = decimal * 60
        decimal = decimal - math.floor(minutes) / 60.0
        seconds = decimal * 3600
        return (degrees, minutes, seconds)
    if lat > 0:
        lat_res = helper(lat) + ('N',)
    else:
        lat_res = helper(-lat) + ('S',)
    if lon > 0:
        lon_res = helper(lon) + ('E',)
    else:
        lon_res = helper(-lon) + ('W',)
    return lat_res, lon_res


# ANGLE HACKING #######################################################################

# Angles need special treatment, beyond what is needed for general signals.

def sign_angle(angles, degrees=True):
    "Convert to signed angles around zero, rather than compass."
    angles = np.asarray(angles)
    if degrees:
        return np.mod((angles + 180), 360) - 180
    else:
        return np.mod((angles + math.pi), 2 * math.pi) - math.pi


def compass_angle(angles, degrees=True):
    "Convert a signed angle (like AWA) to a compass style angle."
    angles = np.asarray(angles)
    if degrees:
        return np.mod(angles, 360)
    else:
        return np.mod(angles, 2 * math.pi)


def rad(angle_in_degrees):
    "Convert angle to radians."
    return np.radians(angle_in_degrees)


def deg(angle_in_radians):
    "Convert angle to degrees."
    return np.degrees(angle_in_radians)


def angle_diff_d(deg1, deg2):
    "Compute the difference in degrees between arguments, handing values greater than 360 correctly."
    delta = deg1 - deg2
    return np.mod((delta + 180), 360) - 180


def unwrap_d(angular_signal_d):
    """
    Angular signals, vary in time and sometimes cross the *wrap around* boundary,
    introducing a discontinuity which is an artifact.  Unwrapping removes these
    discontinuities by adding or subtracting 360.  Note, this can lead to large 'windup'
    where the values of angles continue to increase/decrease.  This is not uncommon in
    races with port roundings (or starboard).
    """
    return np.degrees(np.unwrap(np.radians(angular_signal_d)))


def match_wrap(reference, signal):
    "Given a reference angular signal, add or subtract 2PI to signal in order to match the wrap."
    # Compute difference mod 2PI
    delta = np.fmod(signal - reference, 2 * math.pi)
    # Pick the smaller abs angle
    delta[delta > math.pi] = delta[delta > math.pi] - 2 * math.pi
    delta[delta < -math.pi] = delta[delta < -math.pi] + 2 * math.pi
    # Add it back, which essentially re-wraps the result
    return reference + delta


def match_wrap_d(reference, signal):
    res = match_wrap(np.radians(reference), np.radians(signal))
    return np.degrees(res)


def north_d(degrees):
    "By convention, NORTH is at zero degrees."
    return np.cos(np.radians(degrees))

def east_d(degrees):
    "By convention, EAST is at 90 degrees."
    return np.sin(np.radians(degrees))


def cos_d(degrees):
    "Compute the COSINE of an angle in degrees."
    return np.cos(np.radians(degrees))

def sin_d(degrees):
    "Compute the SINE of an angle in degrees."
    return np.sin(np.radians(degrees))


# Signal Filtering ###################################################################

def delay(signal, shift):
    "Delay a signal by shift steps. Pad the new values with the current boundary value."
    s = np.asarray(signal)
    if shift > 0:
        shifted = np.roll(s, shift)
        shifted[:shift] = s[0]
        return shifted
    elif shift < 0:
        shifted = np.roll(s, shift)
        shifted[len(s)+shift:] = s[-1]
        return shifted
    return s

def local_average_filter(width):
    "Super simple running average."
    b = np.ones((width)) / width
    a = np.zeros((width))
    a[0] = 1
    return b, a


def butterworth_filter(cutoff, order):
    fs = 10                     # Sampling frequency
    omega = cutoff / (fs / 2)   # Normalize the frequency
    return scipy.signal.butter(order, omega
    , 'low')


def butterworth_bandpass(lowcut, highcut, order=5):
    fs = 10                     # Sampling frequency
    nyq = fs / 2
    low = lowcut / nyq
    high = highcut / nyq
    return scipy.signal.butter(order, [low, high], btype='band')


def chebyshev_filter(cutoff, order):
    fs = 10                     # Sampling frequency
    omega = cutoff / (fs / 2)   # Normalize the frequency
    N, Wn = scipy.signal.cheb1ord(omega, 1.5 * omega, 3, 40)
    b, a = scipy.signal.cheby1(order, 1, Wn, 'low')
    return b, a


def smooth(coeff, signal, causal=False):
    b, a = coeff
    signal = np.asarray(signal)
    if causal:
        zi = scipy.signal.lfilter_zi(b, a)
        res1, _ = scipy.signal.lfilter(b, a, signal, zi= zi * signal[0])
        res, _ = scipy.signal.lfilter(b, a, res1,   zi= zi * signal[0])
    else:
        res = scipy.signal.filtfilt(b, a, signal)
    return res


def smooth_angle(coeff, signal_degrees, degrees=True, causal=False, plot=False):
    """
    Filter an angle, which is tricky since a difference of 2pi is really zero.  This
    messes up linear filters.
    """
    signal_degrees = np.asarray(signal_degrees)
    rads = np.radians(signal_degrees) if degrees else signal_degrees
    unwrap_rads = np.unwrap(rads)
    filter_rads = smooth(coeff, unwrap_rads, causal)

    res = match_wrap(rads, filter_rads)
    ret = np.degrees(res) if degrees else res

    return ret


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

@jit(nopython=True)
def exponential_filter_angle(sig, alpha, max_error):
    """
    Apply a non-linear exponential filter, where alpha is the decay.  If the error is
    greater than max_error, then alpha is reduced by (faster decay).  This filter will
    follow the angle as it wraps around 360.
    """
    beta = beta0 = 1 - alpha
    res = np.zeros(sig.shape)
    betas = np.zeros(sig.shape)
    res[0] = sig[0]
    betas[0] = beta
    for i in range(1, len(sig)):
        delta = res[i-1] - sig[i]
        # Did the output wrap?  If so wrap the filter output.
        if delta > 360:
            res[i-1] -= 360
        elif delta < -360:
            res[i-1] += 360
        res[i] = (1 - beta) * res[i-1] + beta * sig[i]
        if np.abs(res[i] - sig[i]) > max_error:
            beta = min(1.5 * beta, 0.5)
        else:
            beta = (beta + beta0) / 2
        betas[i] = beta
    return res, betas


def exponential_filtfilt(sig, alpha, max_error):
    ## TODO
    # Is there a way to do this forward and then backward... so that there is no latency introduced??
    pass


# Miscellaneous ###################################################################

def least_square_fit(target, signal):
    "Compute the least squares fit of target from signal."
    # add a column of ones...  homogenous coordinates
    a = np.vstack((signal, np.ones((len(signal))))).T
    b = np.asarray(target)
    fit = np.linalg.lstsq(a, b, rcond=None)[0]
    predictions = a.dot(fit)
    return fit, predictions


@jit(nopython=True)
def find_runs(a):
    "Given a numpy sequence, return the start and ends of runs of non-zeros."
    res = []
    if a[0] > 0:
        started, start = True, 0
    else:
        started, start = False, 0
    for i in range(a.shape[0]):
        if started:
            if not a[i] > 0:
                res.append((start, i))
                started = False
        else:
            if a[i] > 0:
                started, start = True, i
    if started:
        res.append((start, i))
    return res


def max_min_mid(values, border=0.1):
    "Return the range of a series, with a buffer added which is border times the range."
    max = values.max()
    min = values.min()
    mid = 0.5 * (max + min)
    delta = (max - min)
    max = max + border * delta
    min = min - border * delta
    return np.array((max, min, mid))

# Instrument processing

# For copious documuntation and discussion of this function, see Boat_Instruments.ipynb
# 
@jit(nopython=True)
def estimate_true_wind_helper_leeway(epsilon, aws, awa, hdg, spd, cog, sog, 
                                     tws_init, twd_init, variation, tws_mult):
    # TWD/TWS are ouptputs.  This sets the initial conditions.
    twd = np.radians(twd_init) + np.zeros(awa.shape)
    tws = tws_init + np.zeros(awa.shape)
    # Residuals are stored here.
    res_n = np.zeros(awa.shape)
    res_e = np.zeros(awa.shape)
    # Process apparent wind to decompose into boat relative components.
    aw_n = aws * np.cos(np.radians(awa))
    aw_e = aws * np.sin(np.radians(awa))

    # preconvert some values
    rhdg = np.radians(hdg)
    rcog = np.radians(cog)
    variation = np.radians(variation)
    
    time_shift = 10
    eps = epsilon
    for i in range(1, len(aws)-time_shift):
        # Transform to boat relative angles.
        twa = twd[i-1] - (rhdg[i] + variation)
        course_angle = rcog[i+time_shift]  - (rhdg[i] + variation)
        
        # Useful below
        c = np.cos(twa)
        s = np.sin(twa)
        
        # Boat relative vector of true wind
        twn = c * tws[i-1]
        twe = s * tws[i-1]
        
        # Boat relative vector of travel
        btn = np.cos(course_angle) * sog[i+time_shift]
        bte = -np.sin(course_angle) * sog[i+time_shift]
        
        # The forward predictions, introduce leeway
        f_aw_n = twn + btn
        f_aw_e = twe + bte

        # Residuals
        res_n[i] = (aw_n[i] - f_aw_n)
        res_e[i] = (aw_e[i] - f_aw_e)

        # derivatives
        delta_tws = res_n[i] * c + res_e[i] * s
        delta_twd = res_n[i] * tws[i-1] * -s + res_e[i] * tws[i-1] * c

        if np.sqrt(np.square(res_n[i]) + np.square(res_e[i])) > 1.5:
            eps = min(5*epsilon, 1.1 * eps)
        else:
            eps -= 0.2 * (eps - epsilon)
        
        # Hack, which the update to TWS and TWD don't have the same magnitude
        tws[i] = eps * tws_mult * delta_tws + tws[i-1]
        twd[i] = eps * delta_twd + twd[i-1]

    return np.degrees(twd), tws, res_n, res_e


@jit(nopython=True)
def estimate_true_wind_helper_leeway(epsilon, aws, awa, hdg, spd, cog, sog, 
                                     tws_init, twd_init, variation, tws_mult):
    # Components of True Wind, initialized.
    tw_n = np.cos(np.radians(twd_init)) * tws_init + np.zeros(awa.shape)
    tw_e = np.sin(np.radians(twd_init)) * tws_init + np.zeros(awa.shape)    
    
    # Residuals are stored here.
    res_n = np.zeros(awa.shape)
    res_e = np.zeros(awa.shape)
    # Process apparent wind to decompose into boat relative components.
    aw_n = aws * np.cos(np.radians(awa))
    aw_e = aws * np.sin(np.radians(awa))

    # preconvert some values
    rhdg = np.radians(hdg + variation)
    rcog = np.radians(cog)
    variation = np.radians(variation)
    
    time_shift = 10
    eps = epsilon
    for i in range(1, len(aws)-time_shift):
        # Useful below
        c = np.cos(rhdg[i])
        s = np.sin(rhdg[i])

        # True wind rotated into boat coordinates
        tw_bn =  c * tw_n[i-i] + s * tw_e[i-1]
        tw_be = -s * tw_n[i-i] + c * tw_e[i-1]
        course_angle = rcog[i+time_shift]  - rhdg[i]

        # Boat relative vector of travel
        btn = np.cos(course_angle) * sog[i+time_shift]
        bte = -np.sin(course_angle) * sog[i+time_shift]
        
        # The forward predictions, introduce leeway
        f_aw_n = tw_bn + btn
        f_aw_e = tw_be + bte

        # Residuals
        res_n[i] = (aw_n[i] - f_aw_n)
        res_e[i] = (aw_e[i] - f_aw_e)

        # derivatives
        delta_tw_n = res_n[i] * c + res_e[i] * -s
        delta_tw_e = res_n[i] * s + res_e[i] * c        
        
        if np.sqrt(np.square(res_n[i]) + np.square(res_e[i])) > 1.5:
            eps = min(5*epsilon, 1.1 * eps)
        else:
            eps -= 0.2 * (eps - epsilon)
        
        # Hack, which the update to TWS and TWD don't have the same magnitude
        tw_n[i] = eps * delta_tw_n + tw_n[i-1]
        tw_e[i] = eps * delta_tw_e + tw_e[i-1]

    return np.degrees(twd), tws, res_n, res_e

