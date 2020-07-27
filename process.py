"""
Contains generic tools for processing and transforming boat data.
"""
import math
import os
import shutil
import os.path
import subprocess
import itertools as it
import re

import concurrent.futures

from numba import jit

import numpy as np
import scipy
import scipy.signal

from global_variables import G
import canboat as cb

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
        res1, _ = scipy.signal.lfilter(b, a, signal, zi = zi * signal[0])
        res, _ = scipy.signal.lfilter(b, a, res1,   zi = zi * res1[0])
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
    Apply a non-linear exponential filter, where alpha is the decay (higher alpha is
    longer decay and higher smoothing).  If the error is greater than max_error, then
    alpha is repeatedly reduced (faster decay).
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

################################################################
################ Log processing pipeline code

################ Generic helpers: copying, finding and modifying files  ################

def find_matching_files(directory, regex_pattern):
    "Find the matching files in a directory."
    regex = re.compile(regex_pattern)
    files = os.listdir(directory)
    return [f for f in files if regex.match(f)]

def copy_with_timestamp(from_path, to_path):
    "Copy a file and ensure the timestamps are the same."
    # Not sure why but copy2 does not always preserve times.
    G.logger.debug(f"Copying {from_path} TO {to_path}.")
    shutil.copy2(from_path, to_path)
    atime = os.path.getatime(from_path)
    mtime = os.path.getmtime(from_path)
    os.utime(to_path, times=(atime, mtime))

def copy_all_files(from_dir, to_dir, regex_pattern):
    "Copy all files that match regex."
    files = find_matching_files(from_dir, regex_pattern)
    G.logger.info(f"Found {len(files)}.")
    for f in files:
        from_path = os.path.join(from_dir, f)
        to_path   = os.path.join(to_dir, f)
        if not os.path.exists(to_path):
            copy_with_timestamp(from_path, to_path)

def update_utimes(from_dir, to_dir, regex_pattern):
    "Copy utimes from files in FROM_DIR to files in TO_DIR. If file does not exist in TO_DIR just skip."
    old_files = find_matching_files(from_dir, regex_pattern)
    res = []
    for f in old_files:
        from_path = os.path.join(from_dir, f)
        mtime = os.path.getmtime(from_path)
        atime = os.path.getatime(from_path)
        res.append((mtime, atime, f))
    res = sorted(res, key=lambda x: x[0])
    for mtime, atime, f in res:
        to_path = os.path.join(to_dir, f)
        if os.path.exists(to_path):
            os.utime(to_path, times=(atime, mtime))


################ Application specific helpers.  Have knowledge of file types/locations. ################

def usb_drive_available():
    return os.path.exists(G.USB_LOGS_DIRECTORY)

def log_basename(file):
    "Remove the extensions from log files, either .log or .log.gz."
    m = re.match(r'(^.*)\.log\.gz$', file)
    if m is not None:
        return m.group(1)
    m = re.match(r'(^.*)\.log$', file)
    if m is not None:
        return m.group(1)
    return file

def usb_log_path(filename):
    return os.path.join(G.USB_LOGS_DIRECTORY, filename)

def compressed_log_path(filename):
    return os.path.join(G.COMPRESSED_LOGS_DIRECTORY, filename)

def named_log_path(filename):
    return os.path.join(G.NAMED_LOGS_DIRECTORY, filename)

def gpx_path(filename):
    return os.path.join(G.GPX_LOGS_DIRECTORY, filename)

def pandas_path(filename):
    return os.path.join(G.PANDAS_LOGS_DIRECTORY, filename)

def usb_log_files():
    return find_matching_files(G.USB_LOGS_DIRECTORY, r'(^.*).log$')

def compressed_log_files():
    return find_matching_files(G.COMPRESSED_LOGS_DIRECTORY, r'^.*\.log\.gz$')

def named_log_files():
    return find_matching_files(G.NAMED_LOGS_DIRECTORY, r'^.*\.log\.gz$')

def uncompressed_log_files():
    return find_matching_files(G.COMPRESSED_LOGS_DIRECTORY, r'^.*\.log$')

def gpx_files():
    return find_matching_files(G.GPX_LOGS_DIRECTORY, r'^.*gpx$')

def pandas_files():
    return find_matching_files(G.PANDAS_LOGS_DIRECTORY, r'^.*pd.gz$')

def copy_from_usb(filename):
    copy_with_timestamp(usb_log_path(filename), compressed_log_path(filename))

################ Application operations. ################

def copy_err_files_from_usb():
    "Copy all the log error files from the USB drive."
    G.logger.info("Copying error files.")
    copy_all_files(G.USB_LOGS_DIRECTORY, G.COMPRESSED_LOGS_DIRECTORY, r".*\.err")

def copy_log_files_from_usb():
    "Copy all logs files which have not yet been copied from the USB drive to the compressed file staging area."
    current_logs = compressed_log_files() + uncompressed_log_files()
    G.logger.info(f"Found {len(current_logs)} current log files.")
    usb_logs = usb_log_files()
    G.logger.info(f"Found {len(usb_logs)} usb log files.")
    log_basenames = [log_basename(clog) for clog in current_logs]
    for ulog in usb_logs:
        if log_basename(ulog) not in log_basenames:
            G.logger.debug(f"Missing log {ulog}, copying.")
            copy_from_usb(ulog)

def compress_log(filename):
    path = compressed_log_path(filename)
    G.logger.info(f"Compressing {path}")
    subprocess.run(f"gzip {path}", shell=True)

def create_compressed_log_files():
    current_logs = uncompressed_log_files()
    G.logger.info(f"Found {len(current_logs)} uncompressed log files.")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        res = list(ex.map(compress_log, current_logs))
    return res

def is_valuable_compressed_log(file):
    "A valuable compressed log is not too small."
    path = compressed_log_path(file)
    return os.stat(path).st_size > G.MIN_LOG_FILE_SIZE

def extract_log_name(json_file):
    "From JSON log file, extract GPS datetime and convert to an appropriate filename."
    records = cb.json_records( cb.file_lines(json_file),
                               cb.pgn_filter(G.PGN_WHITELIST), 1)
    gps_time = cb.log_gpstime(records)
    local_time = gps_time.to('US/Pacific')
    return local_time.format('YYYY-MM-DD_HH:mm') + '.log.gz'

def symlink(cwd_path_destination, cwd_path_link):
    """
    Create a symbolic link between two files.  Use this rather than the os.symlink since
    most of the paths we use are relative, and this handles relative links correctly.
    """
    full_path_destination = os.path.abspath(cwd_path_destination)
    full_path_link = os.path.abspath(cwd_path_link)
    link_dir, _ = os.path.split(full_path_link)
    G.logger.debug(f"Linking {full_path_link} to {full_path_destination}")
    os.symlink(os.path.relpath(full_path_destination, link_dir), full_path_link)

def link_named_log_file(compressed_filename):
    """
    " Starting with a compressed log file, compute the true datetime, convert to a
    filename, and then link to the original with this new filename.
    """
    G.logger.info(f'Linking {compressed_filename}')
    compressed_path = compressed_log_path(compressed_filename)

    tmp_path = "/tmp"
    json_file = os.path.join(tmp_path, compressed_filename + ".json")
    log_file = os.path.join(tmp_path, compressed_filename + ".stderr")

    # Extract the head...  just need the first record.
    # TODO: fix constant
    command = f"zcat {compressed_path} | analyzer -json 2> {log_file} | head -10000 > {json_file}"
    G.logger.info(f'Running "{command}"')
    subprocess.run(command, shell=True)

    named_filename = extract_log_name(json_file)
    # Careful to remove JSON file, since it is not complete!  Otherwise it will be
    # considered a cached and complete file.
    os.remove(json_file)

    named_path = named_log_path(named_filename)
    G.logger.info("Linking {compressed_filename} to {named_path}")
    symlink(compressed_path, named_path)

def create_named_log_files():
    "Find the compressed files that are not yet named and linked into the named directory."
    clogs = set(compressed_log_files())
    nlogs = named_log_files()
    G.logger.info(f"Found {len(clogs)} compressed log files, {len(nlogs)} named log files.")
    for nlog in nlogs:
        target = os.path.realpath(named_log_path(nlog))
        _, name = os.path.split(target)
        G.logger.debug(f"{nlog} is linked to {name}, removing.")
        clogs.discard(name)
    G.logger.info(f"Found {len(clogs)} unlinked files.")
    for f in list(clogs):
        if not is_valuable_compressed_log(f):
            clogs.discard(f)
    G.logger.info(f"Found {len(clogs)} unlinked files are valuable.")
    for f in clogs:
        link_named_log_file(f)


def json_to_gpx(json_file, gpx_output_file, skip=3, src=5):
    "From a Canboat JSON file compute a GPX file."
    print("Reading canboat JSON log: {0}".format(json_file))

    # Skip is here to reduce the size of the resulting GPX
    lla_records = it.islice(cb.lla_records(json_file, src=src), 0, None, skip)
    gpx = cb.lla_to_gpx(lla_records, stop=None)

    print("Writing GPX result to {0}".format(gpx_output_file))
    with open(gpx_output_file, 'w') as gpx_fs:
        gpx_fs.write(gpx.to_xml())

def json_to_pandas(json_file, pandas_file, count, trim):
    "Convert the JSON file version of the canboat log to a pandas dataframe."
    print(f"Converting {json_file} to {pandas_file}.")
    df = cb.json_to_data_frame(json_file, count=count, trim=trim)
    df.to_pickle(pandas_file)
    print(f"Done converting {json_file}.")

def json_from_named_file(named_file, directory="/tmp", force=False):
    "Compute JSON from named log file.  Place in directory.  Return path to json."
    G.logger.info(f"{named_file}: Creating JSON")
    named_path = named_log_path(named_file)
    basename = log_basename(named_file)

    json_path = os.path.join(directory, basename + ".json")
    log_path = os.path.join(directory, basename + ".stderr")

    if force or not os.path.exists(json_path):
        command = f"zcat {named_path} | analyzer -json 2> {log_path} > {json_path}"
        G.logger.debug(f'Running "{command}"')
        subprocess.run(command, shell=True)
    else:
        G.logger.info(f"File {json_path} exists, skipping.")
    return json_path

def json_from_matching_named_file(named_file_prefix, directory="/tmp"):
    "Find the first matching named log file, compute JSON, and return path."
    for named_file in named_log_files():
        if named_file.startswith(named_file_prefix):
            return json_from_named_file(named_file, directory=directory)
    G.logger.warn(f"Could not find a matching named log {named_file_prefix}")
    return None

def convert_named_log_to_gpx_and_pandas_file(named_file, count, trim):
    """
    Given a named log file (compressed JSON) will convert to GPX and/or PANDAS (if missing).
    """
    G.logger.info(f"{named_file}: Examining")
    basename = log_basename(named_file)

    gpx_filename = basename + ".gpx"
    g_path = gpx_path(gpx_filename)
    pandas_filename = basename + ".pd"
    compressed_pandas_filename = basename + ".pd.gz"
    cp_path = pandas_path(compressed_pandas_filename)
    p_path = pandas_path(pandas_filename)

    if os.path.exists(g_path) and os.path.exists(cp_path):
        G.logger.debug("{named_file}: GPX and Pandas logs exist. Skipping.")
        return f"{named_file}: Skipped"

    json_path = json_from_named_file(named_file, directory="/tmp")

    if not os.path.exists(g_path):
        G.logger.debug(f"{named_file}: Creating GPX file.")
        json_to_gpx(json_path, g_path)

    if not os.path.exists(cp_path):
        G.logger.info(f"{named_file}: Creating Pandas file.")
        json_to_pandas(json_path, p_path, count, trim)
        G.logger.debug(f"{named_file}: Compressing {p_path}")
        subprocess.run(f"gzip {p_path}", shell=True)

    return f"{named_file}: Processed"

def create_gpx_and_pandas_files(count, trim):
    "Examines all named log files and computes GPX/PANDAS if necessary."
    nlogs = named_log_files()
    G.logger.info(f"Found {len(nlogs)} named log files.")
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as ex:
        res = list(ex.map(convert_named_log_to_gpx_and_pandas_file, nlogs, it.cycle([count]), it.cycle([trim])))
    return res
