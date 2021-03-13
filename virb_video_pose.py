
"""
Garmin Virb 360 produces two types of a files: MP4 videos and FIT metadata.  In a
given recording there is a single FIT file, while videos are broken into 30 min chunks
(the FIT file contains annotation showing the precise start and end of each video).  The
FIT file is processed to find a camera poses associated with each video, which are then
stored in separate binary pose files.

First task: given a set of video files and pose files, find the correct association by
timestamp.

The pose file timestamp is encoded in its filename.  Pose files data and timestamp are
extracted from the FIT file.  These times are assumed to be precise (milliseconds are
encoded).

The video file creation times are encoded in the MP4 header, which is extracted by
exiftool (stored in videos.json).  Video files appear to be created after the beginning of
the capture.  Note, the two timestamps are not identical.  The videos files are typically created 27ish
seconds after the pose file begins.

    exiftool -api largefilesupport=1 -createdate -json *.MP4 > videos.json

In addition we may have log data captured using the raspberry PI and stored in pandas (an
entirely separate system running on a different clock).  The clocks can be compared (in
most cases) because both are synced to GPS.

Similar to a FIT file, there is a single instrument log for the entire day.  The first
task is to find the correct log for each video.  Each video then refers to a slice of the
log, and this slice is extracted and stored in a binary log file.

The binary log files are stored using the same name as the pose file (since this is the
precise time encoding) and placed in a different directory.

When the log exists, this is stored in vidoe_poses.json as well.
"""

import datetime

import arrow
import json
import re
import os
import bisect
import struct
import pandas as pd

import numpy as np

from global_variables import G
import process as p
import race_logs

G.init_seattle()

WEB_DIR = "/Users/viola/www"

# videos.json is created by exiftool
VIDEO_FILENAME = os.path.join(WEB_DIR, "videos.json")
POSE_DIRECTORY = os.path.join(WEB_DIR, "Poses")
LOG_DIRECTORY = os.path.join(WEB_DIR, "Logs")
FULL_LOG_DIRECTORY = os.path.join(WEB_DIR, "FullLogs")

def get_video_metadata(filename):
    """
    Video data is stored in a json file with filename and datetime.  Read and convert
    datetime string to timestamp.
    """
    with open(filename, 'r') as fs:
        videos = json.load(fs)

    for i, video in enumerate(videos):
        vdate = arrow.get(video['CreateDate'], 'YYYY:MM:DD HH:mm:ss', tzinfo='US/Pacific')
        video['id'] = i
        video['ts'] = vdate.timestamp

    return videos


def get_poses(directory):
    """
    Read all of the pose files, and create an ordereddict by timestamp (which is encoded
    in the filename).
    """
    regex = re.compile(r".*.bin$")
    files = sorted(os.listdir(directory))

    res = []
    for file in files:
        if regex.match(file):
            date_string = os.path.splitext(file)[0]
            date = arrow.get(date_string, "YYYY-MM-DDTHH_mm_ss.S", tzinfo='US/Pacific')
            ts = date.timestamp
            ts_ms = ts + date.microsecond/1000000.0
            res.append(dict(PoseFile=file, pose_date=date, ts=ts, ts_ms=ts_ms))
    return res


def pandas_basename(file):
    "Given the filename of a pandas log file, return the basename."
    m = re.match(r'(^.*)\.pd.gz$', file)
    if m is not None:
        return m.group(1)
    else:
        G.logger.warning("Might not be pandas log. Could not match {file}.")
        return None


def get_log_files():
    log_files = sorted(p.pandas_files())
    res = []
    for i, file in enumerate(log_files):
        base = pandas_basename(file)
        date = arrow.get(base, 'YYYY-MM-DD_HH:mm', tzinfo='US/Pacific')
        ts = date.timestamp
        ts_ms = ts + date.microsecond/1000000.0
        res.append(dict(LogFile = file, date = date, ts=ts, ts_ms=ts_ms))
    return res


def find_neareset(sorted_values, probe):
    "Find nearest element in a sorted list.  The missing function in bisect."
    index = bisect.bisect(sorted_values, probe)
    if index == 0:
        pass                    # first element is larger
    elif index == len(sorted_values):
        index = index - 1       # last element is smaller
    else:
        if (probe - sorted_values[index-1]) < (sorted_values[index] - probe):
            index = index - 1   # Closest to last element
    return index, sorted_values[index]


def find_matching_pose_file(videos, poses):
    """
    For each video, find the pose file that has the closest match in timestamp.  For some
    reason the stamp is often different by up to 30 seconds.  If the match is worse than
    60 seconds, then complain.
    """
    pose_times = sorted([p['ts_ms'] for p in poses])

    for video in videos:
        ts = video['ts']
        index, closest_ts = find_neareset(pose_times, ts)
        delta = abs(pose_times[index] - ts)
        if delta < 60:
            pose = poses[index]
            video.update(pose)
            video['delta_ts'] = delta
        else:
            G.logger.warning(f"Time delta between video and pose is too large: {delta}")


def find_matching_log_files(videos, logs, max_time_gap_hours=12):
    log_times = sorted([l['ts_ms'] for l in logs])

    for video in videos:
        video_timestamp = video['ts_ms']
        ## Find the index to item which is greater
        index = bisect.bisect_left(log_times, video_timestamp)
        if index > 0:
            # grab the one before
            log = logs[index-1]
            delta = video['pose_date'] - log['date']
            if delta < datetime.timedelta(hours=max_time_gap_hours):
                # add this pose file to the log
                log['video_files'] = log.get('video_files', []) + [video]
                continue
        G.logger.warning(f"No matching log file for video: {video['SourceFile']}. Closest is {delta}.")
    return [log for log in logs if log.get('video_files')]


def pose_file_stats(filename):
    "Return the stats of the pose file: nrows, ncols, duration in ms"
    with open(filename, mode='rb') as dfile:  # b is important -> binary
        data = dfile.read()

    nrow, ncol = struct.unpack("qq", data[:16])
    last_row = data[16 + ((nrow-1) * ncol * 8):]

    # rows are 5 doubles: timestamp, followed by 4 for the pose quaternion
    deconstruct = "d" * 5
    row_data = struct.unpack(deconstruct, last_row)
    duration = row_data[0]
    return nrow, ncol, duration


def clean_video_keys(videos):
    keys_to_keep = "SourceFile CreateDate PoseFile LogFile id ts".split()
    return list(map( lambda v: {k: v[k] for k in keys_to_keep if k in v}, videos))


def spad(val, length=40):
    "Return a left padded string."
    return str(val).ljust(length)


def extract_log_data(logs, df_convert, regenerate=False):
    """
    Given a set of logs that and assocated pose files, extract key log data and
    save to a raw binary file.
    """
    for log in logs:
        print()
        print()
        G.logger.info(f"Processing log file: {log['LogFile']}")
        df = None
        for video in log['video_files']:
            pose_file = video['PoseFile']
            log_file = os.path.join(LOG_DIRECTORY, pose_file)
            G.logger.info(f"Generating {log_file}")
            if os.path.exists(log_file) and not regenerate:
                G.logger.info(f"{log_file} exists, skipping")
                video['LogFile'] = pose_file
                continue
            if df is None:
                # only load the log dataframe if necessary.
                df = race_logs.read_log_file(log['LogFile'])
            nrows, ncols, duration = pose_file_stats(os.path.join(POSE_DIRECTORY, pose_file))
            start_time = video['pose_date']
            end_time = start_time.shift(seconds=duration/1000.0)
            print(spad(start_time), spad(end_time))
            sdf = df[(df.row_times >= start_time.datetime) & (df.row_times <= end_time.datetime)]
            if len(sdf) > 5:
                print(spad(sdf.row_times.iloc[0]), spad(sdf.row_times.iloc[-1]))
                # make sure there is some data in the log
                sdf = df_convert(sdf)
                sdf.row_seconds = sdf.row_seconds - sdf.row_seconds.iloc[0]
                print(sdf.shape)

                video['LogFile'] = pose_file  # log file saved with same name as pose file
                log_file = os.path.join(LOG_DIRECTORY, pose_file)
                save_log_to_binary(sdf, log_file)
            else:
                print(spad(df.row_times.iloc[0]), spad(df.row_times.iloc[-1]))
                G.logger.warning(f"Missing data in log file {log['LogFile']}")


def save_log_to_binary(df, log_file):
    with open(log_file, 'wb') as fs:
        rows, cols = df.shape
        fs.write(struct.pack("qq", rows, cols))
        fs.write(df.values.tobytes('F'))


COLUMN_LIST = ['row_seconds', 'spd', 'sog',
               'latitude', 'longitude', 'turn_rate',
               'rudder',  'depth', 'altitude',
               'zg100_pitch', 'zg100_roll', 
               'awa', 'aws', 'twd', 'tws', 'twa',
               'hdg', 'cog', 'map_lon', 'map_lat',
               'row_times']


KNOTS_CONVERSION_LIST = ['spd', 'sog', 'aws', 'tws']

def knots_convert(df):
    """Convert a set of columns from meters per second to knots."""
    for field in KNOTS_CONVERSION_LIST:
        df[field] = df[field] * G.MS_2_KNOTS


def latlon_convert(df):
    """Convert the longitude and latitude to North/East meters."""
    lon, lat = np.asarray(df.longitude), np.asarray(df.latitude)
    map_lon, map_lat = G.MAP(lon, lat)
    df['map_lon'] = map_lon
    df['map_lat'] = map_lat

def javascript_row_times(df):
    utc_not_localized = df.row_times.dt.tz_convert('UTC').dt.tz_localize(None)
    # like Unix epoch, but in milliseconds
    df['row_times'] = (utc_not_localized - pd.Timestamp("1970-01-01")) // pd.Timedelta("1ms")


def df_convert(df):
    "Convert and cleanup the dataframe before serializing."
    cdf = df.copy()
    latlon_convert(cdf)
    for c in COLUMN_LIST:
        if c not in cdf.columns:
            G.logger.warning(f"Missing columns {c}.")
            # can't miss a column, since the data is dumped in raw binary and all columns
            # are assumed.
            cdf[c] = 0.0  # just through in a default
    knots_convert(cdf)
    javascript_row_times(cdf)
    return cdf[COLUMN_LIST]


def column_dict():
    "Return a dictionary that maps from column name to column number."
    return {c: i for i, c in enumerate(COLUMN_LIST)}


def fix_video_times(videos, poses, video_sequence_number, first_pose_file):
    """
    If you fire up the camera and capture video *before* getting GPS lock, then all the
    VIDEO timestampes are screwed up.  The camera just uses the internal clock which does
    not have a battery.  Likely the date times are just after the last capture.  The FIT
    files do have the right times.  So if you process the FIT files the times can be recaptured.
    """
    # find the videos in the sequence
    # find the pose file sequence
    # copy the times from the poses to the video file records.
    pose_files = []
    matching = False
    for pose in poses:
        if matching:
            pose_files.append(pose)
        elif pose['PoseFile'] == first_pose_file:
            matching = True
            pose_files.append(pose)
    res = []
    for video in videos:
        if video['SourceFile'].startswith(video_sequence_number):
            print(video)
            print(pose_files[0])
            video = {**video, **(pose_files.pop(0))}
            # fix the createdate
            video['CreateDate'] = pose['pose_date'].format()
        res.append(video)
    return res
        

def find_video_poses_logs():
    """
    Read the video metadata file and associate correct pose file.
    """

    videos = get_video_metadata(VIDEO_FILENAME)
    poses = get_poses(POSE_DIRECTORY)
    logs = get_log_files()

    find_matching_pose_file(videos, poses)
    videos = fix_video_times(videos, poses, "V118", '2021-03-06T09_24_14.145.bin')    
    videos_with_pose = [v for v in videos if v.get('PoseFile') is not None]

    logs_with_videos = find_matching_log_files(videos_with_pose, logs)

    extract_log_data(logs_with_videos, df_convert)

    output_videos = clean_video_keys(videos_with_pose)
    res_filename = os.path.join(WEB_DIR, "video_poses_logs.json")
    with open(res_filename, 'w') as fs:
        json.dump(dict(videos = output_videos, columns = column_dict()), fs, indent=4)


def convert_logs(force_recompute=False):
    """
    Process each panda file and construct a javascript friendly binary file.
    """
    logs = get_log_files()
    # logs = logs[:2]

    for log in logs:
        log_file = log['LogFile']
        base = log_file.replace('.pd.gz', '')
        binary_file = os.path.join(FULL_LOG_DIRECTORY, base + ".bin")
        if os.path.exists(binary_file) and not force_recompute:
            G.logger.info(f"Skipping: {log_file} binary file exists.")
        else:
            G.logger.info(f"Processing log file: {log_file}")
            df = race_logs.read_log_file(log_file)
            if df is not None:
                sdf = df_convert(df)
                G.logger.info(f"Read: {df.shape}.")

                binary_file = os.path.join(FULL_LOG_DIRECTORY, base + ".bin")
                save_log_to_binary(sdf, binary_file)

    res_filename = os.path.join(WEB_DIR, "columns.json")
    with open(res_filename, 'w') as fs:
        json.dump(dict(columns = column_dict()), fs, indent=4)


