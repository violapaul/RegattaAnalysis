
#### Cell #0 Type: markdown ####################################################

#: # Virb Video
#: 
#: The goal is to be able to automatically process video from 
#: 
#: - Read the gyro/accel/mag data from the FitFile
#: - Figure out timing.  Is the data synched, or really async
#: - Estimate rotation
#: 


#### Cell #1 Type: module ######################################################

import itertools as it
import pandas as pd
import collections

# These are libraries written for RaceAnalysis
from global_variables import G
from nbutils import display_markdown, display

G.init_seattle(logging_level="INFO")

import matplotlib.pyplot as plt
import numpy as np

# This is the python-fitparse library
from fitparse import FitFile

#### Cell #2 Type: module ######################################################

# Read a smallish fit file.  Note, reading FIT file with these tools is *SLOW* (10s of seconds or more).
# This may not be a problem, since we are reading the file in order to process video, which is slower still.
# But consider pre-processing and the serializing as a pandas file.

# ff = FitFile('Data/Virb360Fit/2020-04-09-21-41-03.fit')
# ff = FitFile('/Volumes/Big/Virb/GMetrix/2020-07-11-09-27-15.fit')
# ff = FitFile('Data/Virb360Fit/2020-07-11-09-27-15.fit')

# Short video from my backyard
ff = FitFile('Data/Virb360Fit/2020-08-30-17-24-00.fit')

recording = dict(fitfile = '/Users/viola/Python/sailing/Data/Virb360Fit/2020-09-06-16-48-20.fit',
                 video_file_name = 'V0920368.MP4',
                 description = 'rotate 180 to right yaw, rotate 180 to right yaw, pause, wiggle in yaw, rotate left 360 in yaw, fast right 360 yaw, roll left 90 degrees, pitch up and then pitch down (several times), roll back, returning to original position')

recording2 = dict(fitfile = '/Users/viola/Python/sailing/Data/Virb360Fit/2020-12-20-15-20-42.fit',
                 video_file_name = 'V1110476.MP4',
                 description = 'inside on desk.  No gps.  Pitch up 90.  Pitch down 90. Roll left 90. Roll right 90. Yaw left 90. Yaw right 90.')
           

ff = FitFile(recording2['fitfile'])
# ff = FitFile('Data/Virb360Fit/2020-07-11-09-27-15.fit')
fitfile = ff

#### Cell #3 Type: module ######################################################

# print a single message

messages = it.islice(ff.get_messages(), 0, 5, None)
msg = next(messages)
print(repr(msg))

#### Cell #4 Type: module ######################################################

# Display the various operations available on a message.

def display_msg_details(msg):
    display_markdown("**`mesg_num` refers to the FIT file global schema**")
    print(msg.mesg_num)
    display_markdown("**`as_dict()` The reference to the raw data is likely helpful, and very verbose**")
    print(msg.as_dict())
    display_markdown("**`def_mesg` Every data message has an associated definition message.**")
    print(msg.def_mesg)
    display_markdown("**`fields` The fields are defined in the definition message, along with info on conversions.**")
    print(msg.fields)
    display_markdown("**`get_values()`: the values in the record...  converted when possible.**")
    print(msg.get_values())
    display_markdown("**`header`**")
    print(msg.header)
    display_markdown("**`name`: more like the type of the data message.**")
    print(msg.name)
    display_markdown("**`type`: either data or definition.**")
    print(msg.type)
    
display_msg_details(msg)

#### Cell #5 Type: module ######################################################

# Find all the types of messages and keep one of each.

msg_dict = {}
for i, m in enumerate(it.islice(ff.get_messages(), 0, 50000, None)):
    if i % 10000 == 0:
        print(i)
    msg_dict[m.name] = m

display(list(msg_dict.keys()))
message_types = ""
for k, msg in msg_dict.items():
    message_types += f"- **{k}**\n   - {repr(msg)}\n"
    
display_markdown(message_types)

#### Cell #6 Type: module ######################################################

# Display some messages with time stamps

messages = it.islice(ff.get_messages(), 0, 10, None)

def message_time(msg_values):
    """
    Given message values, return the message timestamp, and if available the fulltime,
    which is the time plus offset in milliseconds.
    """
    ts = msg_values.get('timestamp', None)
    ts_ms = msg_values.get('timestamp_ms', None)
    fulltime = None
    if ts is not None:
        fulltime = float(ts)
        if ts_ms is not None:
            fulltime = ts + ts_ms/1000.0
    return ts, fulltime


def display_messages(messages):
    for i, msg in enumerate(messages):
        vals = msg.get_values()
        ts, ft = message_time(vals)
        if ts is None:
            ts = 0
            ft = 0.0
        print(f"# {i} TS: {ts:5d} FT: {ft:5.2f} -------------")
        print(repr(msg))

display_messages(messages)

#### Cell #7 Type: module ######################################################

# Timestamps are in seconds (most likely to save file space), finer time resolution is encoded 
# in the message. 

# Note the messages are **NOT** in timestamp order!!!

messages = it.islice(ff.get_messages(), 0, 50, None)

# Note, the messages are *NOT* in monotonic order!  Why?
def print_message_times(messages):
    msg = None
    for i, msg in enumerate(messages):
        vals = msg.get_values()
        ts, ft = message_time(vals)
        if ts is None:
            ts, ft = 0, 0.0
        print(f"{i:3d}, {ts:5d}, {ft:7.2f} {msg.name}")
    return msg
        
print_message_times(messages)

#### Cell #8 Type: module ######################################################

# Barometer data is very regular
messages = it.islice(ff.get_messages('barometer_data'), 0, 10000, 10)

msg = print_message_times(messages)

display_markdown("The data in the baro message contains many samples at different offsets.")
display(msg.get_values())


#### Cell #9 Type: module ######################################################

# Frequent messgaes have the same timestamp.  Time is encoded in ms
# Any given message type appears to be in order.  

display_markdown("Every message.")
messages = it.islice(ff.get_messages('gps_metadata'), 0, 100, 1)
print_message_times(messages)

display_markdown("Every 100th message.")
messages = it.islice(ff.get_messages('gps_metadata'), 0, 1000, 100)
print_message_times(messages)

display_markdown("Notice the slight drift in time.")

#### Cell #10 Type: module #####################################################

def extract_flat(messages, fields):
    rows = []
    for i, msg in enumerate(messages):
        vals = msg.get_values()
        ts_sec = vals.get('timestamp', 0)
        ts_ms = vals.get('timestamp_ms', 0)
        ts = ts_sec + ts_ms / 1000.0
        row = dict(ts = ts)
        for key in fields:
            row[key] = vals.get(key, None)
            rows.append(row)
    return pd.DataFrame(rows)


msg = msg_dict.get('gps_metadata')
if msg:
    print(msg)

    messages = it.islice(ff.get_messages('gps_metadata'), 0, 3000, 1)

    gps_fields = ['timestamp', 'position_lat', 'position_long', 'enhanced_altitude',
                  'enhanced_speed', 'utc_timestamp', 'timestamp_ms', 'heading', 'velocity']

    gps_df = extract_flat(messages, gps_fields)

def extract_multitime_measurements(messages, fields):
    """
    A multitime message, has many measurements embedded in a single messages (for
    efficiency).  The message has a timestamp in two parts, seconds and milliseconds.
    Each measurement additional has a offset from that in ms.

    Additionally a multitime message may have multiple measurements (gyro and accel have
    3: X, Y, and Z).

       - <DataMessage: gyroscope_data (#164) -- local mesg: #14, fields: [timestamp: 194, sample_time_offset: (0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, None, None, None, None, None), gyro_x: (32808, 32813, 32817, 32823, 32825, 32826, 32818, 32812, 32809, 32801, 32802, 32805, 32807, 32804, 32800, 32799, 32797, 32766, 32686, 32926, 32909, 32708, 32891, 32825, 32776, None, None, None, None, None), gyro_y: (32764, 32762, 32764, 32763, 32766, 32770, 32771, 32775, 32777, 32777, 32776, 32779, 32782, 32782, 32785, 32786, 32786, 32789, 32811, 32772, 32789, 32788, 32791, 32794, 32781, None, None, None, None, None), gyro_z: (32697, 32697, 32699, 32699, 32701, 32701, 32701, 32701, 32698, 32698, 32699, 32699, 32699, 32698, 32697, 32694, 32702, 32713, 32423, 32487, 32741, 32691, 32693, 32693, 32693, None, None, None, None, None), timestamp_ms: 239]>
       - <DataMessage: accelerometer_data (#165) -- local mesg: #15, fields: [timestamp: 194, sample_time_offset: (0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, None, None, None, None, None), accel_x: (32724, 32728, 32730, 32720, 32730, 32728, 32726, 32737, 32739, 32733, 32734, 32735, 32734, 32736, 32737, 32752, 32650, 32737, 32810, 32735, 32736, 32728, 32773, 32742, 32752, None, None, None, None, None), accel_y: (32786, 32794, 32791, 32787, 32778, 32774, 32786, 32776, 32780, 32790, 32790, 32788, 32780, 32778, 32774, 32831, 32282, 32642, 33045, 32374, 33018, 32811, 32637, 32872, 32786, None, None, None, None, None), accel_z: (30665, 30666, 30662, 30655, 30664, 30660, 30659, 30669, 30665, 30663, 30666, 30660, 30661, 30669, 30674, 30671, 30685, 30713, 30695, 30752, 30594, 30652, 30712, 30656, 30685, None, None, None, None, None), timestamp_ms: 239]>
    """
    columns = collections.defaultdict(list)
    for i, msg in enumerate(messages):
        vals = msg.get_values()
        ts_sec = vals.get('timestamp', 0)
        ts_ms = vals.get('timestamp_ms', 0)
        ts = ts_sec + ts_ms / 1000.0
        offsets = [dt for dt in vals.get('sample_time_offset') if dt is not None]
        columns['ts'] += [ts + dt/1000.0 for dt in offsets]
        columns['sample_time_offset'] += [dt for dt in offsets]
        columns['timestamp'] += [ts_sec for dt in offsets]
        columns['timestamp_ms'] += [ts_ms for dt in offsets]        
        for key in fields:
            sensor_values = [v for v, dt in zip(vals.get(key), offsets)]
            columns[key] += sensor_values
    return pd.DataFrame(data=columns)

messages = it.islice(ff.get_messages('barometer_data'), 0, 300, 1)
baro_fields = ['baro_pres']
baro_df = extract_multitime_measurements(messages, baro_fields)


def virb_video_segments(messages):
    """
    A single FIT file often refers to multiple video files (each about 30 mins).  The
    start and end times of these segments is signaled by start and end messages.

    Returns a list of start/end times.
    """
    segments = []
    current = [None, None]
    for msg in messages:
        if msg.name == 'camera_event':
            vals = msg.get_values()
            if vals.get('camera_event_type', None) == 'video_start':
                _, current[0] = message_time(vals)
            if vals.get('camera_event_type', None) == 'video_end':
                _, current[1] = message_time(vals)
                segments.append(current)
                current = [None, None]
    return segments


def three_d_sensor_calibrations(messages, sensor_type):
    """
    Extract the sequence of calibration messages from the file.  Note, currently designed
    for gyro and accel.

    Example:
    
     {'timestamp': 200,
      'gyro_cal_factor': 5,
      'calibration_divisor': 82,
      'level_shift': 32768,
      'offset_cal': (35, 13, -70),
      'orientation_matrix': (-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
      'sensor_type': 'gyroscope'}]
    """
    res = []
    for msg in messages:
        if msg.name == 'three_d_sensor_calibration':
            vals = msg.get_values()
            if sensor_type == vals.get('sensor_type', None):
                res.append(vals)
    return res


def calibrate_sensors(df, calibrations, fields):
    """
    Convert the sensor measurements from raw form into more meaningful units.  Typically
    converts from an unsigned int16(?) to signed float.  Includes bias offsets (this is
    really the calibration bit, the rest is unit conversion).

    Works for gyro and accel, which are measured in triples (x, y, z).
    """
    for cal1, cal2 in pairwise_longest(calibrations):
        if cal2 is None:
            row_selector = df.ts > cal1['timestamp']
        else:
            row_selector = (df.ts > cal1['timestamp']) & (df.ts <= cal2['timestamp'])
        
        for i, field in enumerate(fields):
            # Iterate over each of the field (typically x, y, z).
            df.loc[row_selector, field+"_raw"] = df.loc[row_selector, field]
            # Shift, typically to make it signed.
            df.loc[row_selector, field] -= cal1['level_shift']
            # Include bias offset (not clear how this is determined)
            df.loc[row_selector, field] -= cal1['offset_cal'][i]
            # These could have been combined, but we seems to prefer INTS.
            df.loc[row_selector, field] *= cal_factor(cal1)
            df.loc[row_selector, field] /= cal1['calibration_divisor']


def fitfile_messages(fitfile, msg_name=None, msg_slice=slice(None, None, None)):
    "Extract the set of messages which match msg_name and msg_slice."
    for msg in it.islice(fitfile.get_messages(), msg_slice.start, msg_slice.stop, msg_slice.step):
        if msg_name is None or msg.name == msg_name:
            yield msg


def pairwise_longest(stuff):
    "Returns a list of sequential pairs, with the last having NULL as the second element."
    one, two = it.tee(stuff)
    next(two)
    return it.zip_longest(one, two)


def cal_factor(calibration_record):
    "Abstract the extraction of the cal factor for two types of messages: gyro and accel."
    if 'gyro_cal_factor' in calibration_record:
        return calibration_record['gyro_cal_factor']
    elif 'accel_cal_factor' in calibration_record:
        return calibration_record['accel_cal_factor']
    else:
        raise Exception("cal_factor missing from {calibration_record}")


def gyro_process(fitfile, msg_slice=slice(None, None, None)):
    return imu_process(fitfile, "gyro_x gyro_y gyro_z".split(), 'gyroscope')

def accel_process(fitfile, msg_slice=slice(None, None, None)):
    return imu_process(fitfile, "accel_x accel_y accel_z".split(), 'accelerometer')


def imu_process(fitfile, fields, sensor_name, msg_slice=slice(None, None, None)):
    """
    Extract IMU data and convert to meaningful units.  Their are two types of IMU
    messages, GYROSCOPE and ACCELEROMETER, stored in two different messages (though there
    appears to be a single sensor, so all messages are timestamped with the same times).
    """

    if False:  # test
        fitfile = ff
        msg_slice = slice(None, None, None)
        sensor_name = "accelerometer"
        fields = "accel_x accel_y accel_z".split()
        
    calibrations = three_d_sensor_calibrations(fitfile_messages(fitfile, msg_slice=msg_slice), sensor_name)

    # A large FIT file can be broken into multiple video segments.  Extract and label these.
    segment_times = virb_video_segments(fitfile_messages(fitfile, msg_slice=msg_slice))

    df = extract_multitime_measurements(fitfile_messages(fitfile, sensor_name+'_data', msg_slice=msg_slice), fields)
    df['diff'] = df.ts.diff()

    calibrate_sensors(df, calibrations, fields)
    
    video_segments(df, segment_times, fields)
    plot_sensors(df, fields)

    return df


def video_segments(df, segment_times, fields):
    df['segment'] = 0
    df['ts_segment'] = df.ts 
    for segment_number, (ts_start, ts_end) in enumerate(segment_times):
        row_selector = (df.ts > ts_start) & (df.ts <= ts_end)
        df.loc[row_selector, 'ts_segment'] = df.ts_segment - ts_start
        df.loc[row_selector, 'segment'] = segment_number + 1


def plot_sensors(df, fields, fignum=None):
    fig = plt.figure(num=fignum)
    fig.clf()
    ax = fig.add_subplot(111)    
    for field, color in zip(fields, "r g b".split()):    
        ax.plot(df.ts, df[field], color=color)
    ax.legend(fields)

def plot_sensors_fancy(df, fields, fignum=None):
    fig = plt.figure(num=fignum)
    fig.clf()
    ax = fig.add_subplot(111)    
    ts_start = df[df.segment >= 0].ts.min()

    max_val = 0
    for field, color in zip(fields, "r g b".split()):    
        ax.plot(df.ts-ts_start, df[field], color=color)
        max_val = max(max_val, df[field].max())
    segment_count = df.segment.max()
    ax.plot(df.ts-ts_start, max_val*df.segment/segment_count, color='orange')
    ax.legend(fields + ['video'])

    

################################################################

file = recording2['fitfile']
fitfile = FitFile(file)


plt.close('all')
# Extract the gyro and accel messages and stuff them in dataframes.
gdf = gyro_process(fitfile)
adf = accel_process(fitfile)

# Combines rows that have the same timestamp.  Assumes the accel and gyro messages are in
# *perfect* sync (they are essentially the same sensor, but encoded in two messages).
imu_df = adf.merge(gdf['ts gyro_x gyro_y gyro_z'.split()], on='ts')

# Drop the first N messages... just to get warmed up.
imu_df = imu_df.iloc[100:]

import geometry as geo

# Pose of the camera in camera coordinates?
pose = np.eye(3)
ihat = pose[:,0] # Right
jhat = pose[:,1] # Back
khat = pose[:,2] # Down

# Estimated rotation 
qest = geo.q_identity()

# We will construct a complementary filter, to estimate the roll and pitch of the camera
# in the world (and not yaw).  There are two sensors for roll/pitch the accel and gyro.
#
# The gyro is a measure of the rotation rate, in theory the integral of the rate should
# yield the rotation, except for the constant offset which is the initial condition
# (remember that C that appears when integrating?).  Additionally the gyro is often
# corrupted by bias, which is a small offset in the rate estimate.  The bias is not a big
# deal **unless* you integrate, in which case you get the integral of the bias term, which
# can be a large source of error.
#
# The accel measures the forces applied to the camera (recall F = ma).  If the camera was
# not moving, then the only source of acceleration is gravity, which by convention points
# down.  If we could measure down, the roll and pitch are perpendicular.  The challenge is
# that accel also measure camera forces, and is quite noisy.
#
# A complementary filter combines these two sensors into a single estimate.
#
# Note, while we want to correct for pitch and roll, we do not want the camera to track
# yaw.  Huh?  The horizon should be horizontal, even if the camera is mounted at a canted
# angle, or if the sailboat is heeled over.  But the forward view of the camera should
# track as the vehicle (boat) steers around the race course (for example, you are always
# looking along the direction of boat travel rather at a particular compass heading).
# Note, neither the gyro nor accel measure yaw (i.e. compass orientation).  This is not
# needed for camera stabilization, since we want the camera to "look forward" (often in
# the direction of motion) even if it is rotating the world.  Yaw could be estimated by
# integrating the onboard compass.

def down_plot(down, fignum=None, clf=False):
    fig = plt.figure(num=fignum)
    if clf:
        fig.clf()
    ax = fig.add_subplot(111)    
    # Plot angles vs. various orientations on the camera.
    dd = np.dot(down, ihat)
    ax.plot(np.degrees(np.arccos(dd)), 'r')
    dd = np.dot(down, jhat)
    ax.plot(np.degrees(np.arccos(dd)), 'g')
    dd = np.dot(down, khat)
    ax.plot(np.degrees(np.arccos(dd)), 'b')


# Down points away from the direction of acceleration.  
down = -1 * np.vstack([imu_df.accel_x, imu_df.accel_y, imu_df.accel_z]).T
# Normalize, since we care only about orientation of gravity.
down = down / np.linalg.norm(down, axis=1, keepdims=True)

gyro = np.radians(np.vstack([imu_df.gyro_x, imu_df.gyro_y, imu_df.gyro_z]).T)

delta_t = 0.004

qest = geo.q_identity()
poses = []

i = 0
for i in range(down.shape[0]):
    # map the camera local khat to estimated k
    pose_est = np.dot(geo.q_mat(qest), pose)
    k_est = pose_est[:,2]
    j_est = pose_est[:,1]


    if True:
        # compute the rotation error from down
        tw_error = np.cross(k_est, down[i])
        q_update = geo.twist_q(0.02 * tw_error)
        qest = geo.q_compose(q_update, qest)

        # compute the rotation error from "look forward", where forward is measured after
        # rotation to correct for down.

        # Get rid of the component of j that is perpendicular to the k, this leaves a vector
        # the plane of zero elevation.  Thus, rotation will not effect elevation.
        j_flat = j_est - np.dot(j_est, khat) * khat

        tw_error = np.cross(j_flat, jhat)
        q_update = geo.twist_q(0.02 * tw_error)
        qest = geo.q_compose(q_update, qest)

    if True:
        # Apply the gyro rotations, around each axis.
        qest = geo.q_compose(geo.axis_angle_q(ihat, -delta_t * gyro[i, 0]), qest)
        qest = geo.q_compose(geo.axis_angle_q(jhat, -delta_t * gyro[i, 1]), qest)
        qest = geo.q_compose(geo.axis_angle_q(khat, -delta_t * gyro[i, 2]), qest)

    poses.append(qest)
    if i % 1000 == 0:
        print(i, np.linalg.norm(tw_error))

pmats = [geo.q_mat(p) for p in poses]
down_est = -1 * np.array([p[:,2] for p in pmats])

down_plot(down_est, fignum=1)
plt.close('all')

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)


zero = np.zeros((3))

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, projection='3d')

for qest in it.islice(poses, 0, None, 1000):
    pose_est = np.dot(geo.q_mat(qest), pose)
    for c, row in zip('r g b'.split(), pose_est):
        coords = [[z, r] for z, r in zip(zero, row)]
        a = Arrow3D(*coords, mutation_scale=20, 
                    lw=3, arrowstyle="-|>", color=c)
        ax.add_artist(a)

plt.draw()
plt.show()


################################################################

def raw_iterator(data):
    deconstruct = "d" * 11
    for row_data in struct.iter_unpack(deconstruct, data):
        r = Row()
        r.ts = row_data[0]
        r.gyro_x = row_data[1]
        r.gyro_y = row_data[2]
        r.gyro_z = row_data[3]
        r.accel_x = row_data[4]
        r.accel_y = row_data[5]
        r.accel_z = row_data[6]
        r.q = geo.xyzw_q( row_data[8], row_data[9], row_data[10], row_data[7])
        yield r.__dict__

def fit_iterator(data):
    deconstruct = "d" * 5
    for row_data in struct.iter_unpack(deconstruct, data):
        r = Row()
        r.ts = row_data[0]
        r.q = geo.xyzw_q( row_data[2], row_data[3], row_data[4], row_data[1])
        yield r.__dict__

        
class Row:
    pass

if True:
    cpp_file = "../../FitFile/decode/eigen.bin"

    with open(cpp_file, mode='rb') as dfile:  # b is important -> binary
        data = dfile.read()

    import struct
    nrow, ncol = struct.unpack("qq", data[:16])
    print(nrow, ncol)
    
    offset = 16

    cpp_df = pd.DataFrame(list(raw_iterator(data[offset:offset+(11*8*10000)])))

    # pmats = [geo.q_mat(geo.q_inverse(q)) for q in list(cpp_df.q)]
    pmats = [geo.q_mat(q) for q in list(cpp_df.q)]
    dest = -1 * np.array([p[:,2] for p in pmats])

    cpp_df['est_x'] = dest[:,0]
    cpp_df['est_y'] = dest[:,1]
    cpp_df['est_z'] = dest[:,2]


    plt.close('all')

    plot_sensors(cpp_df, "accel_x accel_y accel_z".split())
    plot_sensors(cpp_df, "est_x est_y est_z".split())

    
    plt.close('all')
    down_plot(down_est, fignum=1)
    down_plot(dest, fignum=10, clf=True)


if True:
    cpp_file = "/Users/viola/www/Poses/2020-07-11T10_23_36.061.bin"

    with open(cpp_file, mode='rb') as dfile: # b is important -> binary
        data = dfile.read()

    import struct
    nrow, ncol = struct.unpack("qq", data[:16])
    print(nrow, ncol)
    
    offset = 16

    # cpp_df = pd.DataFrame(list(cpp_iterator(data[offset:]))).iloc[100:-10]

    # cpp_df = pd.DataFrame(list(fit_iterator(data[offset:]))).iloc[100:-10]
    cpp_df = pd.DataFrame(list(fit_iterator(data[offset:])))
    
    pmats = [geo.q_mat(q) for q in list(cpp_df.q)]
    dest = -1 * np.array([p[:,2] for p in pmats])


    plt.close('all')
    down_plot(down_est, fignum=1)
    down_plot(dest, fignum=10, clf=True)
