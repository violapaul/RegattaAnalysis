
#### Cell #0 Type: module ######################################################

# These are libraries written for RaceAnalysis
from global_variables import G
from nbutils import display_markdown, display

G.init_seattle(logging_level="INFO")

import itertools as it
import pandas as pd
import numpy as np

import race_logs
import process as p
import analysis as a
import chart as c

# This is the python-fitparse library
from fitparse import FitFile

# ff = FitFile('Data/Virb360Fit/2020-04-09-21-41-03.fit')
# ff = FitFile('/Volumes/Big/Virb/GMetrix/2020-07-11-09-27-15.fit')
ff = FitFile('Data/Virb360Fit/2020-05-06-18-32-22.fit')

#####

NTSC_FRAME_RATE = 29.97 # fps

VIRB_VIDEO =



#### Cell #1 Type: module ######################################################

def get_message(fit_file, message_type=None):
    return next(ff.get_messages(message_type))

def print_messages(fit_file, message_type, count):
    messages = it.islice(fit_file.get_messages(message_type), 0, count, None)
    for m in messages:
        print(repr(m))

def before_and_after(fit_file, message_type):
    before = None
    message = None
    after = None
    count = 0
    for m in fit_file.get_messages():
        if count % 10000 == 0:
            G.logger.info(count)
            G.logger.info(m.mesg_type)            
        count += 1
        if m.mesg_type is not None and m.mesg_type.name == message_type:
            G.logger.info(f"Found message {m}")
            message = m
        elif message is not None:
            return before, message, m
        else:
            before = m
    return None

def show_values(fit_file, field, start, count):
    for m in it.islice(fit_file.get_messages(), start, count, None):
        yield m.mesg_type, m.get_value(field)

def my_map(seq, key_func):
    dd = {}
    for i, m in enumerate(seq):
        if i % 10000 == 0:
            print(i)
        key = key_func(m)
        mlist = dd.get(key, [])
        mlist.append(m)
        dd[key] = mlist
    return dd

def my_reduce(dict_map, reduce_func):
    res = {}
    for k, v in dict_map.items():
        res[k] = reduce_func(v)
    return res
        

def mesg_name(mesg):
    if mesg.mesg_type is not None:
        return mesg.mesg_type.name
    else:
        return mesg.mesg_type

dd = my_map(it.islice(ff.get_messages(), 0, 1000000, None), mesg_name)

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = it.tee(iterable)
    next(b, None)
    return zip(a, b)

def block_lengths(messages):
    for m0, m1 in pairwise(messages):
        t0 = m0.get_value('timestamp') * 1000 + m0.get_value('timestamp_ms')
        t1 = m1.get_value('timestamp') * 1000 + m1.get_value('timestamp_ms')
        yield t1 - t0

def milliseconds_to_mins_seconds(m):
    fmins = (m / 1000) / 60
    mins = np.floor(fmins)
    secs = np.round((fmins - mins) * 60)
    return mins, secs

list(block_lengths(dd['camera_event']))

def get_gyro
        



pmessages = it.islice(ff.get_messages('file_id'), 0, 10, None)
file_id = next(messages).get_values()
print(file_id)

dt = file_id['time_created']
print(dt)

dd = {}
for i, m in :
    if i % 10000 == 0:
        print(i)
    mlist = dd.get(m.mesg_type, [])
    mlist.append(m)
    dd[m.mesg_type] = mlist

for k, v in dd.items():
    print(k, len(v))
    print(repr(v[0]))
    


#### Cell #2 Type: module ######################################################

print(dir(m))
m.mesg_type
m.fields

#### Cell #3 Type: module ######################################################

help(ff.get_messages)

#### Cell #4 Type: module ######################################################


messages = list(it.islice(ff.get_messages('camera_event'), 0, 2, None))

dfs = []
rows = [m.get_values() for m in messages]
dfs.append(pd.DataFrame(rows))

df = dfs[0]

df
    

#### Cell #5 Type: module ######################################################


messages = list(it.islice(ff.get_messages('gyroscope_data'), 0, 2, None))

dfs = []
rows = [m.get_values() for m in messages]
dfs.append(pd.DataFrame(rows))

df = dfs[0]

df
    

#### Cell #6 Type: module ######################################################



#### Cell #7 Type: metadata ####################################################

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
#:     "timestamp": "2020-07-19T17:25:21.291706-07:00"
#:   },
#:   "nbformat": 4,
#:   "nbformat_minor": 2
#: }

#### Cell #8 Type: finish ######################################################

