

import datetime
import utils

short = 11
long = 18

secs_per_mile = 550  # at PHRF 0, more conventionaly listed at 650 secs per mile at PHRF
                     # 100.

def time_on_course(phrf, length):
    corrected = secs_per_mile + phrf
    return corrected * length


def start_time(phrf, course_length, finish_time):
    ft = utils.time_from_string(f"2020-11-01 {finish_time}")
    st = ft - datetime.timedelta(seconds=time_on_course(phrf, course_length))
    print(f"PHRF: {phrf} on course: {course_length} nm, start time is {st.format('HH:mm:ss')}")



start_time(phrf=45, course_length=18, finish_time="14:30:00")
start_time(phrf=250, course_length=11, finish_time="14:30:00")
