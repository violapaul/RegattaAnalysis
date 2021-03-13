"""
Handy routines for dealing with GPX files.
"""

import os
import itertools as it
import arrow

import gpxpy
import gpxpy.gpx

def write_gpx(gpx, gpx_output_file):
    with open(gpx_output_file, 'w') as gpx_fs:
        gpx_fs.write(gpx.to_xml())


def read_gpx(gpx_path):
    with open(gpx_path, 'r') as gpx_file:
        gpx_obj = gpxpy.parse(gpx_file)
    return gpx_obj


def gpx_datetimes(gpx_obj, max_count=10, skip=0, minutes=False):
    for i, track in zip(it.count(), gpx_obj.tracks):
        print("Track {}".format(i))
        for j, segment in zip(it.count(), track.segments):
            print("   Segment {}", j)
            for k, point in enumerate(it.islice(segment.points, skip, skip+max_count)):
                gps_time = arrow.get(point.time)
                local = gps_time.to('US/Pacific')
                if k == 0:
                    print(f"First time {local}")
            print(f"Last time {local}")


def point_local_time(point):
    "Return the local time of the trackpiont."
    gps_time = arrow.get(point.time)
    return gps_time.to('US/Pacific')


def gpx_date_chop(gpx_obj, max_count=10000, skip=0, step=1):
    """
    Chop a gpx object into 'days'.  Return a list of points for each day.

    Assume there is a single track and segment.
    """
    tracks = gpx_obj.tracks
    if len(tracks) > 1:
        raise Exception(f"More than one track: {len(tracks)}")
    segments = tracks[0].segments
    if len(segments) > 1:
        raise Exception(f"More than one segment: {len(segments)}")
    current_date = None
    days = []
    date_points = []
    # When the date changes then craete a new day.
    for k, point in enumerate(it.islice(segments[0].points, skip, skip+max_count)):
        local = point_local_time(point)
        point_date = arrow.get(local.date())
        if current_date is None:
            current_date = point_date
            print(f"First date is {current_date}")
        else:
            if current_date == point_date:
                date_points.append(point)
            else:
                current_date = point_date
                print(f"New date is {current_date}")
                days.append(date_points)
                date_points = []
    days.append(date_points)
    return days


def gpx_from_points(points):
    """
    Create a GPX object from a list of GPXTrackPoint's.  Contains a single track and a
    single segment.
    """
    # Empty object, add track, and segment.
    gpx_out = gpxpy.gpx.GPX()  
    gpx_track = gpxpy.gpx.GPXTrack()
    gpx_out.tracks.append(gpx_track)
    gpx_segment = gpxpy.gpx.GPXTrackSegment()
    gpx_track.segments.append(gpx_segment)
    # Add points to segment
    for p in points:
        gpx_segment.points.append(p)
    return gpx_out


def gpx_split_dates(gpx_path):
    """
    Given a GPX file that tracks overnight, split into different days.
    
    GPX files that cross midnight seem to screw up RaceQs.
    """
    directory, gpx_file = os.path.split(gpx_path)
    ggg = read_gpx(gpx_path)
    days = gpx_date_chop(ggg, max_count=10000000)
    for d in days:
        first_point = d[0]
        name = point_local_time(first_point).format('YYYY-MM-DD_HH:mm')
        gpx = gpx_from_points(d)
        new_path = os.path.join(directory, f"{name}.gpx")
        print(f"Writing {new_path} with {len(d)} points.")
        write_gpx(gpx, new_path)


def test():
    gpx_split_dates("/Users/viola/Downloads/night.gpx")
