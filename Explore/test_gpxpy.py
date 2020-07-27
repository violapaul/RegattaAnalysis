
import gpxpy
import gpxpy.gpx
import itertools as it
import arrow
import gpx


def print_gpx(gpx_obj, max_count=10, skip=0, minutes=False):
    for i, track in zip(it.count(), gpx_obj.tracks):
        print("Track {}".format(i))
        for j, segment in zip(it.count(), track.segments):
            print("   Segment {}", j)
            for k, point in zip(it.count(), it.islice(segment.points, skip, skip+max_count)):
                print(point)
                gps_time = arrow.get(point.time)
                local = gps_time.to('US/Pacific')
                if minutes:
                    deg, min = gpx.degrees_to_degrees_minutes(point.latitude)
                    lat = "{0} {1}".format(deg, min)
                    deg, min = gpx.degrees_to_degrees_minutes(point.longitude)
                    lon = "{0} {1}".format(deg, min)
                else:
                    lat = point.latitude
                    lon = point.longitude
                print('       Point at T={3} ({0},{1}) -> {2}'.format(lat, lon,
                                                                      point.elevation,
                                                                      local.format('YYYY-MM-DD HH:mm:ss')))

def print_gpx_file(gpx_path, max_count=10):
    with open(gpx_path, 'r') as gpx_file:
        gpx_obj = gpxpy.parse(gpx_file)
        print_gpx(gpx_obj, max_count=max_count)


def gpx_sample(gpx_obj, max_count=10, skip=0, step=1):
    "Given a sequence of lat/lon/alt records, create a GPX datastructure."
    gpx_out = gpxpy.gpx.GPX()  # Empty

    for i, track in zip(it.count(), gpx_obj.tracks):
        print("Track {}".format(i))
        gpx_track = gpxpy.gpx.GPXTrack()
        gpx_out.tracks.append(gpx_track)

        for j, segment in zip(it.count(), track.segments):
            print("   Segment {}", j)
            # Create first segment in our GPX track:
            gpx_segment = gpxpy.gpx.GPXTrackSegment()
            gpx_track.segments.append(gpx_segment)

            for k, point in zip(it.count(), it.islice(segment.points, skip, skip+max_count, step)):
                new_point = gpxpy.gpx.GPXTrackPoint(
                    latitude = point.latitude,
                    longitude = point.longitude,
                    elevation = point.elevation,
                    time = point.time)
                gpx_segment.points.append(new_point)
    return gpx_out

def write_gpx(gpx, gpx_output_file):
    with open(gpx_output_file, 'w') as gpx_fs:
        gpx_fs.write(gpx.to_xml())
    

def test_read():
    gpx_path = '/Users/viola/Downloads/activity_4128691681.gpx'
    gpx_path = '/Users/viola/Downloads/Navionics-archive-export.gpx'
    gpx_path = '/Users/viola/Downloads/sara.gpx'
    with open(gpx_path, 'r') as gpx_file:
        gpx_obj = gpxpy.parse(gpx_file)
    print_gpx(gpx_obj, 10, 1000)


write_gpx(ggg, "/tmp/foo.gpx")
