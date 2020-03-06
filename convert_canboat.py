#!/usr/bin/env python

import os
import os.path
import glob
import subprocess
import itertools as it
import multiprocessing

import gpxpy
import gpxpy.gpx

import canboat as cb
import global_variables as G
import utils


description = [
    "Convert a raw canboat log to a json file and then both to a GPX and a PANDAS pickle",
    "file.  Extract date/time along the way and use it for the filenames.  Moves the log",
    "files to the OldLogs directory, and the JSON files to the JSON directory."
]

description = "\n".join(description)

def canboat_to_json(raw_file, trim=100):
    """
    Converts the logs captured with actisense-serial canboat software on the raspberry PI
    and converts it to JSON.  Names the file with the date and time logged.
    """
    path, file = os.path.split(raw_file)
    base, extension = os.path.splitext(file)
    json_file = os.path.join(path, base + ".json")
    if extension.casefold() == '.gz':
        command = "zcat {0} | analyzer -json > {1}".format(raw_file, json_file)
    else:
        command = "cat {0} | analyzer -json > {1}".format(raw_file, json_file)
    print("Running ", command)
    subprocess.run(command, shell=True)

    records = cb.matching_records( cb.file_lines(json_file),
                                   cb.pgn_filter(G.PGN_WHITELIST), 1)
    # Throw out the first trim records, just to make sure things are working
    records = it.islice(records, trim, None)

    gps_time = cb.log_gpstime(records)

    local_time = gps_time.to('US/Pacific')
    new_name = local_time.format('YYYY-MM-DD_HH:mm') + ".json"
    new_file = os.path.join(path, new_name)
    os.rename(json_file, new_file)


def json_to_pandas_helper(json_file, pandas_file, count, trim):
    "Convert the JSON file version of the canboat log to a pandas dataframe."
    print(f"Converting {json_file} to {pandas_file}.")
    df = cb.json_to_data_frame(json_file, count=count, trim=trim)
    df.to_pickle(pandas_file)
    print(f"Done converting {json_file}.")


def json_to_pandas(json_file, pandas_file, count, trim):
    "Convert the JSON file version of the canboat log to a pandas dataframe."
    p = multiprocessing.Process(target=json_to_pandas_helper, args=(json_file, pandas_file, count, trim))
    p.start()
    return p


def lla_to_gpx(lla_records, stop=None):
    gpx = gpxpy.gpx.GPX()

    # Create first track in our GPX:
    gpx_track = gpxpy.gpx.GPXTrack()
    gpx.tracks.append(gpx_track)

    # Create first segment in our GPX track:
    gpx_segment = gpxpy.gpx.GPXTrackSegment()
    gpx_track.segments.append(gpx_segment)

    for gpx_data in lla_records:
        gpx_segment.points.append(
            gpxpy.gpx.GPXTrackPoint(
                gpx_data['lat'],
                gpx_data['lon'],
                elevation=gpx_data['alt'],
                time=gpx_data['timestamp'].datetime))
    return gpx


def json_to_gpx(json_file, gpx_output_file, skip=3, src=5):
    print("Reading canboat JSON log: {0}".format(json_file))

    lla_records = it.islice(cb.lla_records(json_file, src=src), 0, None, skip)
    gpx = lla_to_gpx(lla_records, stop=None)

    print("Writing GPX result to {0}".format(gpx_output_file))
    with open(gpx_output_file, 'w') as gpx_fs:
        gpx_fs.write(gpx.to_xml())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--trim", help="skip this many records before extracting time", type=int, default=100)
    parser.add_argument("--count", help="maximum number of records to process", type=int, default=10000000)
    parser.add_argument('--json', help='Convert canboat logs to json.', action='store_true', default=True)
    parser.add_argument('--gpx', help='Convert canboat json to GPX files.', action='store_true', default=True)
    parser.add_argument('--pandas', help='Convert canboat json to pandas dataframes.', action='store_true', default=True)
    parser.add_argument('--cleanup', help='Cleanup raw files.', action='store_true', default=True)
    args = parser.parse_args()

    cwd = os.path.realpath(os.getcwd())

    log_files = glob.glob("actisense*.log") + glob.glob("actisense*.log.gz")

    if args.json:
        # We are going to squirrel away the old logs.
        for file in log_files:
            print("Converting log {0} to JSON".format(file))
            log_path = os.path.join(cwd, file)
            canboat_to_json(log_path, trim=args.trim)

    json_files = glob.glob("*.json")
    gpx_files = utils.extract_base_names(glob.glob("*.gpx"))
    pandas_files = utils.extract_base_names(glob.glob("*.pd"))

    if args.gpx:
        print("Converting JSON files to GPX.")
        for jfile in json_files:
            base = utils.file_base_name(jfile)
            if base in gpx_files:
                print(f"Skipping {jfile} since GPX file already exists.")
            else:
                print(f"Converting {jfile} to GPX.")
                json_to_gpx(os.path.join(cwd, jfile), os.path.join(cwd, base + ".gpx"))

    if args.pandas:
        print("Converting JSON files to PANDAS.")
        threads = []
        for jfile in json_files:
            base = utils.file_base_name(jfile)
            if base in pandas_files:
                print(f"Skipping {jfile} since PANDAS file already exists.")
            else:
                j_fullpath = os.path.join(cwd, jfile)
                pd_fullpath = os.path.join(cwd, base + ".pd")
                threads.append(json_to_pandas(j_fullpath, pd_fullpath, args.count, args.trim))
        for thread in threads:
            thread.join()
        print(f"Done creating {len(json_files)} pandas dataframes!")

    if args.cleanup:
        print("Cleaning up raw files.")
        old_logs_dir = os.path.join(cwd, "OldLogs")
        utils.ensure_directory(old_logs_dir)
        for file in log_files:
            log_path = os.path.join(cwd, file)
            os.rename(log_path, os.path.join(old_logs_dir, file))
        json_dir = os.path.join(cwd, "JSON")
        utils.ensure_directory(json_dir)
        for jfile in json_files:
            j_fullpath = os.path.join(cwd, jfile)
            os.rename(j_fullpath, os.path.join(json_dir, jfile))
