#!/usr/bin/env python

import os
import shutil
import os.path
import subprocess
import itertools as it
import re

import concurrent.futures
import logging

import global_variables
G = global_variables.init_seattle()

import canboat as cb

# Remove all handlers associated with the root logger object.
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

description = [
    "Convert a raw canboat log to a json file and then both to a GPX and a PANDAS pickle",
    "file.  Extract date/time along the way and use it for the filenames."
]

description = "\n".join(description)

################ Generic helpers: copying, finding and modifying files  ################

def find_matching_files(directory, regex_pattern):
    "Find the matching files in a directory."
    regex = re.compile(regex_pattern)
    files = os.listdir(directory)
    return [f for f in files if regex.match(f)]

def copy_with_timestamp(from_path, to_path):
    "Copy a file and ensure the timestamps are the same."
    # Not sure why but copy2 does not always preserve times.
    logging.debug(f"Copying {from_path} TO {to_path}.")
    shutil.copy2(from_path, to_path)
    atime = os.path.getatime(from_path)
    mtime = os.path.getmtime(from_path)
    os.utime(to_path, times=(atime, mtime))

def copy_all_files(from_dir, to_dir, regex_pattern):
    "Copy all files that match regex."
    files = find_matching_files(from_dir, regex_pattern)
    logging.info(f"Found {len(files)}.")
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
    return find_matching_files(G.GPX_LOGS_DIRECTORY, r'^.*$')

def pandas_files():
    return find_matching_files(G.PANDAS_LOGS_DIRECTORY, r'^.*$')

def copy_from_usb(filename):
    copy_with_timestamp(usb_log_path(filename), compressed_log_path(filename))

################ Application operations. ################

def copy_err_files_from_usb():
    "Copy all the log error files from the USB drive."
    logging.info("Copying error files.")
    copy_all_files(G.USB_LOGS_DIRECTORY, G.COMPRESSED_LOGS_DIRECTORY, r".*\.err")

def copy_log_files_from_usb():
    "Copy all logs files which have not yet been copied from the USB drive to the compressed file staging area."
    current_logs = compressed_log_files() + uncompressed_log_files()
    logging.info(f"Found {len(current_logs)} current log files.")
    usb_logs = usb_log_files()
    logging.info(f"Found {len(usb_logs)} usb log files.")
    log_basenames = [log_basename(clog) for clog in current_logs]
    for ulog in usb_logs:
        if log_basename(ulog) not in log_basenames:
            logging.debug(f"Missing log {ulog}, copying.")
            copy_from_usb(ulog)

def compress_log(filename):
    path = compressed_log_path(filename)
    logging.info(f"Compressing {path}")
    subprocess.run(f"gzip {path}", shell=True)

def create_compressed_log_files():
    current_logs = uncompressed_log_files()
    logging.info(f"Found {len(current_logs)} uncompressed log files.")
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
    logging.debug(f"Linking {full_path_link} to {full_path_destination}")
    os.symlink(os.path.relpath(full_path_destination, link_dir), full_path_link)

def link_named_log_file(compressed_filename):
    """
    " Starting with a compressed log file, compute the true datetime, convert to a
    filename, and then link to the original with this new filename.
    """
    logging.info(f'Linking {compressed_filename}')
    compressed_path = compressed_log_path(compressed_filename)

    tmp_path = "/tmp"
    json_file = os.path.join(tmp_path, compressed_filename + ".json")
    log_file = os.path.join(tmp_path, compressed_filename + ".stderr")

    # Extract the head...  just need the first record.
    # TODO: fix constant
    command = f"zcat {compressed_path} | analyzer -json 2> {log_file} | head -10000 > {json_file}"
    logging.info(f'Running "{command}"')
    subprocess.run(command, shell=True)

    named_filename = extract_log_name(json_file)
    named_path = named_log_path(named_filename)
    logging.info("Linking {compressed_filename} to {named_path}")
    symlink(compressed_path, named_path)

def create_named_log_files():
    "Find the compressed files that are not yet named and linked into the named directory."
    clogs = set(compressed_log_files())
    nlogs = named_log_files()
    logging.info(f"Found {len(clogs)} compressed log files, {len(nlogs)} named log files.")
    for nlog in nlogs:
        target = os.path.realpath(named_log_path(nlog))
        _, name = os.path.split(target)
        logging.debug(f"{nlog} is linked to {name}, removing.")
        clogs.discard(name)
    logging.info(f"Found {len(clogs)} unlinked files.")
    for f in list(clogs):
        if not is_valuable_compressed_log(f):
            clogs.discard(f)
    logging.info(f"Found {len(clogs)} unlinked files are valuable.")
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

def convert_named_log_to_gpx_and_pandas_file(named_file, count, trim):
    """
    Given a named log file (compressed JSON) will convert to GPX and/or PANDAS (if missing).
    """
    logging.info(f"{named_file}: Examining")
    named_path = named_log_path(named_file)
    basename = log_basename(named_file)

    gpx_filename = basename + ".gpx"
    g_path = gpx_path(gpx_filename)
    pandas_filename = basename + ".pd"
    compressed_pandas_filename = basename + ".pd.gz"
    cp_path = pandas_path(compressed_pandas_filename)
    p_path = pandas_path(pandas_filename)

    if os.path.exists(g_path) and os.path.exists(cp_path):
        logging.debug("{named_file}: GPX and Pandas logs exist. Skipping.")
        return f"{named_file}: Skipped"

    tmp_path = "/tmp"
    json_path = os.path.join(tmp_path, basename + ".json")
    log_path = os.path.join(tmp_path, basename + ".stderr")

    command = f"zcat {named_path} | analyzer -json 2> {log_path} > {json_path}"
    logging.debug(f"{named_file}: Creating JSON file.")
    logging.debug(f'Running "{command}"')
    subprocess.run(command, shell=True)

    if not os.path.exists(g_path):
        logging.debug(f"{named_file}: Creating GPX file.")
        json_to_gpx(json_path, g_path)

    if not os.path.exists(cp_path):
        logging.info(f"{named_file}: Creating Pandas file.")
        json_to_pandas(json_path, p_path, count, trim)
        logging.debug(f"{named_file}: Compressing {p_path}")
        subprocess.run(f"gzip {p_path}", shell=True)

    return f"{named_file}: Processed"

def create_gpx_and_pandas_files(count, trim):
    "Examines all named log files and computes GPX/PANDAS if necessary."
    nlogs = named_log_files()
    logging.info(f"Found {len(nlogs)} named log files.")
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as ex:
        res = list(ex.map(convert_named_log_to_gpx_and_pandas_file, nlogs, it.cycle([count]), it.cycle([trim])))
    return res

def process_all():
    if usb_drive_available():
        copy_err_files_from_usb()
        copy_log_files_from_usb()
    else:
        logging.warning(f"USB drive not found.  Skipping copy.")
    create_compressed_log_files()
    create_named_log_files()
    create_gpx_and_pandas_files(100000000, 100)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--log", help="Logging level", type=str, default='warning')
    args = parser.parse_args()

    numeric_level = getattr(logging, args.log.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {args.log}")

    logging.basicConfig(level=numeric_level,
                        format='%(asctime)s|%(levelname)s|%(funcName)s| %(message)s')

    process_all()
