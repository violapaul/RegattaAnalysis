"""
Routines for processing the canboat JSON files to produce pandas dataframes in a
standardized format.

The canboat stream is asynchronous and contains multiple fields from multiple sendors.

The dataframe is constructed so that the signals appear synchronous where all the fields
have been flattened in to a single record.  This introduces redundancy and it removes
specific information about arrival times.  But the result is a square (time vs. field)
matrix of data where columns can be compared and combined.  This facilitates plotting and
analysis.

Main functions:

    Filtering out uninteresting records.

    Deconflicting names.  Each canboat record has a PGN and some fields.  In some cases
    the semantics of the field depends on the PGN, and in others it does not.  In other
    cases the semantics of the field name depends on another field (like true vs. relative
    wind speed).  And in other cases there can be multiple senders of the same information
    and these need to be exploded into different fields.

    Normalization of data.
"""
import json
import itertools as it
import arrow
import pandas as pd
import copy
from global_variables import G

def pgn_code(name):
    # This is a real mouthful when a dict would do...  let's go with it for a while.
    try:
        return G.PGNS[G.PGNS['Description'] == name].id.values[0]
    except Exception:
        print(f"pgn_code: {name} not found.")
        return None


def device_code(name):
    # This is a real mouthful when a dict would do...  let's go with it for a while.
    try:
        return G.DEVICES[G.DEVICES['Device'] == name].src.values[0]
    except Exception:
        return None


def substring_matcher(substring_to_match):
    "Return a function that returns True if a string contains to_match."
    return lambda s: substring_to_match in s


def dict_matcher(**match_dict):
    "Returns a function which returns True dict has all key value pairs, exact match."
    def helper(j):
        for key in match_dict:
            if j[key] != match_dict[key]:
                return False
        return True
    return helper


def pgn_filter(pgn_list):
    "Return True if the canboat record matches one of these PGNs."
    pgn_set = set(pgn_list)

    def helper(record):
        return record['pgn'] in pgn_set

    return helper


def matching_file_lines(path, line_matcher=None):
    "Generate the lines of a file if and only if the LINE_MATCHER function returns True."
    with open(path) as lines:
        for line in lines:
            if (line_matcher is None) or line_matcher(line):
                yield line


def matching_records(json_lines, json_matcher=None, line_skip=1):
    "Return parsed json for each line, optionally json_matcher must return True, and optionally skipping lines."
    for line in it.islice(json_lines, 0, None, line_skip):
        record = json.loads(line)
        if (json_matcher is None) or json_matcher(record):
            yield record


def convert_gnss_date_time(gnss_date, gnss_time):
    # canboat time is in a funny format
    date = gnss_date.replace(".", "-")
    time = gnss_time
    return arrow.get(date + "T" + time)


def gnss_convert(record):
    "Convert to a GNSS record, with lat, lon, alt."
    datetime = convert_gnss_date_time(record['fields']['Date'],
                                      record['fields']['Time'])
    return dict(timestamp = datetime,
                lat = float(record['fields']['Latitude']),
                lon = float(record['fields']['Longitude']),
                alt = float(record['fields']['Altitude']))


def valid_gnss_record(record, src):
    correct_src = record['src'] == src
    fields = record['fields']
    has_date_time = ('Date' in fields) and ('Time' in fields)
    has_latlonalt = ('Latitude' in fields) and ('Longitude' in fields) and ('Altitude' in fields)
    return correct_src and has_date_time and has_latlonalt


def lla_records(json_log_path, src=5):
    "Construct a sequence of lat/lon/alt GNSS records from from a JSON log file."
    return map( gnss_convert,
                matching_records(
                    # Return online lines which contain GNSS position data.
                    matching_file_lines(json_log_path,
                                        substring_matcher('GNSS Position Data')),
                    # And a valid gnss record
                    lambda record: valid_gnss_record(record, src=src),
                    line_skip=1))


def set_of_pgns(record_generator, line_count=500000):
    # DEBUG
    pgn_dict = {}
    for record in it.islice(record_generator, 0, line_count):
        pgn = record['pgn']
        src = record['src']
        pgn_dict[(src, pgn)] = record
    return pgn_dict


def print_pgns(pgn_dict):
    # DEBUG    
    for key in pgn_dict:
        print(key, pgn_dict[key]['description'])


def flatten_pgns(record_generator, line_count=500000):
    # DEBUG
    pgn_dict = {}
    for record in it.islice(record_generator, 0, line_count):
        pgn = record['pgn']
        src = record['src']
        fields = record['fields']
        fields['timestamp'] = record.get('timestamp', None)
        fields['description'] = record.get('description', None)
        pgn_dict[(src, pgn)] = record
        yield pgn_dict


def log_gpstime(records, retries=100):
    "Find the first System Time record and return its datetime."
    time_pgn = pgn_code('System Time')

    def match_system_time(record):
        return record['pgn'] == time_pgn and record['src'] == 5

    # During initialization the records may be missing date/time data.
    for i in range(retries):
        system_time_record = next(filter(match_system_time, records))
        print(system_time_record)
        fields = system_time_record['fields']
        if 'Date' in fields and 'Time' in fields:
            return convert_gnss_date_time(fields['Date'],
                                          fields['Time'])
    raise Exception("Could not find record to extract Date/Time from.")


def time_records(records, src=device_code("ZG100 Antenna")):
    time_pgn = pgn_code('System Time')

    for record in records:
        if record['pgn'] == time_pgn and record['src'] == src:
            record['gps_time'] = convert_gnss_date_time(record['fields']['Date'],
                                                        record['fields']['Time'])
            record['pi_time'] = arrow.get(record['timestamp'])
            record['delta_time'] = record['gps_time'] - record['pi_time']
            yield record


def gnss_records(records, src=device_code("ZG100 Antenna")):
    code = pgn_code('GNSS Position Data')

    for record in records:
        if record['pgn'] == code and record['src'] == src:
            record['gps_time'] = convert_gnss_date_time(record['fields']['Date'],
                                                        record['fields']['Time'])
            record['pi_time'] = arrow.get(record['timestamp'])
            record['delta_time'] = record['gps_time'] - record['pi_time']
            yield record


# Clean up and filter records ################
PGN_WIND_DATA = pgn_code("Wind Data")
PGN_RUDDER = pgn_code("Rudder")
PGN_DEPTH = pgn_code('Water Depth')
PGN_RATE_OF_TURN = pgn_code('Rate of Turn')
DEVICE_ZEUS_IGPS = device_code("Zeus iGPS")
DEVICE_ZG100_COMPASS = device_code("ZG100 Compass")
DEVICE_ZG100_ANTENNA = device_code("ZG100 Antenna")

def remove_record(record):
    return False


def canonical_field_name(field_name):
    "Downcase and underscore separate a field name.  E.G. 'Wind Angle' -> wind_angle"
    return "_".join(map(lambda s: s.casefold(), field_name.split()))


def prefix_field_names(record, prefix):
    fields = record.get('fields', None)
    if fields is not None:
        new_fields = {}
        for key in fields:
            new_fields[prefix + key] = fields[key]
        record['fields'] = new_fields


def transform_record(record):
    "Transform the canboat record to resolve ambiguities in field names."
    # BOAT 
    # Wind Data can be True or Apparent, and then collide.  Deconflict
    if record['pgn'] == PGN_WIND_DATA:
        fields = record.get('fields', None)
        if fields is not None:
            if "True (boat referenced)" in fields.get('Reference', ""):
                prefix_field_names(record, 'True ')
            if "True (ground referenced to North)" in fields.get('Reference', ""):
                prefix_field_names(record, 'True North ')

    if record['pgn'] == PGN_RUDDER:
        prefix_field_names(record, "Rudder ")
    if record['pgn'] == PGN_RATE_OF_TURN:
        prefix_field_names(record, "Turn ")

    if record['src'] == DEVICE_ZG100_COMPASS:
        prefix_field_names(record, "ZG100 ")
    elif record['src'] == DEVICE_ZEUS_IGPS:
        prefix_field_names(record, "ZEUS ")

    fields = record.get('fields', [])
    flat_fields = {}
    for key in fields:
        flat_fields[canonical_field_name(key)] = fields[key]
    if record['pgn'] == PGN_DEPTH:
        if 'depth' not in flat_fields.keys():
            flat_fields['depth'] = 99
    record['fields'] = flat_fields
    return record


def json_to_data_frame(json_file, count=1000000, trim=100, rhythm=0.1):
    """
    Main function.  Reads a json file and produces a dataframe.
       COUNT  : max number of records to process *after* filtering by PGNS
       TRIM   : trim off the first N records,  since the beginning can be wonky
       RHYTHM : Rate at which records are generated.
    """
    records = matching_records( matching_file_lines(json_file), pgn_filter(G.PGN_WHITELIST), 1)
    # Throw out the first trim records, just to make sure things are working
    records = it.islice(records, trim, None)

    base_time = arrow.get(next(records)['timestamp'])

    rows = []
    data_dict = {}
    rhythm = 0.1
    row_seconds = 0
    for record_num, record in zip(it.count(), it.islice(records, 0, count)):
        if record_num % 10000 == 0:
            print(record_num)
        if remove_record(record):
            continue
        record = transform_record(record)
        record_time = arrow.get(record['timestamp']) - base_time
        log_seconds = record_time.seconds + record_time.microseconds/1000000.0
        data_dict['log_seconds'] = log_seconds
        if 'fields' in record:
            data_dict.update(record['fields'])
        if log_seconds > row_seconds:
            data_dict['row_seconds'] = row_seconds
            rows.append(copy.copy(data_dict))
            row_seconds += rhythm
    return pd.DataFrame(rows)

