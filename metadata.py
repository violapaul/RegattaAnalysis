"""
# Metadata Module

**This is a literate notebook.**

## Motivation

From [Wikipedia](https://en.wikipedia.org/wiki/Metadata)
> Metadata is "data that provides information about other data". In other words, it is "data about data."

What were the conditions on a particular day?  The crew?  What sort of jib settings did we use?  The finishing position?  What were the shroud settings?  How did they perform?  

I started out by writing a email for each race, trying to including learnings, conditions, results.  I moved to creating a Google doc for each race, easier to edit and update. And then I moved to creating a Jupyter notebook for each race day, easier to include data from the actual race all in one place.

The problems with these approaches:

- Repeated work.  Each email/gdoc/notebook is a vague copy of the previous, updated with new info.  This copy/edit process is annoying.
  - For example, one step is to grab the weather/tides, and just this step takes a while by hand.
- The data is locked in a human readable document, not in a machine readable representation.
  - No easy way to generate a single document (i.e. table of contents that shows all races, dates and times).
- No way to analyze the data in one place.  Where can we look to see trends or issues that are inconsistent?

Philosophically, I like metadata which can be searched and cross-referenced.  Data should be easy to edit and update and view.

The solution is to store all this metadata in a single easy to edit datastructure which can then be analyzed/created/edited/rendered for various needs.
"""

#### Cell #2 Type: module ######################################################

# imports
import os
import copy
import numbers
import re

import yaml  # We'll use YAML for our metadata
import json

import arrow
import numpy as np
import pandas as pd

# These are libraries written for RegattaAnalysis
from global_variables import G  # global variables
import utils
from utils import DictClass
import process as p
from nbutils import display_markdown, display

#### Cell #5 Type: module ######################################################

# code to render a metadata entry in markdown... 

def display_race_metadata(race_record, include_extras=True):
    "Summarize a race."
    display_markdown(f"# {race_record['title']}: {race_record['date']}")
    rr = race_record.copy()
    for k in "description conditions performance learnings".split():
        if k in rr:
            display_section(k.capitalize(), rr.pop(k))
    links = "raceqs raceqs_video".split()
    if has_key(links, race_record):
        display_markdown("## Links")
        lines = ""
        for k in links:
            if k in rr:
                lines += lines_url(make_title(k), rr.pop(k))
        display_markdown(lines)
    if include_extras:
        keys = list(rr.keys())
        if len(keys) > 0:
            lines = ""
            display_markdown("## Extras")
            lines = lines_dict("", keys, rr)
            display_markdown(lines)

def has_key(key_list, dictionary):
    for k in key_list:
        if k in dictionary:
            return True
    return False

def is_list_of_dicts(val):
    return isinstance(val, list) and isinstance(val[0], dict)

def lines_dict(prefix, keys, dictionary):
    lines = ""
    for k in keys:
        val = dictionary[k]
        if is_list_of_dicts(val):
            for i, v in enumerate(val):
                lines += f"{prefix}- **{k}: {i}**\n"
                lines += lines_dict(prefix+"  ", v.keys(), v)
        else:
            lines += f"{prefix}- **{k}**: {val}\n"
    return lines
            
def display_section(title, text):
    "Displays a markdown section with text."
    display_markdown(f"## {title}")
    display_markdown(text)        
        
def is_url(s):
    return s.startswith("http")  # Not great, but OK for now

def make_title(key):
    """"
    YAML keys are python keywords (lowercase and separated by underscores).  This converts to a pretty 
    and printable string.
    """
    words = key.split("_")
    words = [w.capitalize() for w in words]
    return " ".join(words)

def lines_url(link_text, url):
    "Displays a markdown URL."
    return f"- [{link_text}]({url})\n"

#### Cell #7 Type: module ######################################################

# Race metadata is stored as a multi-document sequence in the YAML file.  

def read_metadata():
    """
    Read the race metadata and return a struct, with a 
    - timestamp
    - records:  list of records
    - dates:    dict from date to record
    """
    race_yaml = read_yaml(G.METADATA_PATH)
    dates = {}
    records = []
    for record in race_yaml:
        # If the record is missing a source, assume it was written byhand.
        if 'source' not in record:
            record['source'] = 'byhand'
        if 'date' not in record:
            print(record)
        dates[record['date']] = record
        records.append(record)
    # File timestamp, used to find valid updates in other sources.
    ts = arrow.get(os.path.getmtime(G.METADATA_PATH)).to('US/Pacific')
    G.logger.info(f"Read {len(records)} records.")
    return DictClass(dates=dates, records=records, timestamp=ts)

def save_metadata(race_records):
    """
    Save a sequence of race records to the metadata file.
    """
    G.logger.info(f"Writing {len(race_records)} records.")    
    utils.backup_file(G.METADATA_PATH)
    # For convenience sort the records by date before writing.
    sorted_records = sorted(race_records, key=lambda r: r['date'])
    with open(G.METADATA_PATH, 'w') as yfs:
        save_yaml(sorted_records, yfs)

def save_json(race_records, json_file):
    """
    Save a sequence of race records to JSON.
    """
    G.logger.info(f"Writing {len(race_records)} records.")    
    utils.backup_file(G.METADATA_PATH)
    # For convenience sort the records by date before writing.
    sorted_records = sorted(race_records, key=lambda r: r['date'])
    with open(json_file, 'w') as fs:
        json.dump(sorted_records, fs) 

# A bit of magic here to ensure we have the best loader/dumper.  Specifying this is required when 
# calling load/dump (below).
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

def read_yaml(yaml_path):
    "Read the race records stored in the YAML file.  Return a dict indexed by date string: YYYY-MM-DD."
    with open(yaml_path, 'r') as yaml_stream:
        race_yaml = list(yaml.load_all(yaml_stream, Loader=Loader))
    return race_yaml

                     
def save_yaml(race_entries, stream=None):
    "Save the race metadata as YAML."
    return yaml.dump_all(race_entries, stream, Dumper=Dumper,
                         default_flow_style=False, sort_keys=False)

#### Cell #9 Type: module ######################################################

# Since the data for all days is in one place its easy to generate an all up summary.

def race_summary(race):
    display_markdown(race_summary_lines(race))

def race_summary_lines(race):
    lines = ""
    lines += f"- **{race['date']}**: {race['title']}\n"
    for key in "description conditions".split():
        if key in race and race[key] is not None and len(race[key]) > 0:
            lines += f"  - *{key.capitalize()}:* {race[key]}\n"
    return lines

def display_race_summaries(race_records):
    "Summarize each race."
    lines = ""
    for race in race_records:
        lines += race_summary_lines(race)
    display_markdown(lines)

def summary_table(race_records, columns = None):
    "Return a summary table the races."
    rows = []
    if columns is None:
        columns = "date title file source".split()
    for race in race_records:
        row = {k:race.get(k, '') for k in columns}
        rows.append(row)
    return pd.DataFrame(rows)


#### Cell #16 Type: module #####################################################

# The column names are long and verbose, because the SHEET column names are the form field prompts.

# These names are good as documentation, but painful for programmatic access.  The table below maps from 
# a long name and a convenient short form.

SHORT_COLNAME_TO_LONG = {
    'date'            : 'Date YYYY-MM-DD (e.g. "2020-05-10") ',
    'title'           : 'Title: short name for sail (e.g. SBYC Snowbird #1)', 
    'purpose'         : 'Purpose',
    'crew'            : 'Crew',
    'description'     : 'Description (2-3 sentences, optional)',
    'conditions'      : 'Conditions (i.e. description)',
    'performance'     : 'Performance (i.e. how did we perform vs other boats or polars).',
    'learnings'       : "Learnings (something we'd like to repeat or avoid)",
    'warnings'        : 'Warnings (needed repair, change, etc).',
    'wave'            : 'Wave height (pick best)',
    'wind'            : 'Winds in knots (pick best description)',
    'port_pointing'   : 'Port Pointing',
    'stbd_pointing'   : 'Starboard Pointing',
    'settings'        : 'Settings Summary (how were sail controls set? trim?)',
    'shroud_name'     : 'Shrouds (short name)',
    'shroud_tension'  : 'Shroud tensions (UP, MID, LOW: comma sep: 29,10,0). Pos low is tension, neg low is circle size in cm).',
    'other'           : 'Other (try to be structured!)',
    'additional_crew' : 'Additional Crew (comma separated)',
    'timestamp'       : 'Timestamp',
    'fluid_comments'  : 'Comments on Fluids?',
    'fluids'          : 'Gas, Water, Pump Out, Empty Bilge?',
}

LONG_COLNAME_TO_SHORT = {v:k for k, v in SHORT_COLNAME_TO_LONG.items()}

#### Cell #17 Type: module #####################################################

# Read the sheet and convert to a pandas table.

def read_gsheet():
    "Read the latest GSHEET.  Check that nothing bad has happened, and convert to short names."
    gs = pd.read_csv(G.GSHEET_URL)
    check_columns_changed(gs, LONG_COLNAME_TO_SHORT)
    return convert_to_short_names(gs)

def check_columns_changed(df, long_colname_map):
    """
    Its entirely possible that I will someday edit the form and then the columns will get
    out of whack.  Check to see that there are neither new fields or missing fields.
    """
    new_colnames = []
    missing_colnames = []
    for c in df.columns:
        if c not in long_colname_map:
            new_colnames.append(c)
    for c in long_colname_map:
        if c not in df.columns:
            missing_colnames.append(c)
    if len(new_colnames) == 0 and len(missing_colnames) == 0:
        G.logger.info("No missing or extra columns.")
        return True
    else:
        G.logger.warning(f"Uh Oh. New cols {new_colnames}. Missing cols {missing_colnames}.")
        return False

def convert_to_short_names(raw_metadata):
    "Assuming the columns names have not changed, convert to a short form."
    return raw_metadata.rename(LONG_COLNAME_TO_SHORT, axis='columns')

#### Cell #19 Type: module #####################################################

# Next step is to convert the Google Form spreadsheet rows to race metadata entries.  The goal was to 
# keep the two "close" so that conversion is not onerous,  but we do need to massage some of the fields.

def gsheet_row_to_metadata(row):
    "Convert the Google Form spreadsheet rows to race metadata entries."
    res = {}
    # If both the date and the timestamp is missing then something must be wrong.
    if is_missing_value(row['date']) and is_missing_value(row['timestamp']):
        G.logger.info(f"Encountered a row with missing date and timestamp. Skipping.")
        return None
    for key, val in row.iteritems():
        if key == 'date' and is_missing_value(val):
            val = timestamp_convert(row['timestamp']).format("YYYY-MM-DD")
        if is_missing_value(val):
            continue
        elif key == 'timestamp':
            val = timestamp_convert(row['timestamp']).datetime
        elif key == 'wind':
            val = [int(s.strip()) for s in val.split("-")]
        elif key == 'crew':
            val = [s.strip() for s in val.split(",")]
        elif key == 'shroud_tension':
            val = [int(s.strip()) for s in val.split(",")]
        res[key] = val
    res['source'] = 'gsheet'
    # It is a bit easier if the fields are in the same/similar order.
    return reorder_some_keys(res, SHORT_COLNAME_TO_LONG.keys())

def reorder_some_keys(dictionary, keys):
    "Return a dictionary with the keys in the order presented in keys."
    d = copy.copy(dictionary)
    res = dict()
    # Copy the ones in keys
    for k in keys:
        if k in d:
            res[k] = d.pop(k)
    # Copy the rest.
    for k in d.keys():
        res[k] = d[k]
    return res

def is_missing_value(val):
    "Pandas replaces empty CSV entries with NaN.  Return True if encountered."
    return isinstance(val, numbers.Number) and np.isnan(val)

def timestamp_convert(val):
    "Convert the google forms timestamp to a date."
    return arrow.get(val, 'M/D/YYYY H:mm:ss', tzinfo='US/Pacific')

#### Cell #21 Type: module #####################################################

# If there are new rows in the GSHEET, then add them to metadata.

def update_metadata_from_gsheet():
    "Read metadata.yml and gsheet and update as needed."
    metadata = read_metadata()
    gsheet = read_gsheet()
    new_records = add_gsheet_records(gsheet, metadata)
    save_metadata(new_records)

def add_gsheet_records(gsheet, metadata):
    "Find records in the gsheet which are missing from the existing metadata."
    dates = metadata.dates.copy()
    res = []
    for _, row in gsheet.iterrows():
        record = gsheet_row_to_metadata(row)
        if record is not None:
            date = record['date']
            G.logger.debug(f"Examining record {date}")
            if date not in dates:
                # If the date is missing just add it.
                G.logger.info(f"Found new record for {date} : {record.get('title', '')}")
                res.append(record)
            else:
                # If the date is already present we need to merge.
                existing = dates.pop(date)
                source = existing['source']
                if source == 'byhand':
                    # The existing was entered byhand in metadata.yml.  Be careful!
                    if metadata.timestamp < record['timestamp']:
                        G.logger.warning(f"Duplicate record. GSheet row is newer than byhand metadata: {record['timestamp']}.")
                        # Append both, we'll need to figure this out by hand
                        res.append(existing)
                        res.append(record)
                    else:
                        # Otherwise the GSHEET entry is older than the current record.  Ignore.
                        res.append(existing)
                elif source in ['loginfo', 'logprocess', 'gsheet']:
                    # Source was an automated process.  We can merge the records, overwriting with GSHEET
                    G.logger.debug(f"Merging gsheet into exiting record.")
                    # Overwrite the values in these records.
                    new_record = {**existing, **record}
                    res.append(new_record)
                else:
                    G.logger.warning(f"Found strange source: {source}.")
    return res + list(dates.values())

# notebook

#### Cell #24 Type: module #####################################################

def update_metadata_from_loginfo():
    "Read metadata.yml and loginfo and update as needed."
    metadata = read_metadata()
    loginfo = pd.read_pickle(G.LOG_INFO_PATH)    
    updated = merge_loginfo_records(loginfo, metadata)
    save_metadata(updated)


# Process the legacy loginfo data, this is only needed once.
def merge_loginfo_records(loginfo, metadata):
    """
    Create a metadata record for each loginfo record.  When a key already exists in
    metadata merge the info, overwriting fields in the loginfo record.
    """
    dates = metadata.dates.copy()
    rows = []
    for i, row in loginfo.iterrows():
        adt = datetime_from_log_filename(row.file)
        date_string = date_from_datetime(adt)
        record = {}
        # Munge loginfo data into "metadata" schema.
        record['file'] = row.file
        record['date'] = date_string
        record['title'], record['description'] = loginfo_title(row)
        record['begin'] = row.begin
        record['end'] = row.end
        record['source'] = 'loginfo'
        # Overwrite with the existing record... if it exists.
        if date_string in dates:
            record.update(dates.pop(date_string))
        rows.append(record)
    return rows + list(dates.values())

def loginfo_title(row):
    "Create a title and description from row record."
    if len(row.race) > 0:
        return row.race, row.description
    else:
        return row.description, ''


#### Cell #25 Type: module #####################################################

# Finally, during upload we should ensure that there is a default and empty metadata record 
# for each race.

def add_missing_metadata():
    """
    Working backward from the full list of pandas datafiles, ensure there is a default
    entry in the metadata file for each.
    """
    metadata = read_metadata()
    dates = metadata.dates.copy()
    new_dates = {}
    pfiles = p.pandas_files()
    G.logger.info(f"Found {len(pfiles)} pandas files.")
    for f in sorted(pfiles):
        adt = datetime_from_log_filename(f)
        date = date_from_datetime(adt)
        G.logger.debug(f"Examining {date} : {f}")
        # Default metadata... basically empty
        record = dict(file=f, date=date, title=date, begin=0, end=-1, source='logprocess')
        if date not in dates:
            G.logger.info(f"Found missing entry for {date} : {f}.")
        if date in new_dates:
            G.logger.warning(f"Two files for {date}. Watch out. Skipping.")
        else:
            existing = dates.pop(date, {})
            record.update(existing)
            new_dates[date] = record
    all_records = list(new_dates.values()) + list(dates.values())
    G.logger.info(f"Outputting {len(all_records)} records.")
    save_metadata(all_records)


def datetime_from_log_filename(filename, time_zone='US/Pacific'):
    "Extracts the datetime from a log filename."
    dt_string = re.sub(r".gz$", "", filename)   # Could be compressed
    dt_string = re.sub(r".pd$", "", dt_string)  # Standard .pd
    return arrow.get(dt_string, "YYYY-MM-DD_HH:mm", tzinfo=time_zone)
    

def date_from_datetime(adt):
    return adt.format("YYYY-MM-DD")

#### Cell #26 Type: module #####################################################

def update_race(updated_race_record):
    "Replace the race record, by date."
    date = updated_race_record['date']
    md = read_metadata()
    if date not in md.dates:
        raise Exception(f"Warning, {date} could not be found in the race logs.")
    md.dates[date] = updated_race_record
    save_metadata(list(md.dates.values()))

#### Cell #28 Type: metadata ###################################################

#: {
#:   "metadata": {
#:     "timestamp": "2021-03-13T11:44:51.824521-08:00"
#:   }
#: }

#### Cell #29 Type: finish #####################################################

