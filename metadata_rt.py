
#### Cell #0 Type: markdown ####################################################

#: # Metadata Module
#: 
#: **This is a literate notebook.**
#: 
#: ## Motivation
#: 
#: From [Wikipedia](https://en.wikipedia.org/wiki/Metadata)
#: > Metadata is "data that provides information about other data". In other words, it is "data about data."
#: 
#: What were the conditions on a particular day?  The crew?  What sort of jib settings did we use?  The finishing position?  What were the shroud settings?  How did they perform?  
#: 
#: I started out by writing a email for each race, trying to including learnings, conditions, results.  I moved to creating a Google doc for each race, easier to edit and update. And then I moved to creating a Jupyter notebook for each race day, easier to include data from the actual race all in one place.
#: 
#: The problems with these approaches:
#: 
#: - Repeated work.  Each email/gdoc/notebook is a vague copy of the previous, updated with new info.  This copy/edit process is annoying.
#:   - For example, one step is to grab the weather/tides, and just this step takes a while by hand.
#: - The data is locked in a human readable document, not in a machine readable representation.
#:   - No easy way to generate a single document (i.e. table of contents that shows all races, dates and times).
#: - No way to analyze the data in one place.  Where can we look to see trends or issues that are inconsistent?
#: 
#: Philosophically, I like metadata which can be searched and cross-referenced.  Data should be easy to edit and update and view.
#: 
#: The solution is to store all this metadata in a single easy to edit datastructure which can then be analyzed/created/edited/rendered for various needs.

#### Cell #1 Type: markdown ####################################################

#: ## Overview
#: 
#: Code to process race metadata and associate with race logs.
#: 
#: There are currently 3 sources of hand entered metadata, hopefully fewer soon:
#: - A file called metadata.yaml (in YAML).  This is the final ultimate source for metadata.
#:   - YAML is super powerful, allowing you to enter complex structured info.
#:   - It is also designed to be human readable (unlike a database file, or even a CSV).
#:   - Unfortunately there is no strong schema, and it is not that hard to mess things up.
#: - A google spreadsheet (gsheet) that is fed by a [Google Form](https://forms.gle/JENZZdSWKNuoF8icA) (which is easy to use on the ride home from the race).
#:   - The form determines the schema, which can be changed, but it does enforce some structure.
#:   - The spreadsheet can be downloaded as a CSV from a URL.
#: - An older pandas dataframe, `log_info.pd` which is now deprecated.
#: 
#: And a final source, which is a default and empty metadata record generated when we first upload the file.
#: 
#: In all cases above the **key** is the date (and we therefore assume that there is a single "file" per day).  In practice we may have several races on a single day, though these will be in one file. The YAML file will support the ability to discuss the segments.  The Google form does not.
#: 
#: **How to merge duplicates?**. 
#: 
#: - Multiple rows with same date in the gsheet.
#:   - Delete by hand?  Take the latest?
#: - Same date in gsheet and YAML.
#:   - Note, gsheet row will move to YAML when it first appears.
#:   - Figure out which is newer.  
#:     - If YAML is newer, keep it.
#:     - If gsheet is newer, then keep **both**.  Warn user and ask to edit.
#: 
#:   
#:   
#: 
#: ### References 
#: 
#: - Good YAML reference to start: [YAML tutorial](https://rollout.io/blog/yaml-tutorial-everything-you-need-get-started/)
#:   - [The official reference](https://yaml.org/) Its written in YAML (which makes it a bit weird).
#: - Nice Google page on [how to use Google Forms](https://zapier.com/learn/google-sheets/how-to-use-google-forms/)
#: 
#: 
#: ## TODO
#: 
#: - Add timestamp to all YAML entries?  
#: 
#: - Tides? 
#: 
#: - Currents?  
#: 
#: - Weather: wind, etc.  Weather buoy?
#: 
#: - Create a page before the race? 
#: 
#: - Phone images captured during the race 
#:   - pull them in, link them to the map?
#:   - extract settings?
#:   
#: - Sometimes there are two files from the same day.  In general should not have happened...  but it screws things up.
#: 
#: - Provide a tool to slice a days data into "segments".
#: 
#: 
#: 
#: ## Caveats and concerns
#: 
#: - YAML, as edited by a human author, does not support a strong schema.  Its easy to mess things up, with typos, missing fields, incorrectly named fields, etc.


#### Cell #2 Type: module ######################################################

# imports
import os
import copy
import numbers
import re
from datetime import date, datetime

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

#### Cell #3 Type: notebook ####################################################

# notebook - YAML race metadata example.

# Below is snippet of YAML inline.  I am not going to document YAML here.  But notice that the structure
# is reminiscent of Python itself, and it is somewhat readable.

example = """
file: 2020-04-16_14:54.pd.gz
date: "2020-04-16"
title: Tune-up with Creative
purpose: tune_up
conditions: >-
  Beautiful day. Winds were 3 quickly building to 10ish. Flat
  seas. Upwind to Pt. Wells buoy, raised and raced home to the hamburger.
performance: >-
  Good height and speed vs. Creative on the way upwind. Perhaps a bit
  slow at first downwind, exploring to tradeoffs between depth and
  speed.  Best downwind speed when I was at the shrouds and Sara had a
  hand on the mainsheet.
learnings: >-
  Let the sails out for downwind: both main and kite.  Stand forward
  if possible.

  Shroud settings seemed really great, and versatile.  With only 2 on
  the boat, we sailed very well.  These settings are the new base!
raceqs_video: "https://youtu.be/9a5bLeZw8EM"
raceqs: "https://raceqs.com/tv-beta/tv.htm#userId=1146016&divisionId=64190&updatedAt=2020-04-17T18:05:59Z&dt=2020-04-16T15:43:47-07:00..2020-04-16T17:39:12-07:00&boat=Creative"
segments:
  - winds: [6, 12]
    tensions: [29, 10, 0]
    port: [2.251, 1.953, 999]
    stbd: [2.271, 1.959, 999]
    thoughts: >-
      Overall we have had trouble with the Quantum quick tune card,
      where the uppers are a bit too loose or the middles too tight.
      The result is that the mast falls off at the top, rather than
      staying straight or sagging for power.  We took a bit off the
      middle (12 down to 10).
questions:
  - text: Was the prop set correctly?
    author: sara
    context: Were we slower on one tack than the other? 
    proposed_solution: ??
"""

# The data can be trivialy read in this way.  A simple Python datastructure results, in this case
# a dictionary.  
race_metadata = yaml.load(example, Loader=yaml.Loader)

display_markdown("Notice that `segments` and `questions` are sublist of dicts.")
display(race_metadata)

#### Cell #4 Type: markdown ####################################################

#: ### Render this data to make it more readable
#: 
#: We can prettily easily write code that "renders" these Python structures into readable Markdown.

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

#### Cell #6 Type: notebook ####################################################

# notebook - Create a human readable version of this metadata
display_race_metadata(race_metadata)

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


def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""

    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError ("Type %s not serializable" % type(obj))
        
def save_json(race_records, json_file):
    """
    Save a sequence of race records to JSON.
    """
    G.logger.info(f"Writing {len(race_records)} records.")    
    utils.backup_file(G.METADATA_PATH)
    # For convenience sort the records by date before writing.
    sorted_records = sorted(race_records, key=lambda r: r['date'])
    with open(json_file, 'w') as fs:
        json.dump(sorted_records, fs, indent=4, default=json_serial)
                  

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

#### Cell #8 Type: notebook ####################################################

# notebook - load some metadata

metadata = read_metadata()
display(metadata.timestamp)

display_race_metadata(metadata.dates['2020-04-19'])
display_race_metadata(metadata.dates['2020-06-06'])


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


#### Cell #10 Type: notebook ###################################################

# notebook 

metadata = read_metadata()

display_race_summaries(metadata.records)
summary_table(metadata.records)

#### Cell #11 Type: notebook ###################################################

# notebook - metadata schema... light.

# We currently do not have a schema for the metadata, it is implicit, which is dangerous.  We
# can extract fields and their types.
#
# Note, we make a simplifying assumption that we have three situations: 
# i) primitive types, ii) dicts, ii) list of dicts. 
#
# We do not have dicts containing dicts.

def schema_lite(race_data):
    "Find a lightweight schema from the existing data.  Provides a guide for future entries."
    fd = flatten_dicts("", race_data)
    return distill(fd)    

def distill(race_union):
    res = {}
    for k, v in race_union.items():
        if len(v) == 0:
            res[k] = v[0]
        elif isinstance(v[0], dict):
            # print(k, "collapsing")
            collapsed = collapse_dicts(v)
            # print(k, "distilling", collapsed)
            res[k] = distill(collapsed)
            # print(k, "done")
        else:
            res[k] = set([type(e) for e in v])
    return res

def flatten_dicts(prefix, dicts):
    "Pass over a list of dicts and extract fields, and collect all the values assined to those fields.  For fields which are list of dicts, recurse.  Combine and flatten all fields."
    res = {}
    for d in dicts:
        for k, v in d.items():
            if is_list_of_dicts(v):
                # print(k, v)
                v = flatten_dicts(k + "[]", v)
                res.update(v)
            else:
                res[prefix+k] = res.get(k, []) + [v]
    return res

def distill_types(race_union):
    "For each key in race_union return a set containing the "
    res = {}
    for k, v in race_union.items():
        res[k] = set([type(e) for e in v])
    return res

def add_key(res, key_list, val):
    if len(key_list) > 0:
        base_key = key_list[0]
        if len(key_list) == 1:
            res[base_key] = res.get(base_key, set()).union(val)
        else:
            next_dict = res.get(base_key, {})
            next_dict.update(add_key)
            if base_key in res:
                pass

#### Cell #12 Type: markdown ###################################################

#: ### The fields in the metadata schema.
#: 
#: - The keys in this dict are the fields used.
#:    - If `foo[]bar` then field `foo` is a list of records, each containing the field `bar`.
#: - In both cases the value is a set of types that are encountered.


#### Cell #13 Type: notebook ###################################################

# notebook - extract the "schema"

schema_lite(metadata.records)

#### Cell #14 Type: markdown ###################################################

#: ## Reading metadata from Google Forms
#: 
#: We created a Google form to speed up post race metadata entry: https://forms.gle/JENZZdSWKNuoF8icA
#: 
#: The advantage of this **public** form is that it is easy to enter data from any device (including mobile) at any time.  And the schema is at least "weakly" enforced.
#: 
#: The form provides a scheme for publishing the resulting data as a spreadsheet, which is here: https://docs.google.com/spreadsheets/d/e/2PACX-1vS5g8oeSAMk-CFP-xDi4hu9a23W-iF5SMNjap-Gd78BPWvhA1GGgpDqFkQaEUVD3zoM9Pud1fozuDn8/pub?output=csv
#: 
#: The steps are:
#: - Create a Google form which has the fields you want.
#:   - Get a Google account and drive.
#:   - Create the form here:  https://docs.google.com/forms/u/0/
#: - Form data can be viewed as a Google sheet
#:   - One column for each field.
#: - A Google sheet can be exported as a CSV file.
#: 
#: Note, the Google form is designed to be easy to use and enter data.  The ultimate goal is to produce semi-structured data that can be used to better analyze our performance in races.  The design of the form is a compromise that makes data entry easy (on the drive home from the race), and analysis easy.


#### Cell #15 Type: notebook ###################################################

# notebook - load GSHEET with form data

# This is a magic URL that can be extracted from any Google Sheet.  Look under File and then "Publish to Web".  
# Note this CSV takes a bit of time to be updated from the SHEET.
URL = r"https://docs.google.com/spreadsheets/d/e/2PACX-1vS5g8oeSAMk-CFP-xDi4hu9a23W-iF5SMNjap-Gd78BPWvhA1GGgpDqFkQaEUVD3zoM9Pud1fozuDn8/pub?output=csv"

# Use pandas to read the CSV
raw_gsheet = pd.read_csv(URL)  # index_col=0, parse_dates=True)

display_markdown("### A list of the columns in the GSHEET")

for c in raw_gsheet.columns:
    print(repr(c))


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

#### Cell #18 Type: notebook ###################################################

# notebook - read the sheet and display the rows

gsheet = read_gsheet()
gsheet

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

#### Cell #20 Type: notebook ###################################################

# notebook - convert from gsheet to metadata format

gsheet_records = [gsheet_row_to_metadata(row) for index, row in gsheet.iterrows()]

# Show one row.
print(save_yaml(gsheet_records[-1:]))

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

#### Cell #22 Type: notebook ###################################################

#notebook 

import collections

def find_duplicates(metadata):
    "Find duplicate dates in the metadata."
    dates = collections.defaultdict(list)
    for record in metadata.records:
        date = record['date']
        dates[date] += [record]
    return {k:v for k, v in dates.items() if len(v) > 1}
        

def find_missing_data(metadata):
    records = []
    for record in metadata.records:
        file = record.get('file', None)
        if file is None:
            records.append(record)
    return records
        
    
metadata = read_metadata()

dups = find_duplicates(metadata)
print(dups.keys())

print([(r['date'], r['title']) for r in find_missing_data(metadata)])

#### Cell #23 Type: notebook ###################################################

# notebook 

# loginfo is a legacy location for metadata, need to pull that in...  once.

log_info = pd.read_pickle(G.LOG_INFO_PATH)
display(len(log_info))
log_info[:5]


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

#### Cell #27 Type: notebook ###################################################

# notebook - 

if False:
    update_metadata_from_loginfo()
    add_missing_metadata()
    update_metadata_from_gsheet()

#### Cell #28 Type: metadata ###################################################

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
#:     "timestamp": "2021-03-13T11:44:36.206135-08:00"
#:   },
#:   "nbformat": 4,
#:   "nbformat_minor": 2
#: }

#### Cell #29 Type: finish #####################################################

