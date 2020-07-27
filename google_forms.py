
# imports
import copy
import collections
import numbers

import yaml  # We'll use YAML for our metadata
import arrow
import numpy as np
import pandas as pd

# These are libraries written for RegattaAnalysis
from global_variables import G  # global variables

# A bit of magic here to ensure we have the best loader/dumper.  Specifying this is required when 
# calling load/dump (below).
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

import nbutils
from nbutils import display_markdown, display


URL = r"https://docs.google.com/spreadsheets/d/e/2PACX-1vS5g8oeSAMk-CFP-xDi4hu9a23W-iF5SMNjap-Gd78BPWvhA1GGgpDqFkQaEUVD3zoM9Pud1fozuDn8/pub?output=csv"

raw_metadata = pd.read_csv(URL)  # index_col=0, parse_dates=True)

def print_cols(metadata):
    for c in metadata.columns:
        print(repr(c))

SHORT_COLNAME_TO_LONG = {
    'date'            : 'Date YYYY-MM-DD (e.g. "2020-05-10") or blank for today.',
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
    'shroud_tension'  : 'Shroud tensions (UP, MID, LOW: comma sep: 29, 10, 0). Pos low is tension, neg low is circle size in cm).',
    'other'           : 'Other (try to be structured!)',
    'additional_crew' : 'Additional Crew (comma separated)',
    'timestamp'       : 'Timestamp',
    'fluid_comments'  : 'Comments on Fluids?',
    'fluids'          : 'Gas, Water, Pump Out, Empty Bilge?',
}

LONG_COLNAME_TO_SHORT = {v:k for k, v in SHORT_COLNAME_TO_LONG.items()}


def check_columns_changed(df, long_colname_map):
    new_colnames = []
    missing_colnames = []
    for c in df.columns:
        if c not in long_colname_map:
            new_colnames.append(c)
    for c in long_colname_map:
        if c not in df.columns:
            missing_colnames.append(c)
    if len(new_colnames) == 0 and len(missing_colnames) == 0:
        display("No missing or extra columns.")
    else:
        display(f"Uh Oh. New cols {new_colnames}. Missing cols {missing_colnames}.")
    return new_colnames, missing_colnames




def convert_to_yaml(metadata):
    data = [gform_row_to_metadata(row) for index, row in metadata.iterrows()]
    yaml.dump_all(data, None, Dumper=Dumper, default_flow_style=False, sort_keys=False)
    
