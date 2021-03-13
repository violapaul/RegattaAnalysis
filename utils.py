
import os
import os.path
import re
import shutil
import collections.abc
import numbers
import time
import itertools as it
import datetime
import dateutil
import arrow

import copy

from global_variables import G  # global variables

################ type checking ################

def is_iterable(thing):
    return isinstance(thing, collections.abc.Iterable)

def is_number(thing):
    return isinstance(thing, numbers.Number)

################ itertools ################

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = it.tee(iterable)
    next(b, None)
    return zip(a, b)

def nwise(iterable, n=2):                                                      
    iters = it.tee(iterable, n)                                                     
    for c, i in enumerate(iters):                                               
        next(it.islice(i, c, c), None)                                               
    return zip(*iters)


################ utils ################
def run_system_command(command, dry_run=False):
    "Run a shell command, time it, and log."
    G.logger.debug(f"Running command: {command}")
    if not dry_run:
        start = time.perf_counter()
        res = os.system(command)
        end = time.perf_counter()
        G.logger.debug(f"Command finished in {end-start:.3f} seconds.")
        return res
    else:
        return None

def yes_no_response(message, yes_default=False):
    if yes_default:
        query = "[Y/n]"
    else:
        query = "[y/N]"
    response = input("{0} {1}".format(message, query))
    if response == '':
        return yes_default
    elif response.casefold() == 'y':
        return True
    else:
        return False


def file_base_name(file):
    base, extension = os.path.splitext(file)
    return base


def extract_base_names(file_list):
    for file in file_list:
        base, extension = os.path.splitext(file)
        yield base


def ensure_directory(dir_name, verbose=False):
    # Create target Directory if don't exist
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
        G.logger.debug(f"Directory {dir_name} Created ")
    else:
        G.logger.debug(f"Directory {dir_name} alread exists.")
            

def backup_file(path_to_file, max_count=3):
    "Create up to MAX_COUNT backup versions of a file in a directory called BACKUP."
    if os.path.exists(path_to_file):
        directory, filename = os.path.split(path_to_file)
        backup_directory = os.path.join(directory, "Backup")
        ensure_directory(backup_directory)
        bfiles = [f for f in os.listdir(backup_directory) if f.startswith(filename)]
        bfiles = reversed(sorted(bfiles)[:max_count])
        # move existing files
        for file in bfiles:
            m = re.match(r".*_(\d\d)", file)
            if m:
                num = int(m.group(1))+1
                new_name = f"{filename}_{num:02d}"
                os.rename(os.path.join(backup_directory, file), os.path.join(backup_directory, new_name))
        backup_path = os.path.join(backup_directory, f"{filename}_00")
        G.logger.debug(f"Backing up {path_to_file} to {backup_path}.")
        shutil.copy(path_to_file, backup_path)
        return backup_path

################################################################
class DictClass(dict):
    """
    Class that constructs like a dict, but acesss fields like a class.  Makes things much
    more compact.  So:

    foo = DictClass(one=1, two=2, bar=10)
    foo.bar (rather than foo['bar'])

    Note, it inherits from dict, so it can do all the dict things as well!
    """
    def __init__(self, **args):
        for key in args.keys():
            self[key] = args[key]

    def __setattr__(self, attr, val):
        self[attr] = val
        
    def __getattr__(self, attr):
        return self[attr]

    def union(self, dict_like):
        "Like dict update, but returns a new DictClass.  Keys in the argument overwrite keys in self."
        res = DictClass(**self)
        for k, v in dict_like.items():
            res[k] = v
        return res
    
    def __str__(self):
        res = ""
        for k in sorted(self.keys()):
            v = self[k]
            if isinstance(v, float):
                res += f"{k}:{v:.2f}, "
            else:
                res += f"{k}:{v}, "
        return res

    def __deepcopy__(self, memo):
        return self.__class__(
            **{k: copy.deepcopy(v, memo) for k, v in self.items()}
        )    

################################################################
# Time manipulation functions.  For a while I was enamored with arrow, but I am less clear
# on that now.  In the meantime let's hide time.  Could move to another time library easily.
def time_now():
    return arrow.now()

def time_to_string(time, format=None):
    if format:
        return time.format(format)
    else:
        return time.isoformat()

def time_from_string(s, tz=dateutil.tz.tzlocal()):
    return arrow.get(s).replace(tzinfo=tz)

def time_to_timestamp(time):
    return time.float_timestamp

def time_from_timestamp(ts):
    return arrow.get(ts).to('US/Pacific')

def time_after(tbefore, tafter, epsilon=0):
    "True if tafter is epsilon seconds after tbefore.  "
    delta = tafter - tbefore
    return delta > datetime.timedelta(seconds=epsilon)

################################################################

def update_jpeg_times(jpg_dir):
    """
    Finds the files in a directory and updates the DateTimeOriginal so that programs like Amazon Photos will
    sort them correctly.
    """
    jpg_files = sorted([f for f in os.listdir(jpg_dir) if f.endswith("jpg")])
    for n, f in enumerate(jpg_files):
        path = os.path.join(jpg_dir, f)
        command = f'exiftool "-datetimeoriginal<filemodifydate" {path}'
        print(command)
        d = n+1
        command = f'exiftool -datetimeoriginal+="0:0:0 00:{d:02d}:00" {path}'
        print(command)
        # utils.run_system_command(command)
