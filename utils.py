
import os
import os.path
import re
import shutil

################ Utils (for future use) ################

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


def ensure_directory(dir_name, verbose=False):
    # Create target Directory if don't exist
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
        if verbose:
            print("Directory ", dir_name, " Created ")
    else:
        if verbose:
            print("Directory ", dir_name, " already exists")


def file_base_name(file):
    base, extension = os.path.splitext(file)
    return base


def extract_base_names(file_list):
    for file in file_list:
        base, extension = os.path.splitext(file)
        yield base


def backup_file(path_to_file, max_count=3):
    "Create up to MAX_COUNT backup versions of a file in a directory called BACKUP."
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
    backup_file = os.path.join(backup_directory, f"{filename}_00")
    shutil.copy(path_to_file, backup_file)
    return backup_file

################################################################
class DictClass(dict):
    """
    Class that constructs like a dict, but acesss fields like a class.  Makes things much
    more compact.  So:
    foo = DictClass(one=1, two=2, bar=10)
    foo.bar (rather than foo['bar'])
            """
    def __init__(self, **args):
        for key in args.keys():
            self[key] = args[key]

    def __setattr__(self, attr, val):
        self[attr] = val
        
    def __getattr__(self, attr):
        return self[attr]

    def __str__(self):
        res = ""
        for k in sorted(self.keys()):
            v = self[k]
            if isinstance(v, float):
                res += f"{k}:{v:.2f}, "
            else:
                res += f"{k}:{v}, "
        return res
