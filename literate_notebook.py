"""
# Literate Notebooks

## Motivation

I've found the most productive way to write and document code is by building it up in a Jupyter notebook.  I use the notebook to define the problem, the inputs and outputs, include examples, references and links, and then build up to a final set of solutions.  Along the way I end up writing functions which are similar or identical to a set functions to be collected into a module (the python name for libraries of functions).  Modules are critical because they can be imported and used in other notebooks/modules.  

Unfortunately, the notebook and the module can get out of sync.  I'll find a bug in the module as I am using it, and then go fix it.  Or I'll add a bit of functionality.  Gradually the notebook becomes stale.  Most of the info in the notebook is still true, but it doesn't reflect the final details accurately.

It would be way better if the **Notebook was the Module** and the **Module was the Notebook**.  

The code in this notebook/module provides a means to unify these concepts.
"""

#### Cell #4 Type: module ######################################################

# Load some modules
import os               # For paths and files.
import json             # Notebook files are in JSON
import re               # regular expressions
import logging          # logging of errors, warnings, etc.
import copy
import shutil
import datetime

import arrow            # Arrow time module.  May change.

# Some notebook specific libraries for display
from nbutils import display_markdown, display

#### Cell #5 Type: module ######################################################

# Key public functions, in a sense the rest should be private

def convert_notebook_to_roundtrip(notebook_path, output_path):
    "Create a 'roundtrip file', an editable python file which contains all notebook information."
    notebook = read_notebook(notebook_path)
    save_lines(output_path, notebook_to_roundtrip(notebook))
    
def convert_roundtrip_to_notebook(roundtrip_path, notebook_path):
    "Read a 'roundtrip file' and create a Jupyter notebook."
    lines = read_lines(roundtrip_path)
    notebook = notebook_from_roundtrip(lines)
    save_notebook(notebook, notebook_path)

def convert_notebook_to_module(notebook_path, module_path):
    "Given a notebook (ipynb), extract the module and write file."
    notebook = read_notebook(notebook_path)
    logging.info(f"Creating module {module_path}.")
    save_lines(module_path, notebook_to_module(notebook))

#### Cell #6 Type: module ######################################################

# Some basic routines for reading and writing from files.

def read_notebook(notebook_path):
    "Read the JSON encoding of the notebook ipynb file."
    with open(notebook_path, 'r') as fs:
        nb = json.load(fs)
    return nb

def save_notebook(notebook, notebook_path):
    "Write the JSON encoding of the notebook to an ipynb file."
    with open(notebook_path, 'w') as fs:
        json.dump(notebook, fs, indent=1)

def is_notebook(notebook_path):
    _, filename = os.path.split(notebook_path)
    base, extension = os.path.splitext(filename)
    regex = re.compile(r"[\s\W]+")  # Matches non-alphanumeric and whitespace
    if regex.search(base):
        raise Exception(f"{filename} contains invalid characters.")
    return extension == ".ipynb"
        
def is_literate_notebook(notebook_path):
    _, filename = os.path.split(notebook_path)
    if is_notebook(filename):
        base, extension = os.path.splitext(filename)
        return base.endswith("_Module")
    
# Filename munging
def base_name(notebook_filename):
    """
    By convention notebook names destined for module-hood will be 
    
    Capitilized, underscore delimited, and end in '_Module'.  
    
    E.G. 'Literate_Notebook_Module'
    
    The conversion downcases and removes '_Module'.
    
    Double check for spaces and invalid
    characters, since that would screw things up when defining a module.
    """
    base, extension = os.path.splitext(notebook_filename)
    if is_literate_notebook(notebook_filename):
        ending = "_Module"
        all_lower = base.lower()
        return all_lower[:-len(ending)]
    elif is_notebook(notebook_filename):
        return base.lower()
    else:
        raise Exception(f"{notebook_filename} not a valid name.")

# Create the three related filenames from the basename.  
def module_filename(base_name):
    return base_name + ".py"

def roundtrip_filename(base_name):
    return base_name + "_rt.py"

def notebook_filename(base_name):
    words = base_name.split("_")
    cap_words = [w.capitalize() for w in words]
    return "_".join(cap_words) + "_Module.py"

# Do the same for full paths.
def module_path(notebook_path, new_dir=None):
    path, filename = os.path.split(notebook_path)
    if new_dir is not None:
        path = new_dir
    return os.path.join(path, module_filename(base_name(filename)))

def roundtrip_path(notebook_path, new_dir=None):
    path, filename = os.path.split(notebook_path)
    if new_dir is not None:
        path = new_dir
    return os.path.join(path, roundtrip_filename(base_name(filename)))

# Lower level file and directory manipulation.
def read_lines(path):
    "Read the lines of a file specified by PATH."
    with open(path, 'r') as fs:
        for line in fs:
            yield line

def save_lines(file_path, lines):
    "Save a sequence of lines to file."
    with open(file_path, 'w') as fs:
        for line in lines:
            fs.write(line)

def ensure_directory(dir_name, verbose=False):
    "Create DIR_NAME if don't exist."
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
        logging.debug(f"Directory {dir_name} Created ")
    else:
        logging.debug(f"Directory {dir_name} alread exists.")

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
                from_file = os.path.join(backup_directory, file)
                to_file = os.path.join(backup_directory, new_name)
                logging.debug(f"Renaming {from_file} to {to_file}.")
                os.rename(from_file, to_file)
        backup_file = os.path.join(backup_directory, f"{filename}_00")
        logging.debug(f"Backing up {path_to_file} to {backup_file}")
        shutil.copy(path_to_file, backup_file)
        return backup_file

def tmp_path(path):
    old_dir, filename = os.path.split(path)
    tmp_dir = "/tmp"  # Not terribly portable
    return os.path.join(tmp_dir, filename)

#### Cell #10 Type: module #####################################################

# Code will process each cell in turn, handling markdown and code cells differently.
# Code cells are further differentiated by their cell tag.  
VALID_CELL_TAGS = {'notebook', 'module', 'test'}

# By convention, when you are writing a literate notebook, the cell tag is the first word
# on the first line (assuming that line is a comment).
REGEX_CELL_TAG = re.compile(r"#\s*([a-zA-Z]+).*", re.IGNORECASE)

def extract_cell_tag(line):
    "Return the cell tag on this line."
    m = REGEX_CELL_TAG.match(line)
    if m is not None:
        tag = m.group(1).casefold()
        if tag in VALID_CELL_TAGS:
            return tag
    # default is module
    return 'module'

#### Cell #12 Type: module #####################################################

# Recall there are two goals 
#  1) Create a python module from a literate notebook. 
#  2) Roundtrip from literate notebook to python and back.
#
# The two solutions will end up using many of the same pieces.
#
# When creating a Python file from a notebook, each cell will be rendered as a text
# "chunk". The chunks are delimeted by special comment lines.
#
# These functions are used to flag the beginning and end of chunks.

# Match a cell marker comment
REGEX_CHUNK_MARKER = re.compile(r'#### Cell #(\d+) Type: (\w+) #+$')

def make_chunk_marker(num, cell_type, line_length=80):
    "A magic line which flags the beginning of a chunk."
    return f"#### Cell #{num} Type: {cell_type} {'#' * line_length}"[:line_length]

def is_chunk_marker(line):
    "Returns the cell number and cell type IFF the line is a chunk marker."
    m = REGEX_CHUNK_MARKER.match(line)
    if m is not None:
        return int(m.group(1)), m.group(2)
    return None

def make_chunk_separator(num, cell_type):
    "Surround the marker by blank lines."
    # Add empty lines for readability
    return ["\n", make_chunk_marker(num, cell_type) + "\n"]

# Helpers to construct the list of text lines destined for the python file.
def add_line(results, line):
    "Add a single line to the current list of lines stored in RESULT.  By convention a single line is not newline terminated, so we will add one."
    results.append(line + "\n")

def add_lines(results, list_of_lines):
    "Add multiple lines to the current list of lines stored in RESULT. By convention all lines but the last are newline terminated"
    results.extend(list_of_lines)
    results.append("\n")


#### Cell #14 Type: module #####################################################

# The module version of the literate notebook is primarily just the "module" chunks.  To
# this we add a first docstring chunk and a final metadata chunk.  The docstring chunk is
# a version of the first notebook cell, which is often a markdown cell (see also
# https://www.python.org/dev/peps/pep-0257/).  The final metadata cell is to store
# additional metadata that is not used by python or the casual viewer of the module.

def notebook_to_module(notebook_json, use_chunks=True):
    "Extract the **module** code from the notebook JSON datastructure.  Returns a list of lines."
    res = []
    for num, cell in enumerate(notebook_json['cells']):
        if num == 0 and cell['cell_type'] == 'markdown':
            # first cell is markdown, create the documentation string for the module
            add_line(res, '"""')
            add_lines(res, cell['source'])
            add_line(res, '"""')
        elif cell['cell_type'] == 'code':
            # Otherwise the code 
            source = cell.get('source', [])
            if len(source) > 0 and extract_cell_tag(source[0]) == 'module':
                if use_chunks:
                    add_lines(res, make_chunk_separator(num, 'module'))
                add_lines(res, cell['source'])
    # Finish off with a metadata chunk that includes a timestamp
    add_metadata_chunk(res, num+1, add_timestamp_to_metadata({}))
    # Finish the last chunk
    add_lines(res, make_chunk_separator(num+2, 'finish'))
    return res

def add_metadata_chunk(res, num, metadata):
    add_lines(res, make_chunk_separator(num, 'metadata'))
    jstring = json.dumps(metadata, indent=2)
    jlines = jstring.splitlines(True)
    # Comment each line of JSON
    add_lines(res, [comment_line(l) for l in jlines])

################################################################
# Time manipulation functions.  For a while I was enamored with arrow, but I am less clear
# on that now.  In the meantime let's hide time.  Could move to another time library easily.
def time_now():
    return arrow.now()

def time_to_string(time):
    return time.isoformat()

def time_from_string(s):
    return arrow.get(s)

def time_to_timestamp(time):
    return time.float_timestamp

def time_from_timestamp(ts):
    return arrow.get(ts).to('US/Pacific')

def time_after(tbefore, tafter, epsilon=datetime.timedelta(seconds=0.3)):
    "True if tafter is truly after tbefore."
    delta = tafter - tbefore
    return delta > epsilon

################################################################
# Helper routines for adding and extracting timestamp info from metadata.

def add_timestamp_to_metadata(metadata):
    "Add a timestamp to the metadata which will be serialized."
    res = copy.deepcopy(metadata)
    # Get the metadata key, if not create.
    m = res.get('metadata', {})
    res['metadata'] = m
    # Note, we hide the timestamp in the "metadata" field.  
    m['timestamp'] = time_to_string(time_now())
    return res

def extract_timestamp_from_metadata(metadata):
    if 'metadata' in metadata:
        if 'timestamp' in metadata['metadata']:
            return time_from_string(metadata['metadata']['timestamp'])
    return None

################################################################
# Code to handle commenting and uncommenting non-python regions.

# To encode non-code sections, including markdown and JSON metadata we will use python
# comments...  slightly extended.
COMMENT_MARK = "#:"
COMMENT_TEXT = COMMENT_MARK + " "

def comment_line(line):
    "Turn a line into a Python comment line."
    return COMMENT_TEXT + line

REGEX_WHITESPACE_ONLY = re.compile(r"^\s*$")

def uncomment_line(line):
    "Take a commented line and remove the comment."
    if REGEX_WHITESPACE_ONLY.match(line):
        return line
    if line.startswith(COMMENT_TEXT):
        return line[len(COMMENT_TEXT):]
    # Empty line.  Looks reasonable to a human, but is not generated by the commenter.
    if line.startswith(COMMENT_MARK):
        return line[len(COMMENT_MARK):]
    logging.error(f">{line}<")
    raise Exception(f"Attempt to uncomment a line which is not commented.")

#### Cell #17 Type: module #####################################################

# The second type of python file is a "roundtrip" file.  This file contains all the info
# in the literate notebook (in fact any notebook).  Non-python information is serialized
# in comments.

def notebook_to_roundtrip(notebook_json):
    "Create a roundtrip-able python script from the JSON notebook structure."
    res = []
    for num, cell in enumerate(notebook_json['cells']):
        ctype = cell['cell_type']
        if ctype == 'markdown':
            add_markdown_chunk(res, num, cell)
        elif ctype == 'code':
            add_code_chunk(res, num, cell)
        else:
            logging.warning(f"Found a ctype = {ctype} in cell #{num} which is not supported.")
    # Finally add a chunk that includes notebook metadata
    add_metadata_chunk(res, num+1, notebook_metadata(notebook_json))
    # Finish the last chunk
    add_lines(res, make_chunk_separator(num+2, 'finish'))
    return res

def add_markdown_chunk(res, num, cell):
    "Add a markdown chunk to the list of lines in RES."
    source = cell.get('source', [])
    add_lines(res, make_chunk_separator(num, 'markdown'))
    # markdown text must be commented.
    add_lines(res, [comment_line(l) for l in source])

def add_code_chunk(res, num, cell):
    "Add a code chunk to the list of lines in RES."
    # Otherwise the code 
    source = cell.get('source', [])
    if len(source) == 0:
        # empty cell is by default a module
        ctag = 'module'
    else:
        ctag = extract_cell_tag(source[0])
    add_lines(res, make_chunk_separator(num, ctag))
    add_lines(res, source)

def notebook_metadata(notebook_json):
    "Extract metadata portion of the the notebook."
    # metadata is everything but the cells
    metadata = copy.copy(notebook_json)
    # metadata is everything *but* the code in the cells.
    del metadata['cells']
    # Add a timestamp.  It is safe to tuck it inside of the metadata subdict.
    return add_timestamp_to_metadata(metadata)

#### Cell #19 Type: module #####################################################

# Code to deserialize a rountrip notebook.  

def notebook_from_roundtrip(file_lines):
    "Create a notebook datastructure from the content of a roundtrip-able python file."
    notebook = {}
    cells = []
    for chunk in notebook_chunks(iter(file_lines)):
        num, ctype, lines = chunk
        if ctype in ['module', 'notebook', 'test', 'markdown']:
            cells.append(create_jupyter_cell(ctype, lines))
        elif ctype == 'metadata':
            notebook = extract_metadata(lines)
    notebook['cells'] = cells
    return notebook

# All the code below relies on file_lines acting as an interator.  Iterators have state, and iterating over 
# them (with a for loop) has the side effect of consuming elements. 

def notebook_chunks(file_lines):
    "Iterator which yields chunks in file."
    num, ctype = read_to_first_chunk(file_lines)
    chunk_lines = []
    for line in file_lines:
        marker = is_chunk_marker(line)
        if marker is not None:
            # Found a marker, its the end of the previous chunk
            logging.debug(f"Found {marker}")
            # discard first and last "readability" lines from chunk marker
            chunk_lines = chunk_lines[1:-1]
            yield num, ctype, cleanup_chunk_lines(chunk_lines)
            # Start the next chunk
            chunk_lines = []
            num, ctype = marker
        else:
            chunk_lines.append(line)

def cleanup_chunk_lines(lines):
    # Jupyter maintains a list of source lines, rather than a single long string.
    # Curiously, these lines are also newline terminated, except for the *LAST* line (why?
    # not sure).  There are two ways to denote an empty last line of the cell!  A newline
    # at the end of the last line *OR* an additional empty line.  This function cleans up
    # any confusion.
    if len(lines) > 0:
        lines[-1] = lines[-1].rstrip()
        if lines[-1] == '':
            lines = lines[:-1]
    return lines

def read_to_first_chunk(file_lines):
    "Read to the first chunk marker and return number and type."
    for line in file_lines:
        marker = is_chunk_marker(line)
        if marker is not None:
            # Found a marker, extract a new chunk
            logging.debug(f"Found first marker: {marker}")        
            return marker
    raise Exception("Could not fine a chunk marker.")
            
def empty_code_cell():
    "Create an empty code cell."
    cell = dict(cell_type = 'code',
                execution_count = 1,
                metadata = {},
                outputs = [])
    return cell

def empty_markdown_cell():
    "Create an empty markdown cell."
    cell = dict(cell_type = 'markdown',
                metadata = {})
    return cell

def create_jupyter_cell(chunk_type, lines):
    "Create a cell from a chunk."
    if chunk_type in ['module', 'notebook', 'test']:
        # all of these chunk types map to the code cell type.
        cell = empty_code_cell()
    elif chunk_type == 'markdown':
        cell = empty_markdown_cell()
        # Remove comments from the markdown
        lines = [uncomment_line(l) for l in lines]
    else:
        raise Exception(f"Can't create notebook cell of type = {chunk_type}.")
    if len(lines) > 0:
        # Strip the last newline
        # lines[-1] = lines[-1].rstrip()
        pass
    cell['source'] = lines
    logging.debug(f"Creating cell {chunk_type} length = {len(lines)}")
    return cell

def extract_metadata(lines):
    "Deserialize the JSON in the metadata section."
    lines = [uncomment_line(l) for l in lines]
    return json.loads(" ".join(lines))

#### Cell #23 Type: module #####################################################

def status(notebook_path):
    "Display the status of notebook and associated python files."
    display_markdown(f"### {notebook_path}:")
    rt_status(notebook_path)
    if is_literate_notebook(notebook_path):
        module_status(notebook_path)

def rt_status(notebook_path):
    rt_path = roundtrip_path(notebook_path)
    if os.path.exists(rt_path):
        nb_mod, rt_mod = file_status_helper(notebook_path, rt_path)
        if nb_mod and rt_mod:
            display(f"    Both the notebook and the roundtrip have been modified since the RT was generated.")
            display("       **This is a dangerous situation and requires a hand merge.**")
        elif nb_mod:
            display(f"    The notebook has been updated after the roundtrip was generated.")
        elif rt_mod:
            display(f"    The roundtrip has been updated after the roundtrip was generated.")
        else:
            display(f"    The notebook and roundtrip are in sync.")
    else:
        display(f"    No roundtrip file has been generated: {rt_path}")


def module_status(notebook_path):
    m_path = module_path(notebook_path)
    if os.path.exists(m_path):
        nb_mod, mod = file_status_helper(notebook_path, m_path)
        if nb_mod and mod:
            display(f"    Both the notebook and the module have been modified since the module was generated.")
            display("       **This is a dangerous situation and requires a hand merge.**")
        elif nb_mod:
            display(f"    The notebook has been updated after the module was generated.")
        elif mod:
            display(f"    The module has been updated after the module was generated.")
            display("       **This is a dangerous situation and requires a hand merge.**")                
        else:
            display(f"    The notebook and module are in sync.")
    else:
        display(f"    No module file has been generated: {m_path}")


def make(notebook_path, force_update=False):
    """
    Perform a dependency directed update of the roundtrip and module file associated with
    the notebook.
    """
    if is_literate_notebook(notebook_path):
        status(notebook_path)
        update_roundtrip(notebook_path, force_update=force_update)
        update_module(notebook_path, force_update=force_update)
    elif is_notebook(notebook_path):
        display_markdown(f"### {notebook_path}:")
        rt_status(notebook_path)
        update_roundtrip(notebook_path, force_update=force_update)

#### Cell #24 Type: module #####################################################

# Helpers

def extract_timestamp_from_python(python_path):
    """
    If the python file was serialized with a metadata section AND a timestamp, then read
    the timestamp from the metadata chunk.
    """
    file_lines = read_lines(python_path)
    metadata = None
    for chunk in notebook_chunks(iter(file_lines)):
        num, ctype, lines = chunk
        if ctype == 'metadata':
            metadata = extract_metadata(lines)
    if metadata:
        return extract_timestamp_from_metadata(metadata)
    else:
        return None

def find_literate_notebooks(directory):
    regex = re.compile(".*_Module.ipynb")
    files = os.listdir(directory)
    return [f for f in files if regex.match(f)]

def file_mtime(path):
    return time_from_timestamp(os.path.getmtime(path))

def file_status_helper(nb_path, file_path):
    "Check the modification status of the notebook and the file."

    # Timestamp from file.  This is when the file was created from the notebook.
    tstamp = extract_timestamp_from_python(file_path)
    if tstamp is None:
        raise Exception(f"Cannot access timestamp from {file_path}.")

    # Has the notebook been modified after the file timestamp
    nb_modified = time_after(tstamp, file_mtime(nb_path))

    # Has the file been modified since the stamp
    file_modified = time_after(tstamp, file_mtime(file_path))
    return nb_modified, file_modified

def update_module(notebook_path, force_update=False):
    mpath = module_path(notebook_path)
    if not os.path.exists(mpath):
        logging.info(f"Module missing, creating {mpath}.")
        convert_notebook_to_module(notebook_path, mpath)
    else:
        nb_modified, module_modified = file_status_helper(notebook_path, mpath)
        if module_modified:
            # Someone must have editted the module.  Be careful!
            if force_update:
                logging.warning(f"Module {mpath} has been modified! Forcing! Check backup.")
                backup_file(mpath)                
                convert_notebook_to_module(notebook_path, mpath)
            else:
                tmp_mpath = tmp_path(mpath)
                logging.warning(f"Module {mpath} has been modified! Diff with {tmp_mpath}.")
                convert_notebook_to_module(notebook_path, tmp_mpath)
        else:
            logging.info(f"Module {mpath} not modified. Regenerating.")
            backup_file(mpath)                
            convert_notebook_to_module(notebook_path, mpath)

def update_roundtrip(notebook_path, force_update=False):
    rt_path = roundtrip_path(notebook_path)
    if not os.path.exists(rt_path):
        logging.info(f"Roundtrip missing, creating {rt_path}.")
        convert_notebook_to_roundtrip(notebook_path, rt_path)
    else:
        nb_modified, rt_modified = file_status_helper(notebook_path, rt_path)
        if nb_modified and rt_modified:
            # Someone must have also edited the rt.  Be careful!
            if force_update:
                logging.warning(f"Roundtip {rt_path} has been modified! Forcing! Check backup.")
                backup_file(rt_path)                
                convert_notebook_to_roundtrip(notebook_path, rt_path)
            else:
                tmp_rt_path = tmp_path(rt_path)
                logging.warning(f"Roundtrip {rt_path} has been modified! Diff with {tmp_rt_path}.")
                convert_notebook_to_roundtrip(notebook_path, tmp_rt_path)
        elif nb_modified:
            logging.info(f"Notebook {notebook_path} has been modified. Roundtrip not modified. Updating RT.")
            backup_file(rt_path)                
            convert_notebook_to_roundtrip(notebook_path, rt_path)
        elif rt_modified:
            logging.info(f"Roundtrip file {rt_path} has been modified. Notebook not modified. Updating notebook.")
            backup_file(notebook_path)
            convert_roundtrip_to_notebook(rt_path, notebook_path)
            logging.info(f"Regenerating {rt_path} for consistency.")
            backup_file(rt_path)                
            convert_notebook_to_roundtrip(notebook_path, rt_path)
        else:
            logging.info(f"Neither {notebook_path} nor {rt_path} modified. Pass.")

def force_notebook_update_from_roundtrip(notebook_path):
    "In some cases the roundtrip gets ahead of the notebook, but the general timestamps and file times prevent the natural update.  Forces the update."
    rt_path = roundtrip_path(notebook_path)    
    logging.warning(f"Forcing update of {notebook_path} from {rt_path}.")
    backup_file(notebook_path)
    convert_roundtrip_to_notebook(rt_path, notebook_path)
    logging.info(f"Regenerating {rt_path} for consistency.")
    backup_file(rt_path)                
    convert_notebook_to_roundtrip(notebook_path, rt_path)

#### Cell #27 Type: metadata ###################################################

#: {
#:   "metadata": {
#:     "timestamp": "2020-06-11T18:16:37.522049-07:00"
#:   }
#: }

#### Cell #28 Type: finish #####################################################

