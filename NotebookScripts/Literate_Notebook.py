#!/usr/bin/env python
# coding: utf-8

# # Literate Notebook
# 
# ## Motivation
# 
# I've found the most productive way to write and document code is by building it up in a Jupyter notebook.  I use the notebook to define the problem, the inputs and outputs, include examples, references and links, and then build up to a final set of solutions.  Along the way I end up writing functions which are similar or identical to a set functions to be collected into a module (the python name for libraries of functions).  Modules are critical because they can be imported and used in other notebooks/modules.  
# 
# Unfortunately, the notebook and the module can get out of sync.  I'll find a bug in the module as I am using it, and then go fix it.  Or I'll add a bit of functionality.  Gradually the notebook becomes stale.  Most of the info in the notebook is still true, but it doesn't reflect the final details accurately.
# 
# It would be way better if the **Notebook was the Module** and the **Module was the Notebook**.  
# 
# The code in this notebook/module provides a means to unify these concepts.

# ## Background (Optional)
# 
# This notion is clearly related to *Literate Programming* (LP).  Links:
# 
# - [Knuth's Paper](http://www.literateprogramming.com/knuthweb.pdf), great but not that accessible.
# - [Literate Programming Site](http://www.literateprogramming.com/), oddly structured (what is this site trying to achieve?).
# - [Wikipedia Page](https://en.wikipedia.org/wiki/Literate_programming), always a good place to start.
# - [A Gentle Introduction](http://axiom-developer.org/axiom-website/litprog.html), one of the best I have read.
# - [Physics Based Rendering](http://www.pbr-book.org/3ed-2018/contents.html), a great book which uses LP to teach and define a complete working system.
# 
# The "literate programming" idea is that code is a byproduct of the thinking and teaching process.  So rather than write a program with embedded comments, you write a teaching document in Knuth's WEB language which includes the code, using markup that allows you to later tease the two pieces apart into the code (Pascal in his example) and the document (TeX).  The final code itself is not meant for anyone to read... it is only intended for the compiler.  The surprising part for me is that the code is never really presented in one piece, though it could be.  In the WEB document it is sliced and diced into pieces and only assembled at the end (more on this below).  (BTW, Knuth is brilliant and always right, but I am not a huge fan of the look and feel of the languages he developed.  WEB was never widely used, and of course "web" now means something entirely different. TeX, while universal and awesome, has a programming language which kind of sucks.  The algorithms in his books were written in MIX, another painful choice. Python does not suck and neither do Jupyter notebooks.)
# 
# Jupyter notebooks are sort of similar to "literate programs" containing code and documentation side by side.  As they are currently used, notebooks are both better and worse.  They are better because a notebook can include examples,  running code, and code outputs.  A Notebook is not just a dead document.  Notebooks are worse because they are typically not used to create code libraries which are reusable by other modules and other users (this is the ultimate goal of most programming).
# 
# I am arguing for a **Literate Notebook**, a WYSIWYG document including running examples and results, which can be post-processed to create the module which can then be used by other python programs and notebooks.  
# 
# Doesn't it already work that way?  Unfortunately not.  The simplest (boring) reason is that a notebook is a JSON file and not a python module (the notebook extension is `.ipynb`).  This JSON includes markdown cells (like this one) and code cells.  At a minimum we would need to pull the code out of the ipynb file and put it into a `.py` file.  But a typical notebook also includes non-essential code: references to example data, or partially written functionality, or attempts to decompose the problem (like any textbook would).  This code is not welcome in our streamlined and efficient module (though it is super useful when understanding the module).
# 
# Note, jupyter notebooks already include a scheme for converting ipynb files into "python" (`juypyter nbconvert`).  But the conversion is hamfisted and the code not terribly reusable.  And it includes all the code, both the ephemera and the reusable functions.
# 
# My proposal is quite simple (all good ideas are simple, though not all simple ideas are good): add a few tags, harmlessly included in comments, which flag cells as "notebook only" or "destined for the module".  Using some discipline in the notebook authoring process, it is possible to extract the module from the literate notebook, and they magically stay in sync forever.  The fact that the notebook is straightforward JSON helps tremendously.
# 
# Why doesn't everyone do this?  Honestly not sure. The missing piece is actually quite simple.
# 
# Note, the extracted module, while it can stand on its own and will have embedded comments, should not be read/edited directly: *read and understand the code in the literate notebook.*  Over time it is possible folks will edit the module, disconnecting the module from the literate notebook.  At that point the literate notebook loses most of its value, and should be deleted.
# 
# 
# ### Literate Programming is a bit different
# 
# To be honest, LP can be mysterious, and WEB examples seem sort of complex. Central to WEB is a feature missing from the trivially simple Literate Notebook defined above:  *single functions can be decomposed into pieces and described independently.*  The final "tangling" process, takes these pieces and weaves them back together into a single syntactically valid function.  The  Pascal example from Knuth's original paper is a single tangled mess (hence the name tangle for the process of constructing the source code).  Below is the resulting Pascal (its unfair to judge this code too closely, since later tools do a much better job):
# 
#     {1:}{2:}PROGRAM PRINTPRIMES(OUTPUT);CONST M=1000;{5:}RR=50;CC=4;WW=10;{:5}{19:}ORDMAX=30;{:19}VAR{4:}P:ARRAY[1..M]OF INTEGER;{:4}{7:}PAGENUMBER:INTEGER;PAGEOFFSET:INTEGER;ROWOFFSET:INTEGER;C:0..CC;{:7}{12:}J:INTEGER;K:0..M;{:12}{15:}JPRIME:BOOLEAN;{:15}{17:}ORD:2..ORDMAX;SQUARE:INTEGER;{:17}{23:}N:2..ORDMAX;{:23}{24:}MULT:ARRAY[2..ORDMAX]OF INTEGER;{:24}BEGIN{3:}{11:}{16:}J:=1;K:=1;P[1]:=2;{:16}{18:}ORD:=2;SQUARE:=9;{:18};WHILE K<M DO BEGIN{14:}REPEAT J:=J+2;{20:}IF J=SQUARE THEN BEGIN ORD:=ORD+1;{21:}SQUARE:=P[ORD]*P[ORD];{:21}{25:}MULT[ORD-1]:=J;{:25};END{:20};{22:}N:=2;JPRIME:=TRUE;WHILE(N<ORD)AND JPRIME DO BEGIN{26:}WHILE MULT[N]<J DO MULT[N]:=MULT[N]+P[N]+P[N];IF MULT[N]=J THEN JPRIME:=FALSE{:26};N:=N+1;END{:22};UNTIL JPRIME{:14};K:=K+1;P[K]:=J;END{:11};{8:}BEGIN PAGENUMBER:=1;PAGEOFFSET:=1;WHILE PAGEOFFSET<=M DO BEGIN{9:}BEGIN WRITE(’The First ’);WRITE(M:1);WRITE(’ Prime Numbers --- Page ’);WRITE(PAGENUMBER:1);WRITELN;WRITELN;FOR ROWOFFSET:=PAGEOFFSET TO PAGEOFFSET+RR-1DO{10:}BEGIN FOR C:=0 TO CC-1 DO IF ROWOFFSET+C*RR<=M THEN WRITE(P[ROWOFFSET+C*RR]:WW);WRITELN;END{:10};PAGE;END{:9};PAGENUMBER:=PAGENUMBER+1;PAGEOFFSET:=PAGEOFFSET+RR*CC;END;END{:8}{:3};END.{:2}{:1}
# 
# I think the message is clear, don't look at the code.  If you did **have** to look at it the particular the bit between the pair of comments `{7:} ... {:7}` is *defined* and described in section 7 of the WEB document.
# 
#     {7:} PAGENUMBER:INTEGER;PAGEOFFSET:INTEGER;ROWOFFSET:INTEGER;C:0..CC;{:7}
# 
# Perhaps the simplest way to think of LP using WEB: 
# 
# - Code can be broken down into the tiny pieces.
# - Rather than forcing the programmer to squeeze comments into the code itself, the comments **and the code** are defined and discussed elsewhere.
# - The code can be defined and documented in any order.
# - Everything is assembled at the end.  The document looks great.  The actual final code is left a bit mysterious.
# 
# I guess I write programs in a different way...  by writing programs. How would debugging of this final code work? Or how can I be sure this sliced and diced program even worked at all?  I often need to run code to see what I've missed.  Corner cases.  Missing steps.  One of the greatest things about programming is that the compiler/interpreter will find your bugs by failing to do what you intended.  I am not sure how that feedback loop works with WEB. And modern IDE's find a lot of issues very early (like missing references and syntax errors). 
# 
# I am proposing something simpler and less powerful, but more closely associated with typical programming.  Every modern programming language allows you to break computations into pieces through the use of functions.  Each function is syntactically separate and can be described independently.  I imagine Literate Notebooks using functions to decompose and document functionality.  I also believe you can build up functionality through partial or even failed attempts.  These can all go in the notebook and provide a scheme for understanding the final set of functions.
# 
# I am pretty sure I would **not** like programming in WEB.  Expressions interact in complex ways, mostly through variable definitions that are shared in the various namespaces of the function.  It can be further complex if the syntax of the language does not simply allow you to string together subpieces.  Using LP and WEB is likely easier in a language like lisp; since the syntax is trivial, the pieces can be easily glued back together.  Its even easier in a functional subset of lisp (like [Clojure](https://clojure.org/)), because then the subpieces only interact through the values of the expressions (no side effects).  LP for Clojure might work for me. 
# 
# I honestly got lost as Knuth wrote a Pascal function in WEB.  It was super hard to just see the whole structure of the function, and I was worried that it would not just glue back together in a meaningful way.  I kept asking, *can't I just see the function*?
# 
# **How would I fix it?**  Might as well jot this down.  
# 
# - Have the final function available at all times, nicely formatted (not tangled!).  Perhaps in a window at the side.
# - Allow access to the associated documentation by hovering over the code.
# - One click to run on test data.
# - Allow editing of the code directly, and this updates the WEB doc.
# 
# I think this could work.  But I am not sure I'd love it.  I still write code by writing code.

# ## Code Overview
# 
# The "cell" structure of Jupyter notebook's is central. Each cell is either code or markdown.  We will go further to define sub-types of code cells: notebook and module. Code cells containing the a comment `#notebook` are intended for the notebook only, and are created to explain concepts or to provide examples.  The remaining code cells are "module cells" and contain code destined for the module.  These are the final functions that will be used to implement the required functionality and solve the problem at hand.  
# 
# Markdown cells, which are there for exposition are not included in the module.  The exception is the first markdown cell, which is copied directly to the module documentation string.
# 
# I hate to get all meta, but this notebook is an example of a literate notebook.  
# 
# ## Caveats
# 
# ### Make sure the module code is complete
# 
# If you define a function or constant in a *notebook* cell, you **cannot** use it in a *module* cell.  It will all work just fine in the literate notebook, but when the notebook cells are stripped away, the module will not have access to those symbols.  Unfortunately there is nothing about the Jupyter notebook which will prevent this.  It can lead to bugs that will only appear when the module is loaded and run. This can also lead to some duplication, where the same values are defined both in the notebook and module cells.
# 
# This is tricky to fix... it can only be avoided by being careful.
# 
# ### Jupyter is not the greatest development environment
# 
# Jupyter is designed for lightweight programming, and it is not an IDE. Debugging in Jupyter is suspect. Tracing around in large codebases is hard.  There is no highlighting of syntax errors. Global edit/replace and refactor are marginal. Some programming is just best done in an industrial grade IDE (I use emacs with lots of python add-ins).  
# 
# To support this, literate notebooks will provide a *round-trip* functionality as well.  This extracts all of the cells into a single python file which can be loaded into any editor.  This text file can then be executed, edited, or modified in flexible ways.  When you're done you convert it back to a literate notebook where it can resume its role a beautiful web enabled WYSIWYG experience.
# 
# ### How do you handle references to modules
# 
# There is a desire to reuse and share functions in modules.  That is the point of defining literate notebooks, right?  ... so others can reuse the code.  But how does that work into the "teaching document" idea.  
# 
# Do you link to the related notebook?  Can you link directly to a portion of a notebook?  Just skip it?
# 
# In an IDE I would hover and see the doc (or browse) to the definition.
# 
# ### Can Jupyter Help?
# 
# - Color the different types of code cells?
# - Can some cells by collapsed by default?  
# 
# ### TODO
# 
# Some of the cells below are for testing functionality.  These could go in the module, or stay only in the notebook, or go somewhere else.  
# 
# Figure out if there is a role for testing cells (collapsed by default?).

# In[1]:


# Load some modules
import os               # For paths and files.
import json             # Notebook files are in JSON
import itertools as it  # Useful tools for manipulating streams
import re               # regular expressions
import logging          # logging of errors, warnings, etc.
import copy


# In[2]:


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

def read_lines(path):
    "Read the lines of a file specified by PATH."
    with open(path, 'r') as fs:
        for line in fs:
            yield line

def save_lines(file_path, lines):
    "Save lines to file."
    with open(file_path, 'w') as fs:
        for line in lines:
            fs.write(line)


# In[3]:


# notebook - first example of a "notebook" cell.  

# Some notebook specific libraries for display
from nbutils import display_markdown, display

# Setup logging at debug, so we can see what is going on.
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s|%(levelname)s|%(funcName)s| %(message)s')


# In[4]:


# notebook - start drilling down into the structure of notebooks.

# A notebook to analyze.
nb_file = "Literate_Notebook.ipynb"      # going all meta here,  this is the current notebook!
# nb_file = "Very_Short_Notebook.ipynb"  # a shorter notebook, handy for testing.

# Load the JSON of the notebook
nb_json = read_notebook(nb_file)
    
display_markdown("### Notebooks have a few top level keys.")
display_markdown(repr(list(nb_json.keys())))
display_markdown("The code/markdown is in `cells`. Metadata in the other keys.")

display_markdown("### Example metadata.")
display({k:v for k, v in nb_json.items() if k != 'cells'})  # Display all keys but 'cells'
display_markdown("To be honest, this info is mysterious.  We'll simply copy it when we implement round trip.")


# In[5]:


# notebook - cells are where the action is

display_markdown("### Cell information")
display_markdown("Here is the contents of one cell.  Note, picking a smaller cell.")
display(nb_json['cells'][3])

# Let's print out the content of the cells as well...  just the first 3 lines.  
display_markdown("### Data from cells: first 3 lines.")

line_length = 80
# print the first 3 lines of the first 5 cells
for num, cell in enumerate(nb_json['cells'][:5]):
    ctype = cell['cell_type']
    source_code = cell['source']
    print(f"################## Cell #{num}: {ctype} {'#' * line_length}"[:line_length])
    print("".join(source_code[:3]))
    print("...")


# In[6]:


# Code will process each cell in turn, handling markdown and code cells differently.
# Code cells are further differentiated by their cell tag.  
VALID_CELL_TAGS = {'notebook', 'module', 'test'}

# The cell tag is the first word on the first line.
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


# In[7]:


# notebook - this notebook cell defines the correct behavior and tests

def test_regex_cell_tag():
    "Test some options for notebook tags.  Check that they work as expected."
    possibilities = [
        ['notebook', "# notebook \n"],
        ['notebook', "# NoteBook - an example of an \n"],
        ['notebook', "#notebook"],
        ['test',     "#  Test  "],
        ['module',   "foobar"],
        ['module',   "# baz"]
    ]    
    test = True
    for tag, p in possibilities:
        ctag = extract_cell_tag(p)
        m = tag == ctag
        logging.debug(f"{m} = {tag} == {ctag}, {repr(p)}")
        test = test and m
    assert test

test_regex_cell_tag()


# In[8]:


# Recall there are two goals 
#  1) Create a python module from a literate notebook. 
#  2) Roundtrip from literate notebook to python and back.
#
# The two solutions will end up using many of the same pieces.
#
# Each cell will be rendered as a text "chunk", and the chunks are delimeted by special comment lines.
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


# In[9]:


# notebook - test that the cell markers 

def test_chunk_marker():
    data = it.product(range(5), ['module', 'notebook', 'test'])
    test = True
    for num, ctype in data:
        line = make_chunk_marker(num, ctype)
        m = is_chunk_marker(line)
        logging.debug(f"{m}, {line}")
        if m is not None:
            lnum, ltype = m
            test = test and (num == lnum) and (ctype == ltype)
        else:
            test = False
    return test

test_chunk_marker()


# In[10]:


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
    return res


# In[11]:


# notebook - let's try this out

# Read a notebook from a file as JSON
nb_json = read_notebook(nb_file)
# Generate the python module as a list of lines.
module_lines = notebook_to_module(nb_json, use_chunks=True)

# Print the first 100 lines
for line in module_lines[:100]:
    print(line, end="")
    
# The result should be valid python, and suitable for use as a python module.


# In[12]:


# notebook - let's examine the results of this synthesis

def tmp_path(filename):
    return os.path.join("/tmp", filename)

# Save the result to a temporary python file.
save_lines(tmp_path("test_module.py"), module_lines)
        
# I would recommend opening this file in your favorite IDE!


# In[13]:


# Working toward roundtrip.

# Unlike "compile to module", roundtrip will attempt to store the contents of all cells. 
# To encode markdown regions we'll use python comments.
COMMENT_TEXT = "#: "

def comment_line(line):
    "Turn a line into a Python comment line."
    return COMMENT_TEXT + line

REGEX_WHITESPACE_ONLY = re.compile("^\s*$")

def uncomment_line(line):
    "Take a commented line and remove the comment."
    if REGEX_WHITESPACE_ONLY.match(line):
        return line
    if line.startswith(COMMENT_TEXT):
        return line[len(COMMENT_TEXT):]
    else:
        raise Exception(f"Attempt to uncomment a line which is not commented.")


# In[14]:


# notebook - test commenting and uncommenting on the current notebook!

print(comment_line("This line should get commented!"))

def test_commenting(notebook_json):
    "For evey line in every cell in NOTEBOOK_JSON, first comment and then uncomment."
    test = True
    count = 0
    for num, cell in enumerate(notebook_json['cells']):
        source = cell.get('source', [])
        logging.debug(f"Testing cell {num} with {len(source)} lines.")
        for line in source:
            count += 1
            cline = comment_line(line)
            uline = uncomment_line(cline)
            match = (line == uline)
            if not match:
                logging.warning(f"Lines do not match")
                logging.warning(line)
                logging.warning(uline)
                test = False
    logging.info(f"Tested {count} lines.")
    return test

test_commenting(nb_json)


# In[15]:


# Serialize a notebook into a roundtrip-able python file.

# In the interest of being more like a "literate program", let's present the top-level function first.
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
            loggging.warning(f"Found a ctype = {ctype} in cell #{num} which is not supported.")
    # Finally add a chunk that includes notebook metadata
    add_metadata_chunk(res, num+1, notebook_json)
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
    "Extract metadata portion of the the notebook and serialize as JSON."
    # metadata is everything but the cells
    metadata = copy.copy(notebook_json)
    del metadata['cells']
    # Format so it is human readable and editable, just in case.
    jstring = json.dumps(metadata, indent=2)
    return jstring.splitlines(True)

def add_metadata_chunk(res, num, notebook_json):
    add_lines(res, make_chunk_separator(num, 'metadata'))
    json_of_metadata = notebook_metadata(notebook_json)
    # Comment each line of JSON
    add_lines(res, [comment_line(l) for l in json_of_metadata])


# In[16]:


# notebook - Create a roundtrip python file.

rt_lines = notebook_to_roundtrip(nb_json)

# This is all the data in the notebook: module, notebook, markdown.
for line in rt_lines[:100]:
    print(line, end="")


# In[29]:


# Code to read a rountrip notebook.  Once again starting at the top-level and working down.

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
            logging.info(f"Found {marker}")
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
            logging.info(f"Found first marker: {marker}")        
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
    logging.info(f"Creating cell {chunk_type} length = {len(lines)}")
    return cell

def extract_metadata(lines):
    "Deserialize the JSON in the metadata section."
    lines = [uncomment_line(l) for l in lines]
    return json.loads(" ".join(lines))


# In[30]:


# roundtrip-ing functions

def module_filename(notebook_filename):
    """
    By convention notebook names destined for module-hood will be Capitilized and
    underscore delimited.  The conversion downcases.  Double check for spaces and invalid
    characters, since that would screw things up when defining a module.
    """
    base, extension = os.path.splitext(notebook_filename)
    regex = re.compile(r"[\s\W]+")  # Matches non-alphanumeric and whitespace
    if regex.search(base):
        raise Exception("Filename: {notebook_filename} contains invalid characters.")
    module_filename = base.lower() + ".py"
    return module_filename

def convert_notebook_to_roundtrip(notebook_path, output_path):
    "Create a roundtrip file."
    notebook = read_notebook(notebook_path)
    save_lines(output_path, notebook_to_roundtrip(notebook))

def convert_notebook_to_module(notebook_path):
    "Given a notebook (ipynb), extract the module and write file."
    path, filename = os.path.split(notebook_path)
    output_filename = module_filename(filename)
    notebook = read_notebook(notebook_path)
    logging.info(f"Creating module {output_filename}.")
    save_lines(os.path.join(path, output_filename), notebook_to_module(notebook))


def convert_roundtrip_to_notebook(roundtrip_path, notebook_path):
    lines = read_lines(roundtrip_path)
    notebook = notebook_from_roundtrip(lines)
    save_notebook(notebook, notebook_path)


# In[ ]:





# In[31]:


# notebook - some end to end tests

# Note, in Python '=='' is a deep comparison by value (btw, very expensive).  But if it fails you don't 
# know where, so we'll build custom comparison.  Additionally, we do not gaurantee to roundtrip 
# some of the cell fields (in particular the execution_count and output fields).

def compare_notebooks(n1, n2):
    "Compare the structure and content of two notebooks."
    test = True
    for key in n1.keys():
        if key == 'cells':
            # Handle 'cells' differently, since we don't track execution count and output
            test = test and compare_all_cells(n1[key], n2[key], [key])
        else:
            test = test and compare_objects(n1[key], n2[key], [key])
    return test

# Comparing deep recursive structures is done most simply using recursion.  But then it can be tricky
# to understand where exactly the comparison failed.  In the functions below, PATH is the depth first
# path to the current comparison (for debugging purposes).

# Tests like these can fail early, on the first mismatch, or fail late, continuing to find all 
# mismatches.  We'll fail late and find as many as possible.

def compare_all_cells(cell_list1, cell_list2, path):
    "Compare all cells."
    test = True
    for num, (item1, item2) in enumerate(zip(cell_list1, cell_list2)):
        # Compare each cell, but skips some keys
        eq = compare_dicts(item1, item2, path + [num], skip_keys={'execution_count', 'outputs'})
        if not eq:
            logging.info(f"Cell not equal. Types {item1['cell_type']} {item2['cell_type']}")
        test = test and eq
    if len(cell_list1) != len(cell_list2):
        logging.info(f"Not equal: {path} lengths {len(cell_list1)} != {len(cell_list2)}")
        return False
    return test
    
def compare_objects(o1, o2, path):
    # recursively walk until bottoming
    if type(o1) == dict and type(o2) == dict:
        return compare_dicts(o1, o2, path)
    if type(o1) == list and type(o2) == list:
        return compare_lists(o1, o2, path)
    elif o1 == o2:
        return True
    else:
        logging.info(f"Not equal: {path} o1: {type(o1)}, o2: {type(o2)}")
        return False

def compare_lists(l1, l2, path):
    test = True
    # Check all items in common
    for num, (item1, item2) in enumerate(zip(l1, l2)):
        eq = compare_objects(item1, item2, path + [num])
        test = test and eq
    # Also check lengths, zip will just stop 
    if len(l1) != len(l2):
        logging.info(f"Not equal: {path} lengths {len(l1)} != {len(l2)}")
        test = False
    if not test:
        if len(l1) > 0:
            logging.info(f"L1 First element: |{l1[0]}|")
            logging.info(f"L1 Last  element: |{l1[-1]}|")
        if len(l2) > 0:
            logging.info(f"L2 First element: |{l2[0]}|")
            logging.info(f"L2 Last  element: |{l2[-1]}|")
    return test
    
def compare_dicts(d1, d2, path, skip_keys={}):
    test = True
    all_keys = set(d1.keys()).union(d2.keys())  # keys in both dicts
    test_keys = all_keys.difference(skip_keys)  # remove those to skip
    for key in test_keys:
        if key in d1 and key in d2:
            test = test and compare_objects(d1[key], d2[key], path + [key])
        elif key in d1:
            logging.info(f"Not equal: {path}  {key} missing from d2")
            test = False
        else:
            logging.info(f"Not equal: {path}  {key} missing from d1")
            test = False
    return test


# In[33]:


# notebook

if True:
    # Be super careful here
    convert_notebook_to_module("Literate_Notebook.ipynb")
    convert_notebook_to_roundtrip("Literate_Notebook.ipynb", tmp_path("test_roundtrip.py"))
    convert_roundtrip_to_notebook(tmp_path("test_roundtrip.py"), tmp_path("duplicate_Literate_Notebook.ipynb"))

    original = read_notebook("Literate_Notebook.ipynb")
    duplicate = read_notebook(tmp_path("duplicate_Literate_Notebook.ipynb"))
    
    display(compare_notebooks(original, duplicate))
    
if False:
    # Careful here!  Use this only if you've edited the roundtrip and would like to update
    # the notebook.
    convert_roundtrip_to_notebook(tmp_path("test_roundtrip.py"), "Literate_Notebook.ipynb")


# ## TODO
# 
# - Write script that uses these tools to extract modules.
#   - Makefile
# - What is the right workflow around roundtrip-ing.
#   - It would be great if you could just export a roundtrip file form Jupyter
#   - And then would you reload it somehow?
