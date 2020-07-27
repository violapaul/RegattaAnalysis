#!/usr/bin/env python
# coding: utf-8

# # Data Collection Pipeline
# 
# **Summary**: This notebook contains a discussion of the data which is captured on boat and how that data makes its way from the boat network, to the Raspberry Pi, and then onto a laptop for analysis.  This is ultimately boring stuff ("just plumbing"), but it is important that this be done right or data will be lost, duplicated, or corrupted.
# 
# If you are interested in the content of this data, please read [Preparing Boat Instrument Data for Analysis](Data_Preparation.ipynb).
# 
# 
# ## Tenets
# 
# Every design should have tenets.  These are high level requirements that potentially impact a large portion of the design.
# 
# - Keep the original raw data. If there is a bug in the processing pipeline then the data can be re-processed.
#   - Keep at least one copy of all data.
# - Process all data, but flag useful vs useless.
# - Only copy changes, to make things fast
# - It should be easy to navigate data
#   - Filenames should make sense and be organized
#   - Additional Databases that cannot be easily hand editted are bad.  YAML?
# - The RPi is ephemeral.  Assume it can be lost!
# 
# 
# ## TODO
# 
# - Eliminate full rsync.  Might be expensive, and could compete with logging.  Not
#     needed.
# - Add a full_backup script that does do a full backup!  Rsync, etc.
# - Delete old logs from RPi.
# - Reprocess all old logs.
# 
# 
# ## User Experience
# 
# - Come home, pull out USB stick.
# - Put in Mac.
# - Run pull data script
#   - Copies all data over to Mac
#   - Finds new logs and processes them.
# 
# 
# ## Design
# 
# - Invariant: All processed logs files are compressed (saves 10x for 300 Megs).
# - Operations are idempotent where possible (you can run them again and again).
#   - Operations compute what would result, and then only trigger if needed.
# - All data kept in a raw logs directory, using RPi filenames
#   - Find the new files on the USB stick which do not have compressed versions.
#   - Copy and then compress.
# - Maintain another directory with links to useful logs, with friendly names (based on datetime and boat name).
# 
# - Keep things simple. Operations are: `op file_in file_out` with an optional `force`.
#   - If `file_out` exists, then you skip the step unless `force`.
# 
# - Operations:
#   - `copy_rpi_log_file(
#   - `extract_file_name(compressed_log_file)`:  Given a raw log, extract a better filename.
#     - Got to be super careful here that we are using the **first** datetime record!
#         Otherwise multiple files can be generated.
#   - `link_file(compressed_log_file, new_path)`
#   - `canboat_to_json(compressed_log_file, json_path)`
#   - `json_to_gpx(json_path, gpx_path)`
#   - `json_to_pandas(json_path, gpx_path)`
# 
# - Orchestration can be done by the file, or by the directory.
#   - Process directory: find all files, and then trigger the pipeline.
#     - User will want this, because it will automaically process all new logs
#   - Or a single file can be processed, most likely with `force`.
# 
# 
# 
# ## Questions
# 
# - what processing on the RPi vs. on the laptop?
#   - generally want to keep the RPi very simple... for now.
# - Some files are in the git repo, others not.  Which ones? Is this good?
# - What about the err files?
# - Do we need to clean up files?  Remove from RPi?
#   - Mostly empty files, from runs at home where there is no canbus
#   - very old files
# 
# 
# ## Facts
# 
# - Current logs go back to about October
# - Raw logs are large, up to 350 Meg, (compressed is 1/10 th)
# - Raw logs are processed into: GPX files, JSON files, and Pandas files.
#   - All are large and could be usefully compressed
#   - JSON is not useful in itself, but is more easily readable than raw.
# - We are unlikely to collect more than 10 real logs a month
# - Logging takes about 1Meg a minute.  Less than 10Meg is not useful.
# 
# 
# 
# 
# 
# ## Ideas
# - Does it make sense to use symlinks instead of renaming of files
# - Or just skip renaming, use the RPi log files, and use a association table.
#   - I have found this disorienting, since the filenames are pretty much random.
# 
# 
# - Tenets
#   - Never delete data permanently
#   - Keep the raw data, there is no telling if a bug in the pipeline 
#   - Process all data, but flag useful vs useless.
#   - Only copy changes, to make things fast
#   - It should be easy to navigate data
#     - Filenames should make sense and be organized
#     - Additional DBs that cannot be easily hand editted are bad.  YAML?
#   - RPi is ephemeral.  Assume it can be lost!
# 
# 
# 

# In[ ]:




