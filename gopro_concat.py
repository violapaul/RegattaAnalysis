#!/usr/bin/env python

import glob
from subprocess import call
import re
import os
import os.path
import shutil

import utils

################ Helper Functions ################
# Gopro files are in a funny order.
# 
#     The base looks like this: 'GOPR9839.MP4'.
#     The subsequent files look like this: 'GP019839.MP4', 'GP029839.MP4', 'GP039839.MP4',
#
# If a big bunch of files do not share the same base then they will not sort correctly.

FFMPEG = "/Users/viola/Bin/ffmpeg"

def rename_small_videos(dryrun=False):
    # mkdir Small; for file in *.LRV; do echo $i; mv "$file" "Small/$(basename "$file" .LRV).MP4"; done
    print("Ensure subdirectory Small")
    if not dryrun:
        utils.ensure_directory("Small")
    small_videos = glob.glob("*.LRV")
    for video in small_videos:
        base, extension = os.path.splitext(video)
        result = os.path.join("Small", base + ".MP4")
        print("Renaming {0} to {1}".format(video, result))
        if not dryrun:
            os.rename(video, result)


def extract_base_number(filename):
    mm = re.match(r"^GOPR(\d\d\d\d).MP4$", filename)
    if mm is None:
        raise Exception("Not a valid gopro base file: {}".format(filename))
    if len(mm.groups()) != 1:
        raise Exception("Not a valid gopro base file: {}".format(filename))
    return mm.groups()[0]


def extract_sequence(filename):
    mm = re.match(r"^GP(\d\d)(\d\d\d\d).MP4$", filename)
    if mm is None:
        raise Exception("Not a valid gopro sequence file: {}".format(filename))
    if len(mm.groups()) != 2:
        raise Exception("Not a valid gopro sequence file: {}".format(filename))
    return mm.groups()


def find_video_sequence(basefile):
    "Given the base filename, find the videos in the sequence in the GoPro order."
    basenum = extract_base_number(basefile)
    # Find the follow on segments
    segments = glob.glob("GP*" + basenum + ".MP4")
    segments.sort()
    for segment in segments:
        segnum, seg_basenum = extract_sequence(segment)
    return (int(basenum), [basefile] + segments)


def flatten_video_files(vid_files):
    for n, video_segments in vid_files:
        for f in video_segments:
            yield f


def find_video_files():
    """
    Find all the GOPRO videos in a directory and sort them in a meaningful order.

    Note, its a bit strange. The sorted order is not correct!
    """
    # Find all the base files... beginnings of videos.
    base_files = glob.glob("GOPR*.MP4")
    base_files.sort()
    for base in base_files:
        yield find_video_sequence(base)


def create_video_list_file(outfile, videos):
    """ffmpeg requires a file listing all the videos to convert."""
    with open(outfile, "w") as f:
        for video_file in videos:
            f.write("file \'%s\'\n" % video_file)


def concat_all(output_file, dryrun=False):
    """
    Concat all the GoPro videos in the directory.  GoPro vides are named oddly... deal
    with the fact that the sorted order is not the correct order.
    """
    print("Concatenating Gopro files")
    sequence_file = "files.txt"
    create_video_list_file(sequence_file, flatten_video_files(find_video_files()))
    command = f"{FFMPEG} -safe 0 -f concat -i {sequence_file} -c copy {output_file}"
    print(command)
    if not dryrun:
        call(command.split())
            
def raw_concat(files, output_file, dryrun=False):
    "Used to concat a set of videos into a single video."
    print("Concatenating {0} files into {1}".format(len(files), output_file))
    if os.path.exists(os.path.join(os.path.realpath(os.getcwd()), output_file)):
        print("Output file {0} exists. Skipping.".format(output_file))
        return
    sequence_file = "files.txt"
    create_video_list_file(sequence_file, files)
    command = f"{FFMPEG} -safe 0 -f concat -i {sequence_file} -c copy {output_file}"
    print(command)
    if not dryrun:
        call(command.split())


def concat_videos(dryrun=False):
    "Used to concat each sequence of GOPRO videos into a single file."
    for num, files in find_video_files():
        output_file = "output{0:04d}.mp4".format(num)
        raw_concat(files, output_file, dryrun)


def timelapse_videos(speedup=20, dryrun=False):
    "Produce a timelapse from a sequence of GOPRO videos into a single file."
    for num, files in find_video_files():
        print("Found {0} files with base {1}".format(len(files), files[0]))
        sequence_file = "files{0:04d}.txt".format(num)
        output_file = "timelapse{0:04d}.mp4".format(num)
        create_video_list_file(sequence_file, files)
        rate = 1.0/speedup
        input_command = f"{FFMPEG} -safe 0 -f concat -i {sequence_file}"
        filter_command = f" -an -filter:v setpts={rate:03.3f}*PTS"
        output_command = f" -c:v h264 -crf 20 {output_file}"
        command = input_command + filter_command + output_command
        print(command)
        print(command.split())
        if not dryrun:
            call(command.split())


################ Core verbs of this script ################

def cleanup_files():
    "Cleans up the original files left behind after GOPRO processing."
    cwd = os.path.realpath(os.getcwd())
    print("About to cleanup, in dir: {0}".format(cwd))

    small_dir = os.path.join(cwd, "Small")
    if utils.yes_no_response("Delete path: {0}".format(small_dir)):
        shutil.rmtree(small_dir)

    videos = glob.glob("GP*.MP4")
    print("About to delete original videos.")
    message = "   {0} files found from {1} to {2}".format(len(videos), videos[0], videos[-1])
    if utils.yes_no_response(message):
        for file in videos:
            path = os.path.join(cwd, file)
            print("   Deleting {0}".format(path))
            os.remove(path)


def process_files(dryrun):
    "Creates small videos."
    rename_small_videos(dryrun)
    savedPath = os.getcwd()
    os.chdir("Small")
    concat_videos(dryrun)
    os.chdir(savedPath)


################################################################    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Processes all gopro videos in current directory. ")
    parser.add_argument('--dryrun', help="Don't do the action, just show some info.", action='store_true')

    parser.add_argument('--process', help='Process Files.', action='store_true')
    parser.add_argument('--cleanup', help='Cleanup files.', action='store_true')
    parser.add_argument('--timelapse', help='Produce time lapse videos.', action='store_true')
    parser.add_argument("--speedup", help="Speed of the timelapse videos.", type=int, default=15)

    parser.add_argument("--concat", help="Simple concatenation of multiple files.",  nargs='*')
    parser.add_argument("--concat_all", help="Simple concatenation of multiple files.", action='store_true')
    parser.add_argument("--output", help="Output for concatenation.",  default="output.mp4")

    args = parser.parse_args()

    if args.process:
        process_files(args.dryrun)

    if args.timelapse:
        timelapse_videos(args.speedup, args.dryrun)

    if args.cleanup:
        cleanup_files()

    if args.concat and len(args.concat) > 0:
        raw_concat(args.concat, args.output, args.dryrun)

    if args.concat_all:
        concat_all(args.output, args.dryrun)

        
