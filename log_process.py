#!/usr/bin/env python

import os
import logging

import global_variables
G = global_variables.init_seattle()

import canboat as cb
import process as p 


description = [
    "Convert a raw canboat log to a json file and then both to a GPX and a PANDAS pickle",
    "file.  Extract date/time along the way and use it for the filenames."
]

description = "\n".join(description)


def process_all():
    if p.usb_drive_available():
        p.copy_err_files_from_usb()
        p.copy_log_files_from_usb()
    else:
        logging.warning(f"USB drive not found.  Skipping copy.")
    p.create_compressed_log_files()
    p.create_named_log_files()
    p.create_gpx_and_pandas_files(100000000, 100)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--log", help="Logging level", type=str, default='warning')
    args = parser.parse_args()

    numeric_level = getattr(logging, args.log.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {args.log}")

    # Remove all handlers associated with the root logger object.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(level=numeric_level,
                        format='%(asctime)s|%(levelname)s|%(funcName)s| %(message)s')

    process_all()
