#!/usr/bin/env python

from global_variables import G
import process as p
import metadata
G.init_seattle()

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
        G.logger.warning(f"USB drive not found.  Skipping copy.")
    p.create_compressed_log_files()
    p.create_named_log_files()
    p.create_gpx_and_pandas_files(100000000, 100)

    # Add metadata
    metadata.add_missing_metadata()
    metadata.update_metadata_from_gsheet()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--log", help="Logging level", type=str, default='debug')
    args = parser.parse_args()

    G.set_logging_level(args.log)
    G.logger.info(f"Set log level to {args.log}")

    process_all()
