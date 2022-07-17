#!/usr/bin/env python

import sys
import os
import logging
import utils

description = """
Copy data off of Virb micro SD card.
"""

def set_logging_level(level):
    if isinstance(level, str):
        numeric_level = getattr(logging, level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: level")
        logging.getLogger().setLevel(numeric_level)
    else:
        logging.getLogger().setLevel(level)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--log", help="Logging level", type=str, default='info')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s|%(levelname)s|%(funcName)s| %(message)s')
    set_logging_level(args.log)

    # source_path = args.input

    virb_path = "/Volumes/Virb"
    
    if not os.path.exists(virb_path):
        print(f"Virb is not mounted.")
        sys.exit(0)

    if not os.path.exists("/Volumes/Store"):
        print(f"Big is not mounted.")
        sys.exit(0)
        
    command = f"rsync -av /Volumes/Virb /Volumes/Store"
    res = utils.run_system_command(command)
    if res != 0:
        print(f"Rsync failed with error code: {res}")
        sys.exit(3)

    res = utils.run_system_command(command)
    if res != 0:
        print(f"Rsync failed with error code: {res}")
        sys.exit(3)

    if True:
        print(f"Rsync failed with error code: {res}")
        ls_command = "ls /Volumes/Virb/DCIM/100_VIRB/*"

        res = utils.run_system_command(ls_command)

        rm_command = "rm /Volumes/Virb/DCIM/100_VIRB/*"
        res = utils.run_system_command(rm_command)
        if res != 0:
            print(f"Failed Deleting : {res}")

        res = utils.run_system_command(ls_command)

    unmount_commands = [
        "sync",
        f"diskutil unmountdisk {virb_path}",
        "sync",
        "sleep 1",
        f"diskutil unmountdisk force {virb_path}",
    ]

    for c in unmount_commands:
        utils.run_system_command(c)
