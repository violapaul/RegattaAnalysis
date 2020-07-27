#!/usr/bin/env python

import os
import logging
import literate_notebook

description = """
Convert a literate notebook to a module and/or a roundtrip.
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
    # parser.add_argument("input", help='Literate notebook to process.')
    parser.add_argument('--all', help='Show status for all modules.', action='store_true')
    parser.add_argument('--status', help='Show file status.', type=str)
    parser.add_argument('--make', help='Update generated files..', type=str)
    parser.add_argument('--reverse', help='Force overwrite of Literate Notebook from roundtrip file.', type=str)
    parser.add_argument("--log", help="Logging level", type=str, default='info')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s|%(levelname)s|%(funcName)s| %(message)s')
    set_logging_level(args.log)

    # source_path = args.input

    if args.all:
        notebooks = literate_notebook.find_literate_notebooks(".")
        notebooks = sorted(notebooks, key = lambda n: os.path.getmtime(n))
        for nb in notebooks:
            try:
                literate_notebook.status(nb)
                print("")
            except Exception as e:
                print(f"Exception: {str(e)}.")
    
    if args.status:
        literate_notebook.status(args.status)

    if args.make:
        literate_notebook.make(args.make)

    if args.reverse:
        literate_notebook.force_notebook_update_from_roundtrip(args.reverse)
