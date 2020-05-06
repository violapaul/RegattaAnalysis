#!/usr/bin/env python


import utils

from global_variables import G
import literate_notebook

description = """
Convert a literate notebook to a module and/or a roundtrip.
"""

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("input", help='Literate notebook to process.')
    parser.add_argument('--module', help='Create a module.', action='store_true')
    parser.add_argument('--roundtrip', help='Create a roundtripable python file.')
    parser.add_argument('--reverse', help='Convert roundtrip to notebook.')    
    parser.add_argument("--log", help="Logging level", type=str, default='debug')
    args = parser.parse_args()

    G.set_logging_level(args.log)
    G.logger.info(f"Set log level to {args.log}")

    if args.module:
        module_path = literate_notebook.module_path(args.input)
        G.logger.info(f"Creating module {module_path}")
        utils.backup_file(module_path)
        literate_notebook.convert_notebook_to_module(args.input, module_path)

    if args.roundtrip:
        literate_notebook.convert_notebook_to_roundtrip(args.input, args.roundtrip)

    if args.reverse:
        literate_notebook.convert_roundtrip_to_notebook(args.input, args.reverse)

