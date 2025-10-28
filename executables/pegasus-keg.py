#!/usr/bin/env python3

"""
A sample python script to use for in Pegasus example workflows

Usage: pegasus-keg.py [options]
"""

import argparse
from argparse import ArgumentParser
import logging
import os
import io
import shutil
import sys
# Importing socket library
import socket
import time

PSUTIL_FOUND = True
try:
    import psutil
except Exception:
    PSUTIL_FOUND = False

##
#  Copyright 2007-2017 University Of Southern California
#
#  Licensed under the Apache License, Version 2.0 (the 'License');
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing,
#  software distributed under the License is distributed on an 'AS IS' BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
##
__author__ = "Karan Vahi <vahi@isi.edu>"

# --- global variables ----------------------------------------------------------------

prog_dir = os.path.realpath(os.path.join(os.path.dirname(sys.argv[0])))
prog_base = os.path.split(sys.argv[0])[1]  # Name of this program
buffer = io.StringIO()

logger = logging.getLogger("PegasusKeg")


# --- functions ----------------------------------------------------------------


def setup_logger(verbose):
    # log to the console
    console = logging.StreamHandler()

    # default log level - make logger/console match
    logger.setLevel(logging.INFO)
    console.setLevel(logging.INFO)

    # debug - from command line
    if verbose:
        logger.setLevel(logging.DEBUG)
        console.setLevel(logging.DEBUG)

    # formatter
    formatter = logging.Formatter("pegasus-keg %(levelname)s: %(message)s")
    console.setFormatter(formatter)
    logger.addHandler(console)
    logger.debug("Logger has been configured")


def myexit(rc):
    """
    system exit without a stack trace
    """
    try:
        sys.exit(rc)
    except SystemExit:
        sys.exit(rc)


def get_hostname_and_ip():
    host_name = None
    host_ip = None
    try:
        host_name = socket.gethostname()
        host_ip = socket.gethostbyname(host_name)
    except Exception as e:
        logger.error("Unable to get Hostname and IP {}".format(e))
        
    return "Hostname: {} IP Addr: {}".format(host_name, host_ip)


def read_file_to_buffer(filename):
    global buffer
    indent = "\t"
    try:
        buffer.write("{}--- start {} ----\n".format(indent, filename))
        with open(filename, 'r') as file:
            for line in file:
                buffer.write(indent)
                buffer.write(line)
                buffer.write("\n")
        buffer.write("{}--- end {} ----\n".format(indent, filename))
    except Exception as e:
        logger.error("Unable to read input file {} to buffer: ".format(filename) + str(e))
        myexit(1)


def write_buffer_to_file(filename):
    global buffer

    try:
        with open(filename, 'w') as file:
            file.write("===================== contents start {} =====================\n".format(filename))
            file.write(get_hostname_and_ip())
            file.write("\n")
            file.write(buffer.getvalue())
            file.write("===================== contents end   {} =====================\n".format(filename))
    except Exception as e:
        logger.error("Unable to write output to file {}: ".format(filename) + str(e))
        myexit(1)


def mimic_cpu_usage(duration, target_load):
    """
    Mimics CPU usage for a specified duration and target load.

    Args:
        duration (int): Duration in seconds to mimic CPU usage.
        target_load (float): Target CPU load as a percentage (0.0 to 100.0).
    """
    start_time = time.time()
    while time.time() - start_time < duration:
        current_cpu_usage = psutil.cpu_percent(interval=0.1)
        if current_cpu_usage < target_load:
            # Perform some computationally intensive task
            _ = [i ** 2 for i in range(100000)]  # Example task
        else:
            time.sleep(0.1)  # Wait to avoid exceeding the target load


# --- main ----------------------------------------------------------------------------

def parse_args(args):
    parser = ArgumentParser(prog="pegasus-keg.py")

    # add top level arguments
    parser.add_argument(
        "-a", "--app", dest="appname", help="set name of application to something else, default pegasus-keg"
    )
    parser.add_argument(
        "-t", "--sleeptime", dest="sleeptime", help="sleep for 'to' seconds during execution, default 0"
    )
    parser.add_argument(
        "-T", "--spintime", dest="spintime", help="spin for 'to' seconds during execution, default 0"
    )
    parser.add_argument(
        "-i", "--input", dest="inputs", required=True, help="enumerate comma-separated list input to read and copy"
    )
    parser.add_argument(
        "-o", "--output", dest="outputs", required=True, help="enumerate comma-separated list output files to create"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="enable debug logging"
    )

    return parser, parser.parse_args(args)


# --- Entrypoint ---------------------------------------------------------------
def main():
    parser, args = parse_args(sys.argv[1:])

    setup_logger(args.verbose)

    # just log some hostname and ip
    logger.info(get_hostname_and_ip())

    if args.inputs:
        for f in str.split(args.inputs, ","):
            logger.debug("input file pass is {}".format(f))
            read_file_to_buffer(f)
    logger.debug("buffer contents after reading in all input files \n {}".format(buffer.getvalue()))

    if args.sleeptime:
        logger.debug("Sleeping for {} seconds".format(args.sleeptime))
        time.sleep(int(args.sleeptime))

    if args.spintime:
        if PSUTIL_FOUND:
            load = 70
            logger.debug("Spinning for {} seconds at {}% load ".format(args.sleeptime, load))
            mimic_cpu_usage(int(args.spintime), load)
        else:
            logger.info("psutil not found. instead of spinning will just sleep")
            time.sleep(int(args.spintime))

    if args.outputs:
        for f in str.split(args.outputs, ","):
            logger.debug("output file pass is {}".format(f))
            write_buffer_to_file(f)

    logger.info("Generated outputs to {}".format(args.outputs))

    sys.exit(0)


if __name__ == "__main__":
    main()
