#!/usr/bin/env python3

import argparse
import glob
import logging
import sys
from os import path
from pathlib import Path

import ffmpeg

def convert_with_ffmpeg(input_file, output_file):
    """Converts input file using ffmpeg."""
    try:
        ffmpeg_input = ffmpeg.input(input_file)
        ffmpeg_output = ffmpeg.output(ffmpeg_input, output_file)
        ffmpeg_output.run(capture_stdout=True, capture_stderr=True)
    except ffmpeg.Error as e:
        logging.error(e.stdout.decode("utf8"))
        logging.error(e.stderr.decode("utf8"))
        raise e


def convert2wav(input_dir, output_dir):
    """
    Converts all `.ts` files available in the folder to `.wav`.

    All files will have the following format: `liveXXXX.wav`

    Args:
        `input_dir`: Path to the input directory with `.ts` files.
        `output_dir`: Path to the output directory.
    Returns:
        None
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for input_ts in sorted(glob.glob(path.join(input_dir, "*.ts"))):
        uri = input_ts[input_ts.rfind("/")+1:]
        convert_with_ffmpeg(input_ts, path.join(output_dir, uri).replace(".ts", ".wav"))


if __name__ == "__main__":
    logging.basicConfig(
        format="%(levelname)s:%(message)s", stream=sys.stdout, level=logging.INFO
    )
    parser = argparse.ArgumentParser(
        description="Creates wav for each .ts file in the input directory."
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        default=".",
        help="Path to the input directory with `.ts` files. Default is `.`",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="wav",
        help="Path to the output directory for wavs. Default is `wav`.",
    )
    args = parser.parse_args()

    convert2wav(path.normpath(args.input_dir), args.output_dir)
