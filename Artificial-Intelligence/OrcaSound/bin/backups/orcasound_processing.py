#!/usr/bin/env python3

import argparse
import glob
import logging
import sys
from datetime import datetime
from os import path
from pathlib import Path

import ffmpeg
import m3u8

from create_spectrogram import create_spec_name, save_spectrogram


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
    Converts all `.ts` files from `live.m3u8` to `.wav`.

    All files will have the following format: `%Y-%m-%dT%H-%M-%S.wav`

    Args:
        `input_dir`: Path to the input directory with `.m3u8` playlist and `.ts` files. Should contain Unix timestamp of the stream start.
        `output_dir`: Path to the output directory.
    Returns:
        None
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for input_ts in sorted(glob.glob(path.join(input_dir, "*.ts"))):
        uri = input_ts[input_ts.rfind("/")+1:]
        convert_with_ffmpeg(input_ts, path.join(output_dir, uri))


if __name__ == "__main__":
    logging.basicConfig(
        format="%(levelname)s:%(message)s", stream=sys.stdout, level=logging.INFO
    )
    parser = argparse.ArgumentParser(
        description="Creates spectrogram for each .ts file in the input directory."
    )
    parser.add_argument(
        "input_dir",
        help="Path to the input directory with `.m3u8` playlist and `.ts` files. Should contain Unix timestamp of the stream start.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Path to the output directory for spectrograms. Default is `input_dir`.",
    )
    parser.add_argument(
        "-n",
        "--nfft",
        type=int,
        default=256,
        help="The number of data points used in each block for the FFT. A power 2 is most efficient. Default is %(default)s.",
    )
    args = parser.parse_args()

    convert2wav(path.normpath(args.input_dir), "wav")

    for input_wav in sorted(glob.glob("wav/*.wav")):
        output_fname = create_spec_name(input_wav, args.output)
        save_spectrogram(input_wav, output_fname, args.nfft)
