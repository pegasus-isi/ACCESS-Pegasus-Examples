#!/usr/bin/env python3

import glob
import argparse
import os, sys

parser = argparse.ArgumentParser()
parser.add_argument("prefix")
args = parser.parse_args()
prefix = args.prefix


def main():
    for name in glob.glob('*.png'):
        os.rename(name, prefix + '_' + name)

if __name__ == "__main__":
    main()