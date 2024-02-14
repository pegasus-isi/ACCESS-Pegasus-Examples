#!/usr/bin/env python3

import json
from argparse import ArgumentParser


def merge_predictions(input_files):
    merged_json = None
    json_data = None
    for input_file in input_files:
        with open(input_file, 'r') as f:
            json_data = json.load(f)

        if not merged_json:
            merged_json = json_data
        else:
            for s in json_data:
                if not s in merged_json:
                    merged_json[s] = json_data[s]
                else:
                    for t in json_data[s]:
                        if not t in merged_json[s]:
                            merged_json[s][t] = json_data[s][t]
                        else:
                            merged_json[s][t] += json_data[s][t]

    return merged_json


def main():
    global tree_method, gpu_id, early_stopping_rounds

    parser = ArgumentParser(description="Merge orcasound predictions")
    parser.add_argument("-i", "--input", metavar="INPUT_FILE", nargs='+', help="List of JSON files to be merged.", required=True)
    parser.add_argument("-o", "--output", metavar="OUTPUT_FILE", type=str, default="predictions.json", help="Output file name. Default is `predictions.json`.")

    args = parser.parse_args()

    merged_json = merge_predictions(args.input)

    with open(args.output, 'w') as g:
        json.dump(merged_json, g)


if __name__ == "__main__":
    main()
