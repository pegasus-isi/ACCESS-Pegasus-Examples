#!/usr/bin/env python3

import pathlib
import pickle
import sys
from typing import Dict, Union, Any, Mapping, MutableMapping, Optional, Sequence


from absl import logging

sys.path.append("/app/alphafold")

from alphafold.data import pipeline
from alphafold.data import parsers

import numpy as np

logging.set_verbosity(logging.INFO)


if __name__ == "__main__":
    
    input_fasta_path = sys.argv[1]
    seq_features_file = sys.argv[2]
    
    with open(input_fasta_path) as f:
        input_fasta_str = f.read()
    input_seqs, input_descs = parsers.parse_fasta(input_fasta_str)
    if len(input_seqs) != 1:
        raise ValueError(
          f'More than one input sequence found in {input_fasta_path}.')
    input_sequence = input_seqs[0]
    input_description = input_descs[0]
    num_res = len(input_sequence)
    sequence_features = pipeline.make_sequence_features(
        sequence=input_sequence,
        description=input_description,
        num_res=num_res)
    
    with open(seq_features_file, 'wb') as f:
        pickle.dump(sequence_features, f, protocol=4)
        