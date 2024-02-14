#!/usr/bin/env python3

import json
import os
import pathlib
import pickle
import random
import shutil
import sys
import time
from typing import Dict, Union, Any, Mapping, MutableMapping, Optional, Sequence

from absl import app
from absl import flags
from absl import logging

sys.path.append("/app/alphafold")
from alphafold.common import protein
from alphafold.common import residue_constants
from alphafold.data import msa_identifiers
from alphafold.data import pipeline
from alphafold.data import templates
from alphafold.data import parsers
from alphafold.data.tools import hhblits
from alphafold.data.tools import hhsearch
from alphafold.data.tools import hmmsearch
from alphafold.data.tools import jackhmmer
import numpy as np

logging.set_verbosity(logging.INFO)
                     
MAX_TEMPLATE_HITS = 20

FeatureDict = MutableMapping[str, np.ndarray]
TemplateSearcher = Union[hhsearch.HHSearch, hmmsearch.Hmmsearch]


def read_msa_from_file(msa_out_path: str, msa_format: str, 
                       max_sto_sequences: Optional[int] = None)-> Mapping[str, Any]:
    logging.warning('Reading MSA from file %s', msa_out_path)
    if msa_format == 'sto' and max_sto_sequences is not None:
        precomputed_msa = parsers.truncate_stockholm_msa(msa_out_path, max_sto_sequences)
        result = {'sto': precomputed_msa}
    else:
        with open(msa_out_path, 'r') as f:
            result = {msa_format: f.read()}
    return result

if __name__ == '__main__':
    
    uniref90_msa_file = sys.argv[1]
    mgnify_msa_file = sys.argv[2]
    bfd_msa_file = sys.argv[3]
    msa_features_out_file = sys.argv[4]
    final_msa_size_file = sys.argv[5]
    
    
    jackhmmer_uniref90_result = read_msa_from_file(msa_out_path=uniref90_msa_file,
                                      msa_format='sto')
    uniref90_msa = parsers.parse_stockholm(jackhmmer_uniref90_result['sto'])
    
    jackhmmer_mgnify_result = read_msa_from_file(msa_out_path=mgnify_msa_file,
                                      msa_format='sto')
    mgnify_msa = parsers.parse_stockholm(jackhmmer_mgnify_result['sto'])
    
    jackhmmer_small_bfd_result = read_msa_from_file(msa_out_path=bfd_msa_file,
                                      msa_format='sto')
    bfd_msa = parsers.parse_stockholm(jackhmmer_small_bfd_result['sto'])
    
    msa_features = pipeline.make_msa_features((uniref90_msa, bfd_msa, mgnify_msa))
    
    with open(final_msa_size_file,'w') as f:
            f.write(str(msa_features['num_alignments'][0])+" sequences")
    
    with open(msa_features_out_file, 'wb') as f:
        pickle.dump(msa_features, f, protocol=4)
