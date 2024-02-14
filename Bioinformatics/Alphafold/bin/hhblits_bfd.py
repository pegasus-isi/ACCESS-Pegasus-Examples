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


def run_msa_tool(msa_runner, input_fasta_path: str, msa_out_path: str,
                 msa_format: str, use_precomputed_msas: bool,
                 max_sto_sequences: Optional[int] = None
                 ) -> Mapping[str, Any]:
    """Runs an MSA tool, checking if output already exists first."""
    if not use_precomputed_msas: 
        if msa_format == 'sto' and max_sto_sequences is not None:
            result = msa_runner.query(input_fasta_path, max_sto_sequences)[0]  # pytype: disable=wrong-arg-count
        else:
            result = msa_runner.query(input_fasta_path)[0]
        with open(msa_out_path, 'w') as f:
            f.write(result[msa_format])
    else:
        logging.warning('Reading MSA from file %s', msa_out_path)
        if msa_format == 'sto' and max_sto_sequences is not None:
            precomputed_msa = parsers.truncate_stockholm_msa(
                  msa_out_path, max_sto_sequences)
            result = {'sto': precomputed_msa}
        else:
            with open(msa_out_path, 'r') as f:
                result = {msa_format: f.read()}
    return result

class DataPipeline:
    """Runs the alignment tools and assembles the input features."""
    def __init__(self,
               jackhmmer_binary_path: str,
               hhblits_binary_path: str,
               bfd_database_path: Optional[str],
               uniclust30_database_path: Optional[str],
               small_bfd_database_path: Optional[str],
               use_small_bfd: bool,
               use_precomputed_msas: bool = False):

        """Initializes the data pipeline."""
        self._use_small_bfd = use_small_bfd
        if use_small_bfd:
            self.jackhmmer_small_bfd_runner = jackhmmer.Jackhmmer(
                  binary_path=jackhmmer_binary_path,
                  database_path=small_bfd_database_path)
        else:
            self.hhblits_bfd_uniclust_runner = hhblits.HHBlits(
                  binary_path=hhblits_binary_path,
                  databases=[bfd_database_path, uniclust30_database_path])
        self.use_precomputed_msas = use_precomputed_msas

    def process(self, input_fasta_path: str, msa_output_dir: str, bfd_size_file: str) -> FeatureDict:
        """Runs alignment tools on the input sequence and creates features."""
        with open(input_fasta_path) as f:
            input_fasta_str = f.read()
        input_seqs, input_descs = parsers.parse_fasta(input_fasta_str)
        if len(input_seqs) != 1:
            raise ValueError(
              f'More than one input sequence found in {input_fasta_path}.')
        input_sequence = input_seqs[0]
        input_description = input_descs[0]
        num_res = len(input_sequence)

        if self._use_small_bfd:
            bfd_out_path = msa_output_dir
            jackhmmer_small_bfd_result = run_msa_tool(
                  msa_runner=self.jackhmmer_small_bfd_runner,
                  input_fasta_path=input_fasta_path,
                  msa_out_path=bfd_out_path,
                  msa_format='sto',
                  use_precomputed_msas=self.use_precomputed_msas)
            bfd_msa = parsers.parse_stockholm(jackhmmer_small_bfd_result['sto'])
        else:
            bfd_out_path = msa_output_dir
            hhblits_bfd_uniclust_result = run_msa_tool(
                  msa_runner=self.hhblits_bfd_uniclust_runner,
                  input_fasta_path=input_fasta_path,
                  msa_out_path=bfd_out_path,
                  msa_format='a3m',
                  use_precomputed_msas=self.use_precomputed_msas)
            bfd_msa = parsers.parse_a3m(hhblits_bfd_uniclust_result['a3m'])
            
        with open(bfd_size_file,'w') as f:
            f.write(str(len(bfd_msa))+" sequences")
        

if __name__ == '__main__':
    
    input_seq_file = sys.argv[1]
    bfd_db_file = sys.argv[2]
    bfd_msa_output_file = sys.argv[3]
    bfd_size_file = sys.argv[4]
    
    data_pipeline = DataPipeline(
      jackhmmer_binary_path=shutil.which('jackhmmer'),
      hhblits_binary_path=shutil.which('hhblits'),
      small_bfd_database_path=bfd_db_file,
      bfd_database_path = None,
      uniclust30_database_path = None,
      use_small_bfd=True)
      
    data_pipeline.process(input_seq_file,bfd_msa_output_file,bfd_size_file)

