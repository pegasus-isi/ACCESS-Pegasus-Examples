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


class DataPipeline:
    """Runs the alignment tools and assembles the input features."""
    def __init__(self,
               template_searcher: TemplateSearcher):

        """Initializes the data pipeline."""
        self.template_searcher = template_searcher

    def process(self, msa_output_dir: str, uniref90_hits_path: str) -> FeatureDict:
        """Runs alignment tools on the input sequence and creates features."""
        jackhmmer_uniref90_result = read_msa_from_file(msa_out_path=uniref90_hits_path,msa_format='sto')
        msa_for_templates = jackhmmer_uniref90_result['sto']
        msa_for_templates = parsers.deduplicate_stockholm_msa(msa_for_templates)
        msa_for_templates = parsers.remove_empty_columns_from_stockholm_msa(
                            msa_for_templates)
        
        if self.template_searcher.input_format == 'sto':
            pdb_templates_result = self.template_searcher.query(msa_for_templates)
        elif self.template_searcher.input_format == 'a3m':
            uniref90_msa_as_a3m = parsers.convert_stockholm_to_a3m(msa_for_templates)
            pdb_templates_result = self.template_searcher.query(uniref90_msa_as_a3m)
        else:
            raise ValueError('Unrecognized template input format: '
                       f'{self.template_searcher.input_format}')
        
        with open(msa_output_dir, 'w') as f:
            f.write(pdb_templates_result)
        


if __name__ == '__main__':
    
    pdb70_db_path = sys.argv[1]+"/pdb70"
    uniref90_msa_file = sys.argv[2]
    pdb_hits_outfile = sys.argv[3]
    
    template_searcher = hhsearch.HHSearch(
        binary_path=shutil.which('hhsearch'),
        databases=[pdb70_db_path])
    
    """
    template_featurizer = templates.HhsearchHitFeaturizer(
        mmcif_dir=FLAGS.template_mmcif_dir,
        max_template_date=FLAGS.max_template_date,
        max_hits=MAX_TEMPLATE_HITS,
        kalign_binary_path=FLAGS.kalign_binary_path,
        release_dates_path=None,
        obsolete_pdbs_path=FLAGS.obsolete_pdbs_path)"""
    
    data_pipeline = DataPipeline(template_searcher=template_searcher)
    data_pipeline.process(pdb_hits_outfile,uniref90_msa_file)
            
    
    