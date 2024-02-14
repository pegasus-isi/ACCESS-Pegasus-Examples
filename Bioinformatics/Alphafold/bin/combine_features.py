#!/usr/bin/env python3

import sys
import pickle


if __name__ == "__main__":
    sequence_features_file = sys.argv[1]
    msa_features_file = sys.argv[2]
    pdb70_hits = sys.argv[3]
    features_file = sys.argv[4]
    
    with open(sequence_features_file, 'rb') as f1:
        sequence_features = pickle.load(f1)
    
    with open(msa_features_file, 'rb') as f2:
        msa_features = pickle.load(f2)
        
    combined_features = {**sequence_features, **msa_features}
    
    with open(features_file, 'wb') as f3:
        pickle.dump(combined_features, f3, protocol=4)
    