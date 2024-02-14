#!/usr/bin/env python3

import sys

if __name__ == "__main__":
    uniref90_msa_size = sys.argv[1]
    mgnify_msa_size = sys.argv[2]
    bfd_msa_size = sys.argv[3]
    final_msa_size = sys.argv[4]
    summary_file = sys.argv[5]
    counts = []
    for file in [uniref90_msa_size,mgnify_msa_size,bfd_msa_size,final_msa_size]:
        with open(file,'r') as f:
            counts.append(f.read().split()[0])
    
    with open(summary_file,'w') as f:
        uniref90 = 'Uniref90 MSA size: {} sequences.'.format(counts[0])
        mgnify = 'BFD MSA size: {} sequences.'.format(counts[1])
        bfd = 'MGnify MSA size: {} sequences.'.format(counts[2])
        final = 'Final (deduplicated) MSA size: {} sequences.'.format(counts[3])
        f.write('{}\n{}\n{}\n{}\n'.format(uniref90,mgnify,bfd,final))
        