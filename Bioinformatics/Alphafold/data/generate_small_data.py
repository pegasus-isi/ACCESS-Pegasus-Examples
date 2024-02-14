# (assumes you have biopython installed, e.g. with pip install biopython)

import sys, math
from Bio import SeqIO

def iterator(iterator, batch_size):

    entry = True  # Make sure we loop once
    while entry:
        batch = []
        while len(batch) < batch_size:
            try:
                entry = iterator.__next__()
            except StopIteration:
                entry = None
            if entry is None:
                f
                break # EOF = end of file
            batch.append(entry)
        if batch:
            yield batch

if(len(sys.argv) != 4):
        sys.exit("usage: generate_small_data.py FASTA_FILE NO_OF_SEQUENCES NEW_FILE_NAME")

ffile=sys.argv[1]  #fasta file
chunksize=sys.argv[2] #no. of sequences
filename = sys.argv[3] #file name for newly generated small db

print("Getting records...")
records = SeqIO.parse(open(ffile), "fasta")
print("Records created...")
for i, batch in enumerate(batch_iterator(records, chunksize)):

        with open(filename, "w") as handle:
                count = SeqIO.write(batch, handle, "fasta")
        print("Wrote %i sequences to %s" % (count, filename))
        break
sys.exit("Done.")
