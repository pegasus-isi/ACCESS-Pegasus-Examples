#!/usr/bin/env python3

from Pegasus.api import *
from pathlib import Path
import logging
import argparse
import sys
import os

logging.basicConfig(level=logging.DEBUG)


def generate_wf(input_fasta_path: str, 
                uniref90_db_path: str,
                pdb70_db_path: str,
                mgnify_db_path: str,
                bfd_db_path: str,
                use_large_bfd: bool,
                use_psc: bool
               ):
    # --- Properties ---------------------------------------------------------------
    props = Properties()
    props["pegasus.monitord.encoding"] = "json"                                                                    
    props["pegasus.transfer.links"] = "true"
    props["pegasus.transfer.bypass.input.staging"] = "true"
    props["pegasus.data.configuration"] = "nonsharedfs"                                                                    
    props.write()
    
    # --- Input files locations ---------------------------------------------------
    INPUT_FASTA_FILE = input_fasta_path
    UNIREF90_DB_PATH = uniref90_db_path
    PDB70_DB_DIR = pdb70_db_path
    MGNIFY_DB_PATH = mgnify_db_path
    BFD_DB_PATH = bfd_db_path
    EXECUTE_SITE_USERNAME =  ""       #Enter the username
    EXECUTE_SITE_DIR = ""             #Enter the path to directory on execute site (PSC)
    SSH_KEY_FILE = ""                 #Enter the path to ssh private key on submit host
    CONTAINER_PATH = ""               #Either a docker URL or Path to .sif file on execute site (PSC)
    
    if use_psc:
        EXECUTE_SITE = "psc"
    else:
        EXECUTE_SITE = "condorpool"
    
    # --- Replicas -----------------------------------------------------------------
    rc = ReplicaCatalog()

    protein_sequence_input = File("GA98.fasta")
    rc.add_replica(EXECUTE_SITE,protein_sequence_input,INPUT_FASTA_FILE)

    uniref90_db = File("uniref90.fasta")
    rc.add_replica(EXECUTE_SITE,uniref90_db,UNIREF90_DB_PATH)

    pdb1 = File("md5sum")
    rc.add_replica(EXECUTE_SITE,pdb1,PDB70_DB_DIR/"md5sum")

    pdb2 = File("pdb70_a3m.ffdata")
    rc.add_replica(EXECUTE_SITE,pdb2,PDB70_DB_DIR/"pdb70_a3m.ffdata")

    pdb3 = File("pdb70_a3m.ffindex")
    rc.add_replica(EXECUTE_SITE,pdb3,PDB70_DB_DIR/"pdb70_a3m.ffindex")

    pdb4 = File("pdb70_clu.tsv")
    rc.add_replica(EXECUTE_SITE,pdb4,PDB70_DB_DIR/"pdb70_clu.tsv")

    pdb5 = File("pdb70_cs219.ffdata")
    rc.add_replica(EXECUTE_SITE,pdb5,PDB70_DB_DIR/"pdb70_cs219.ffdata")

    pdb6 = File("pdb70_cs219.ffindex")
    rc.add_replica(EXECUTE_SITE,pdb6,PDB70_DB_DIR/"pdb70_cs219.ffindex")

    pdb7 = File("pdb70_hhm.ffdata")
    rc.add_replica(EXECUTE_SITE,pdb7,PDB70_DB_DIR/"pdb70_hhm.ffdata")

    pdb8 = File("pdb70_hhm.ffindex")
    rc.add_replica(EXECUTE_SITE,pdb8,PDB70_DB_DIR/"pdb70_hhm.ffindex")

    pdb9 = File("pdb_filter.dat")
    rc.add_replica(EXECUTE_SITE,pdb9,PDB70_DB_DIR/"pdb_filter.dat")

    mgnify_db = File("mgnify.fa")
    rc.add_replica(EXECUTE_SITE,mgnify_db,MGNIFY_DB_PATH)

    bfd_db = File("bfd.fasta")
    rc.add_replica(EXECUTE_SITE,bfd_db,BFD_DB_PATH)

    rc.write()
    
    # --- Sites ----------------------------------------------------------
    sc = SiteCatalog()

    shared_scratch_dir = os.path.join(WF_DIR, "scratch")
    local_storage_dir = os.path.join(WF_DIR, "outputs")

    local = Site("local")\
        .add_directories(
                        Directory(Directory.SHARED_SCRATCH, shared_scratch_dir, shared_file_system=True)
                            .add_file_servers(FileServer("file://" + shared_scratch_dir, Operation.ALL)),
                        
                        Directory(Directory.LOCAL_STORAGE, local_storage_dir, shared_file_system=True)
                            .add_file_servers(FileServer("file://" + local_storage_dir, Operation.ALL))
                        )

    psc = Site("psc")\
                .add_directories(
                        Directory(Directory.SHARED_SCRATCH, EXECUTE_SITE_DIR, shared_file_system=True)
                            .add_file_servers(FileServer("scp://"+EXECUTE_SITE_USERNAME+"@bridges2.psc.edu/"+EXECUTE_SITE_DIR, Operation.ALL))
                        )\
                .add_pegasus_profile(
                    style="condor",
                    data_configuration="nonsharedfs"
                )\
                .add_env(key="PEGASUS_HOME", value="/usr")\
                .add_profiles(Namespace.PEGASUS, key="SSH_PRIVATE_KEY", value=SSH_KEY_FILE)
                
    condorpool = Site("condorpool")\
                .add_condor_profile(universe="vanilla")\
                .add_pegasus_profile(
                    style="condor",
                    data_configuration="condorio"
                )

    sc.add_sites(local,condorpool,psc)
    sc.write()
    
    # --- Transformations ----------------------------------------------------------

    tc = TransformationCatalog()

    singularity_container = Container(
                  "singularity-container",
                  Container.SINGULARITY,
                  image=CONTAINER_PATH,
                  image_site=EXECUTE_SITE,
                  bypass_staging=True
               )
    tc.add_containers(singularity_container)


    sequence_features = Transformation(
                "sequence_features",
                site="local",
                pfn= WF_DIR/"bin/sequence_features.py",
                is_stageable=True,
                arch=Arch.X86_64,
                os_type=OS.LINUX,
                container=singularity_container
            )

    jackhmmer_uniref90 = Transformation(
                "jackhmmer_uniref90",
                site="local",
                pfn= WF_DIR/"bin/jackhmmer_uniref90.py",
                is_stageable=True,
                arch=Arch.X86_64,
                os_type=OS.LINUX,
                container=singularity_container
            )
    jackhmmer_uniref90.add_profiles(Namespace.CONDOR, key='request_memory', value='8 GB')

    hhsearch_pdb70 = Transformation(
                "hhsearch_pdb70",
                site="local",
                pfn= WF_DIR/"bin/hhsearch_pdb70.py",
                is_stageable=True,
                arch=Arch.X86_64,
                os_type=OS.LINUX,
                container=singularity_container
            )
    hhsearch_pdb70.add_profiles(Namespace.CONDOR, key='request_memory', value='8 GB')

    jackhmmer_mgnify = Transformation(
                "jackhmmer_mgnify",
                site="local",
                pfn= WF_DIR/"bin/jackhmmer_mgnify.py",
                is_stageable=True,
                arch=Arch.X86_64,
                os_type=OS.LINUX,
                container=singularity_container
            )
    jackhmmer_mgnify.add_profiles(Namespace.CONDOR, key='request_memory', value='8 GB')

    hhblits_bfd = Transformation(
                "hhblits_bfd",
                site="local",
                pfn= WF_DIR/"bin/hhblits_bfd.py",
                is_stageable=True,
                arch=Arch.X86_64,
                os_type=OS.LINUX,
                container=singularity_container
            )
    hhblits_bfd.add_profiles(Namespace.CONDOR, key='request_memory', value='8 GB')

    msa_features = Transformation(
                "msa_features",
                site="local",
                pfn= WF_DIR/"bin/msa_features.py",
                is_stageable=True,
                arch=Arch.X86_64,
                os_type=OS.LINUX,
                container=singularity_container
            )

    features_summary = Transformation(
                "features_summary",
                site="local",
                pfn= WF_DIR/"bin/features_summary.py",
                is_stageable=True,
                arch=Arch.X86_64,
                os_type=OS.LINUX
            )

    combine_features = Transformation(
                "combine_features",
                site="local",
                pfn= WF_DIR/"bin/combine_features.py",
                is_stageable=True,
                arch=Arch.X86_64,
                os_type=OS.LINUX,
                container=singularity_container
            )

    tc.add_transformations(sequence_features,
                       jackhmmer_uniref90,
                       hhsearch_pdb70,
                       jackhmmer_mgnify,
                       hhblits_bfd,
                       msa_features,
                       features_summary,
                       combine_features)

    tc.write()
    
    # --- Jobs ----------------------------------------------------------
    wf = Workflow("Alphafold-workflow")

    sequence_features_file = File('sequence_features.pkl')
    job_sequence_features = Job(sequence_features)\
            .add_args(protein_sequence_input,sequence_features_file)\
            .add_inputs(protein_sequence_input)\
            .add_outputs(sequence_features_file)
    wf.add_jobs(job_sequence_features)

    uniref90_msa = File('uniref90_hits.sto')
    uniref90_msa_size = File('uniref90_msa_size.txt')
    job_jackhmmer_msa = Job(jackhmmer_uniref90)\
            .add_args(protein_sequence_input,uniref90_db,uniref90_msa,uniref90_msa_size)\
            .add_inputs(protein_sequence_input,uniref90_db)\
            .add_outputs(uniref90_msa,uniref90_msa_size)
    wf.add_jobs(job_jackhmmer_msa)

    pdb70_hits = File('pdb70_hits.hhr')
    job_pdb70_search = Job(hhsearch_pdb70)\
            .add_args(".",uniref90_msa,pdb70_hits)\
            .add_inputs(pdb1,pdb2,pdb3,pdb4,pdb5,pdb6,pdb7,pdb8,pdb9,uniref90_msa)\
            .add_outputs(pdb70_hits)
    wf.add_jobs(job_pdb70_search)

    mgnify_msa = File('mgnify_hits.sto')
    mgnify_msa_size = File('mgnify_msa_size.txt')
    job_mgnify_msa = Job(jackhmmer_mgnify)\
            .add_args(protein_sequence_input,mgnify_db,mgnify_msa,mgnify_msa_size)\
            .add_inputs(protein_sequence_input,mgnify_db)\
            .add_outputs(mgnify_msa,mgnify_msa_size)
    wf.add_jobs(job_mgnify_msa)

    bfd_msa = File('bfd_hits.sto')
    bfd_msa_size = File('bfd_msa_size.txt')
    job_bfd_msa = Job(hhblits_bfd)\
            .add_args(protein_sequence_input,bfd_db,bfd_msa,bfd_msa_size)\
            .add_inputs(protein_sequence_input,bfd_db)\
            .add_outputs(bfd_msa,bfd_msa_size)
    wf.add_jobs(job_bfd_msa)

    msa_features_file = File('msa_features_file.pkl')
    final_msa_size = File('final_msa_size.txt')
    job_msa_features = Job(msa_features)\
            .add_args(uniref90_msa,mgnify_msa,bfd_msa,msa_features_file,final_msa_size)\
            .add_inputs(uniref90_msa,mgnify_msa,bfd_msa)\
            .add_outputs(msa_features_file,final_msa_size)
    wf.add_jobs(job_msa_features)

    summary_file = File('features_summary.txt')
    job_features_summary = Job(features_summary)\
            .add_args(uniref90_msa_size,mgnify_msa_size,bfd_msa_size,final_msa_size,summary_file)\
            .add_inputs(uniref90_msa_size,mgnify_msa_size,bfd_msa_size,final_msa_size)\
            .add_outputs(summary_file)
    wf.add_jobs(job_features_summary)

    features_file = File('features.pkl')
    job_combine_features = Job(combine_features)\
            .add_args(sequence_features_file,msa_features_file,pdb70_hits,features_file)\
            .add_inputs(sequence_features_file,msa_features_file,pdb70_hits)\
            .add_outputs(features_file)
    wf.add_jobs(job_combine_features)
    
    try:
        wf.write()
        wf.graph(include_files=True, label="xform-id", output="wf_graph.png")
    except PegasusClientError as e:
        print(e)
        
    try:
        wf.plan(submit=True).wait()
    except PegasusClientError as e:
        print(e)
        
    wf.statistics()
    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Generates Pegasus Alphafold workflow")
    
    parser.add_argument('--psc',dest='use_psc_site', help='This option is used when running on ACCESS and PSC Bridges',
                        action='store_true')
    
    parser.add_argument('--input-fasta-file', dest='input_fasta_file', default=None, required=True,
                        help='Path to the input FASTA file containing one protein sequence')
    
    parser.add_argument('--full-dbs',dest='use_large_bfd', help='It runs the workflow with all genetic databases',
                        action='store_true')
    
    parser.add_argument('--uniref90-db-path', dest='uniref90_db_path', default=None, required=True,
                        help='Path to the UniRef90 database')
    
    parser.add_argument('--pdb70-db-dir', dest='pdb70_db_dir', default=None, required=True,
                        help='Path to the PDB70 database directory')
    
    parser.add_argument('--mgnify-db-path', dest='mgnify_db_path', default=None, required=True,
                        help='Path to the MGnify database')
    
    parser.add_argument('--bfd-db-path', dest='bfd_db_path', default=None, required=True,
                        help='Path to the BFD database')
                                                    
    args = parser.parse_args(sys.argv[1:])
    
    generate_wf(input_fasta_path = Path(args.input_fasta_file).resolve(), 
                    uniref90_db_path = Path(args.uniref90_db_path).resolve(),
                    pdb70_db_path = Path(args.pdb70_db_dir).resolve(),
                    mgnify_db_path = Path(args.mgnify_db_path).resolve(),
                    bfd_db_path = Path(args.bfd_db_path).resolve(),
                    use_large_bfd = args.use_large_bfd,
                    use_psc = args.use_psc_site
                   )
                   
    
    
