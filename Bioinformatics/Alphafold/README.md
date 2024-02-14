# Alphafold Pegasus Workflow

A Pegasus Workflow for running [Alphafold](https://github.com/deepmind/alphafold) model's inference pipeline regarding protein structure
prediction. The current workflow is regarding the Multiple Sequence Alignment (MSA) and 
Feature Generation steps, which produce a `features.pkl` file that can be later used in protein structure inference
stage using the Alphafold model parameters. The workflow is currently limited to the Alphafold `monomer-system` model preset by default.
The workflow is set to run in `sharedfs` mode with no input staging and symlinking turned on.  

## Steps for setup on ACCESS resources
If you are planning to run the workflow on ACCESS resources with [PSC Bridges](https://ondemand.bridges2.psc.edu.) as the resource allocation provider, please follow the steps shown below :
* To get started, point your browser to https://access.pegasus.isi.edu and log in using the ACCESS Pegasus credentials.
  Use the [ACCESS Pegasus Documentation](https://access-ci.atlassian.net/wiki/spaces/ACCESSdocumentation/pages/129140407/ACCESS+Pegasus)
  to configure a basic setup for [ACCESS Pegasus workflows](https://github.com/pegasus-isi/ACCESS-Pegasus-Examples). It's recommended that you try
  to  execute the sample workflows first listed in the documentation in order to avoid any errors, they are simple and easy to execute.
* After the setup is complete and you are able to run the sample ACCESS-Pegasus workflows successfully, next we need to configure a new SSH key to be
  used for file transfers(by `scp` protocol) in our Alphafold workflow. Go to homepage on https://access.pegasus.isi.edu, then open up a 
  shell navigating `Clusters --> Shell Access`. Generate a new SSH key as follows:
  ```
  $ cd .ssh
  $ ssh-keygen -t rsa
  ```
  Make note of absolute path to the private SSH key file `id_rsa`, as this path will be used our Alphafold workflow.
  Since PSC Bridges doesn't allow to configure new SSH keys simply by saving the public key in a file using the shell, the public key
  has to be submitted using PSC Bridges Key Management system. Copy the contents of file `id_rsa.pub` and login to [PSC SSH Key Manager](https://grants.psc.edu/cgi-bin/ssh/listKeys.pl) using the PSC Bridges login credentials. Click on `Submit New Key` and paste the public key copied
  from `id_rsa.pub` file in the `Paste Key` section and then submit it. For more info on PSC SSH Key Management, you can refer to their website : 
  https://www.psc.edu/about-using-ssh/. It takes a couple of hours for the SSH key to be configured on their system.
* The next step is to setup a container to be used in the workflow. Open up shell on PSC Bridges by logging into  
  [PSC Bridges](https://ondemand.bridges2.psc.edu.), then navigating `Clusters --> Shell Access`. Go to your username directory in the RM storage 
  on PSC and clone the Alphafold repository there :
  ```
  $ cd /ocean/projects/<groupname>/<username>/
  $ git clone https://github.com/pegasus-isi/alphafold-pegasus.git
  $ cd alphafold-pegasus
  ```
  Then create a singularity container as follows: 
  ```
  $ docker build -t local/alphafold_container .
  $ singularity build alphafold_container.sif docker-daemon://local/alphafold_container
  ```
  Make note of the absolute path to the conatiner as it will be used in the Alphafold workflow later. More information and notes regarding the 
  container step can be found in the Container section below.
* The next step is to setup a data directory and download all the genetic databases there as follows:
  ```
  $ cd /ocean/projects/<groupname>/<username>/
  $ mkdir data
  $ /ocean/projects/<groupname>/<username>/alphafold-pegasus/data/download_all_data.sh -d /ocean/projects/<groupname>/<username>/data
  ```
  This is a time consuming step and takes somewhere between 6 hours to 8 hours, so please make sure that the session doesn't time out.
  More information and notes regarding data download step can be found in the Genetic Databases section below. 
* Finally we submit and run the workflow from ACCESS-pegasus. Login using your credentials on https://access.pegasus.isi.edu, then open up a 
  shell navigating `Clusters --> Shell Access`, clone this repository and run the `alphafold_workflow.py` script as follows :
  ```
  $ git clone https://github.com/pegasus-isi/alphafold-pegasus.git
  $ cd alphafold-pegasus
  $ python3 alphafold_workflow.py \
    --psc \
    --input-fasta-file=/ocean/projects/<groupname>/<username>/alphafold-pegasus/input/GA98.fasta \
    --uniref90-db-path=/ocean/projects/<groupname>/<username>/data/uniref90 \
    --pdb70-db-dir=/ocean/projects/<groupname>/<username>/data/pdb70 \
    --mgnify-db-path=/ocean/projects/<groupname>/<username>/data/mgnify \
    --bfd-db-path=/ocean/projects/<groupname>/<username>/data/bfd
  ```
  The `--psc` option is used because the compute site for this steup is PSC Bridges, this option is not required when running on a local machine.
  Please make sure that paths to the genetic databases are entered correctly. Some workflow statistics have been shown below for reference.


## Container

The workflow uses a singularity container in order to execute all jobs. It is recommended to build a local container (in a `.sif` file) using the
[Alphafold's](https://github.com/deepmind/alphafold/blob/main/docker/Dockerfile) provided `Dockerfile` which has all the required libraries and tools. It can be done in the following steps :
```
$ git clone https://github.com/deepmind/alphafold.git
$ cd alphafold
$ docker build -t local/alphafold_container .
$ singularity build alphafold_container.sif docker-daemon://local/alphafold_container
```
The container comes with the following main tools along with other common libraries :
* alphafold==2.2.0
* hmmer==3.3.2 
* hhsuite==3.3.0 
* kalign2==2.04
* absl-py==0.13.0 
* biopython==1.79 
* chex==0.0.7 
* dm-haiku==0.0.4 
* dm-tree==0.1.6 
* immutabledict==2.0.0 

:ledger: **Note:** If you are running the workflow on ACCESS, it's recommended to build the container on execute site for reduced execution time.
For example, if your execute site is `PSC Bridges2`, you can `$ cd` into your project directory and build the container following
the above steps. Then set complete path to the container in `alphafold_workflow.py` or `alphafold_workflow_main.ipynb` files.

## Genetic databases
If your machine has `aria2c` installed in it, then it's recommended to use Alphafold's provided database download scripts over 
[here](https://github.com/deepmind/alphafold/tree/main/scripts).
Otherwise the database download scripts provided in this repository (`/data/download_all_data.sh`) use readily available command line utilities.
The following databases are used in the workflow :
*   [BFD](https://bfd.mmseqs.com/)
*   [MGnify](https://www.ebi.ac.uk/metagenomics/)
*   [PDB70](http://wwwuser.gwdg.de/~compbiol/data/hhsuite/databases/hhsuite_dbs/)
*   [UniRef90](https://www.uniprot.org/help/uniref)

```
$ /data/download_all_data.sh -d <DOWNLOAD_DIRECTORY>
```

:ledger: **Note:** If you are running the workflow on ACCESS, please download the data on execute site in order to avoid staging of data.
For example, if your execute site is `PSC Bridges2`, you can `$ cd` into your project directory and download the databases there following
the above steps.

:ledger: **Note:** By default the `download_all_data.sh` script is set to download the reduced version of databases (of size 600 GB). 
If you want to download the full version of databases (of size 2.2 TB), `full_dbs` option can be entered as follows :

```
$ /data/download_all_data.sh -d <DOWNLOAD_DIRECTORY> full_dbs
```

:ledger: **Note: The download directory `<DOWNLOAD_DIR>` should _not_ be a
subdirectory in the AlphaFold repository directory.** If it is, the Docker build
will be slow as the large databases will be copied during the image creation.

## Workflow

![](/images/wf_graph.png)

The jobs and tools used in the workflow are explained below:

*   `sequence_features` – produces the sequence features from the input fasta file
*   `jackhmmer_uniref90` – runs jackhmmer tool on the UniRef90 database to produce MSAs
*   `jackhmmer_mgnify` – runs jackhmmer tool on the MGnify database to produce MSAs
*   `hhblits_bfd` – runs hhblits tool on the BFD database to produce MSAs
*   `hhsearch_pdb70` - runs hhsearch tool on PDB70 database to produce search templates
*   `msa_features` – turns the MSA results into dicts of features
*   `features_summary` – contains a summary of info reagarding all MSAs produced
*   `combine_features` – combines all MSA features, sequence features and templates into features file `features.pkl`


## Running the workflow

The workflow is set to run on a local HTCondor Pool in the default Condorio
data configuration mode, where each job is run in a Singularity container.
To submit a workflow run :
```
    python3 alphafold_workflow.py \
    --input-fasta-file=/path/to/input/fasta/file \
    --uniref90-db-path=/path/to/uniref90_db \
    --pdb70-db-dir=/path/to/pdb70_db \
    --mgnify-db-path=/path/to/mgnify_db \
    --bfd-db-path=/path/to/bfd_db 
```

:ledger: **Note:** It's recommended to first test run the workflow on very small partial databases `UniRef90`,`Mgnify` and `BFD`, some samples
are already included in `/data/small_data` directory with 500 sequences in each file. Small partial databases can be created using the `generate_small_data.py` script in `/data/small_data` as follows: 

```
$ /data/generate_small_data.py <FASTA_FILE> <NO_OF_SEQUENCES> <NEW_FILE_NAME>
```
For example:
```
$ /data/generate_small_data.py uniref90.fasts 5000 small_uniref90.fasta
```

:ledger: **Note:** Workflow statistics have been shown in the `alphafold_workflow_main.ipynb` notebook, this sample workflow run used `GA98`
as input sequence rather than `T1050` sequence used originally in the CASP14 by Alphafold. Thus, workflow execution time may vary depending upon
the input sequence used.

## Workflow statistics

The following table shows workflow wall time corresponding to different setup and size of databases:
| Setup | Partial Database (~70GB) | Complete Database (~600GB) |
| :---         |     :---:      |          :---: |
| Local machine   | 18 min, 23 secs     | --    |
| PSC Bridges 2     | 4 min, 51 secs      | 1 hr, 7 mins      |

:ledger: **Note:** In both cases workflow is set to run in `sharedfs` configuration with no input staging and symlinking is turned on.
