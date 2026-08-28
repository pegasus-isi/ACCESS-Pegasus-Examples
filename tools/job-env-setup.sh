#!/bin/bash

# this file is use sources when the job starts up and
# used to provide a common environment for jobs across
# ACCESS resources

export PATH=/usr/bin:/bin
export USER=`whoami`

hostname=`hostname -f`
stampede_pattern='^[A-Za-z0-9-]+.stampede2.tacc.utexas.edu$'
expanse_pattern='^[A-Za-z0-9-]+.expanse.sdsc.edu$'
laguna_pattern='^[A-Za-z0-9-]+.carc.usc.edu$' 
if [[ $hostname =~ $stampede_pattern ]]; then
    echo "Sourcing job environment for Stampede2"
    # initialize modules
    . /etc/profile.d/z01_lmod.sh
    module load tacc-singularity
elif [[ $hostname =~ $expanse_pattern ]]; then
    echo "Sourcing job environment for Expanse"
    source /etc/profile.d/modules.sh
    module load singularitypro
elif [[ $hostname =~ $laguna_pattern ]]; then
    echo "Sourcing job environment for USC Laguna"
    source /etc/profile.d/modules.sh
    module load apptainer
else
    echo "No specific job environment sourced"
fi


