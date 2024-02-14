#!/bin/bash

set -e 

file=$1

shift

tar -xf "$file.tar.gz"

rm "$file.tar.gz"

cp -r "$file"/* .

rm -r "$file"

chmod +x ./AbinitioRelax.static.linuxgccrelease

set +e

if [ ! -d database ]; then
  tar -xzf database.tar.gz
  rm database.tar.gz
fi

set -e

./AbinitioRelax.static.linuxgccrelease "$@"

