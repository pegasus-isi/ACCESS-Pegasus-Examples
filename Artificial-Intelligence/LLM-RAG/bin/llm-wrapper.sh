#!/bin/bash

BOOK=$1

# create empty outputs so that we do not get
# held jobs if the ollama setup fails
touch $BOOK-ollama.log
touch $BOOK-answers.txt

# work around HTCondor setting HOME=/
export HOME=$_CONDOR_JOB_IWD

# pick a unique port for this instance
export OLLAMA_PORT=$((($RANDOM % 30000) + 10000))
export OLLAMA_HOST=127.0.0.1:$OLLAMA_PORT

ollama serve >$BOOK-ollama.log 2>&1 &
OLLAMA_PID=$!

# block using the ready endpoint
time curl --retry 10 --retry-connrefused --retry-delay 5 -sf http://$OLLAMA_HOST 2>&1

chmod +x llm-rag.py
./llm-rag.py $BOOK >$BOOK-answers.txt 
EC=$?

kill $OLLAMA_PID
wait $OLLAMA_PID

exit $EC
