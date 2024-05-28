#!/bin/bash

# pick a unique port for this instance
export OLLAMA_PORT=$((($RANDOM % 30000) + 10000))
export OLLAMA_HOST=127.0.0.1:$OLLAMA_PORT

ollama serve >ollama.log 2>&1 &
OLLAMA_PID=$!
sleep 15s

chmod +x llm-rag.py
./llm-rag.py >answers.txt
EC=$?

kill $OLLAMA_PID

exit $EC

