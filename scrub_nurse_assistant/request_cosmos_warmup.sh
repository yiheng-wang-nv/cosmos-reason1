#!/bin/bash
# Set your server IP here
SERVER_IP="10.176.228.194"

# Reference image showing surgical environment (you need to provide this)
REFERENCE_IMAGE="images/surgical_tools_example.png"

echo "Starting LLM warm-up for surgical environment..."

python env_warmup.py ${REFERENCE_IMAGE} \
--server http://${SERVER_IP}:8000 \
--max-tokens 1024 \
--temperature 0.0

echo "Warm-up completed. Ready for analysis!"
