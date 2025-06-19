#!/bin/bash
# Set your server IP here
SERVER_IP="10.176.228.194"

# Direct capture without preview
python env_analyze.py --camera 2 \
--server http://${SERVER_IP}:8000 \
--system-file system_message.txt \
--message "$(cat user_message.txt)" \
--max-tokens 1024 \
--temperature 1.0
