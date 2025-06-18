#!/bin/bash
# Set your server IP here
SERVER_IP="10.176.228.194"

# Direct capture without preview
python surgical_env_analyze.py --camera 2 \
--server http://${SERVER_IP}:8000 \
--system-file system_message_analyze.txt \
--message "$(cat user_message_analyze.txt)" \
--max-tokens 1024 \
--temperature 0.0