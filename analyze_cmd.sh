# Set your server IP here
SERVER_IP="0.0.0.0"

python surgical_env_analyze.py debug_images/demo_input.png \
--server http://${SERVER_IP}:8000 \
--system-file system_message_analyze.txt \
--message "$(cat user_message_analyze.txt)" \
--max-tokens 1024 \
--temperature 0.0