# Set your server IP here
SERVER_IP="0.0.0.0"

python surgical_env_analyze.py debug_images/demo_verify.png \
--server http://${SERVER_IP}:8000 \
--system-file system_message_verification.txt \
--message "$(cat user_message_verification.txt)" \
--phase verification \
--max-tokens 512 \
--temperature 0.0