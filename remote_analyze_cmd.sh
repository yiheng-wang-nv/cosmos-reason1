# Set your server IP here
SERVER_IP="10.176.228.194"

python surgical_env_analyze.py /home/venn/Desktop/code/demo_input.png \
--server http://${SERVER_IP}:8000 \
--phase initial \
--system-file system_message_analyze.txt \
--message "$(cat user_message_analyze.txt)" \
--max-tokens 512 \
--temperature 0.0 \
--use-base64
