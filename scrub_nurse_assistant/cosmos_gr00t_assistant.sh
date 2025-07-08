#!/bin/bash
# Set your server IP here
# SERVER_IP="10.176.229.8"

# # # camera direct capture
# python camera_image_analyze.py --camera 0 \
# --server http://${SERVER_IP}:8000 \
# --system-file system_message.txt \
# --message "$(cat user_message.txt)" \
# --max-tokens 1024 \
# --temperature 0.0

# local image
# python local_image_analyze.py images/demo_all_tools.png \
# --server http://${SERVER_IP}:8000 \
# --system-file system_message.txt \
# --message "$(cat user_message.txt)" \
# --max-tokens 1024 \
# --temperature 0.0

# # robot image
# python robot_image_analyze.py --robot \
# --server http://${SERVER_IP}:8000 \
# --system-file system_message.txt \
# --message "$(cat user_message.txt)" \
# --max-tokens 1024 \
# --temperature 0.0


# Full example with all parameters
python robot_image_analyze.py --robot \
    --server http://10.176.229.8:8000 \
    --groot_host localhost \
    --groot_port 5555 \
    --actions_per_task 30 \
    --port_follower /dev/ttyACM2

