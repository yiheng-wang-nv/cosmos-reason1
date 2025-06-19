#!/bin/bash
# Set your server IP here
SERVER_IP="10.176.228.194"

# # camera direct capture
# python camera_image_analyze.py --camera 2 \
# --server http://${SERVER_IP}:8000 \
# --system-file system_message.txt \
# --message "$(cat user_message.txt)" \
# --max-tokens 1024 \
# --temperature 0.0

# # local image
# python local_image_analyze.py images/demo_input.png \
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
    --server http://10.176.228.194:8000 \
    --groot_host localhost \
    --groot_port 5555 \
    --actions_per_task 60 \
    --port_follower /dev/ttyACM1

# record screen: gnome-screenshot --area --delay=3