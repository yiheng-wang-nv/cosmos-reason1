#!/bin/bash

# Simple script to run surgical analysis on single image
# Usage: ./analyze.sh [server_ip]
# 
# To use:
# 1. Modify the IMAGE_PATH below with your image path
# 2. Optionally modify SYSTEM_FILE and USER_FILE paths
# 3. Run: ./analyze.sh [server_ip]

# =============================================================================
# CONFIGURATION - MODIFY THESE PATHS FOR YOUR IMAGE
# =============================================================================

# Default message files (modify if needed)
SYSTEM_FILE="system_message.txt"
USER_FILE="user_message.txt"

# =============================================================================
# SCRIPT LOGIC - NO NEED TO MODIFY BELOW
# =============================================================================

# Get server IP (default to localhost:8000)
SERVER_IP=${1:-"localhost:8000"}

# for image in example_1 to example_4, run the analysis
for image in demo_images/example_*.png; do
    echo "Running analysis for $image"
    python client.py "$image" --server "http://$SERVER_IP" --system-file "$SYSTEM_FILE" --user-file "$USER_FILE"
done
