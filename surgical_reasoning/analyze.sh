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

# Image path - MODIFY THIS FOR YOUR IMAGE
IMAGE_PATH="demo_images/example_1.png"

# =============================================================================
# SCRIPT LOGIC - NO NEED TO MODIFY BELOW
# =============================================================================

# Get server IP (default to localhost:8000)
SERVER_IP=${1:-"localhost:8000"}

echo "🏥 Running surgical analysis..."
echo "Server: $SERVER_IP"
echo "System message: $SYSTEM_FILE"
echo "User message: $USER_FILE"
echo "Image: $IMAGE_PATH"

# Check if message files exist
if [ ! -f "$SYSTEM_FILE" ]; then
    echo "❌ System message file not found: $SYSTEM_FILE"
    exit 1
fi

if [ ! -f "$USER_FILE" ]; then
    echo "❌ User message file not found: $USER_FILE"
    exit 1
fi

# Check if image exists
if [ ! -f "$IMAGE_PATH" ]; then
    echo "❌ Image not found: $IMAGE_PATH"
    echo "Please modify the IMAGE_PATH variable in this script"
    exit 1
fi

# Run the analysis
python client.py "$IMAGE_PATH" --server "http://$SERVER_IP" --system-file "$SYSTEM_FILE" --user-file "$USER_FILE"
