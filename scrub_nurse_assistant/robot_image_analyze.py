#!/usr/bin/env python3
"""
Surgical Environment Analysis Client

This client sends surgical environment images to the analysis server and 
displays the structured response with robot commands for tool organization.
"""

import argparse
import requests
import sys
import re
import base64
import cv2
import io
from PIL import Image
from typing import Optional, List
from datetime import datetime
import os
import time
import torch
import numpy as np
from tqdm import tqdm

# Import the SO101Robot class
from so101_robot_utils import SO101Robot, Gr00tSO101InferenceClient

# Terminal colors and formatting
class Colors:
    """ANSI color codes for terminal output"""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    UNDERLINE = '\033[4m'
    
    # Colors
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    
    # Background colors
    BG_RED = '\033[101m'
    BG_GREEN = '\033[102m'
    BG_YELLOW = '\033[103m'
    BG_BLUE = '\033[104m'

# Icons and symbols
class Icons:
    """Unicode icons for better visual representation"""
    ROBOT = "🤖"
    MEDICAL = "🏥"
    SCISSORS = "✂️"
    TWEEZERS = "🔧"
    SUCCESS = "✅"
    WARNING = "⚠️"
    ERROR = "❌"
    INFO = "ℹ️"
    ANALYSIS = "🔍"
    THINKING = "💭"
    PLAN = "📋"
    ARROW = "➤"
    BULLET = "•"
    GEAR = "⚙️"
    CONFIG = "🎛️"

def colorize(text: str, color: str, bold: bool = False) -> str:
    """Apply color and formatting to text."""
    style = Colors.BOLD if bold else ""
    return f"{style}{color}{text}{Colors.RESET}"

def print_header(title: str, icon: str = "", color: str = Colors.CYAN):
    """Print a formatted header."""
    header_line = "=" * 80
    title_line = f"{icon} {title}" if icon else title
    print(f"\n{colorize(header_line, color)}")
    print(f"{colorize(title_line.center(80), color, bold=True)}")
    print(f"{colorize(header_line, color)}")

def print_section(title: str, content: str, icon: str = "", color: str = Colors.WHITE):
    """Print a formatted section with title and content."""
    title_line = f"{icon} {title}" if icon else title
    print(f"{colorize(title_line, color, bold=True)}")
    print(f"{colorize('-' * 60, color)}")
    print(content)

def print_config_info(max_tokens: int, temperature: float):
    """Print configuration information."""
    print(f"{colorize(Icons.CONFIG, Colors.CYAN)} {colorize('Config:', Colors.WHITE, bold=True)} "
          f"{colorize('max_tokens=', Colors.CYAN)}{colorize(str(max_tokens), Colors.WHITE)} "
          f"{colorize('temperature=', Colors.CYAN)}{colorize(str(temperature), Colors.WHITE)}")

def extract_structured_response(response_text: str) -> dict:
    """
    Extract reasoning and plan sections from the response.
    
    Args:
        response_text (str): Raw response text from the server
        
    Returns:
        dict: Dictionary with 'reasoning' and 'plan' keys
    """
    result = {"reasoning": "", "plan": ""}
    
    # Extract reasoning section
    reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', response_text, re.DOTALL | re.IGNORECASE)
    if reasoning_match:
        result["reasoning"] = reasoning_match.group(1).strip()
    
    # Extract plan section
    plan_match = re.search(r'<plan>(.*?)</plan>', response_text, re.DOTALL | re.IGNORECASE)
    if plan_match:
        result["plan"] = plan_match.group(1).strip()
    
    # Fallback: if no XML tags found, use the entire response as reasoning
    if not result["reasoning"] and not result["plan"]:
        result["reasoning"] = response_text.strip()
    
    return result

def extract_plan_commands(plan_text: str) -> List[str]:
    """
    Extract individual robot commands from the plan text.
    
    Args:
        plan_text (str): Plan section text
        
    Returns:
        List[str]: List of robot commands
    """
    commands = []
    
    # Split by lines and filter out empty lines
    lines = [line.strip() for line in plan_text.split('\n') if line.strip()]
    
    for line in lines:
        # Look for actionable robot commands
        line_lower = line.lower()
        if any(keyword in line_lower for keyword in [
            "grip", "grasp", "pick", "place", "move", "put", 
            "scissor", "tweezer", "box", "tray"
        ]):
            commands.append(line)
    
    return commands

def parse_robot_commands(response_text: str) -> List[str]:
    """
    Parse robot commands from LLM response.
    
    Args:
        response_text (str): Raw response text from LLM
        
    Returns:
        List[str]: List of actionable robot commands
    """
    sections = extract_structured_response(response_text)
    commands = []
    
    # Extract commands from plan section
    if sections["plan"]:
        plan_commands = extract_plan_commands(sections["plan"])
        commands.extend(plan_commands)
    
    # If no structured plan, try to extract from reasoning
    if not commands and sections["reasoning"]:
        reasoning_commands = extract_plan_commands(sections["reasoning"])
        commands.extend(reasoning_commands)
    
    # Convert to specific task descriptions for GR00T
    robot_tasks = []
    for cmd in commands:
        cmd_lower = cmd.lower()
        if "scissor" in cmd_lower and ("grip" in cmd_lower or "pick" in cmd_lower):
            robot_tasks.append("Grip a straight scissor and put it in the box.")
        elif "tweezer" in cmd_lower and ("grip" in cmd_lower or "pick" in cmd_lower):
            robot_tasks.append("Grip a tweezer and put it in the box.")
        elif "grip" in cmd_lower or "pick" in cmd_lower:
            # Generic pick and place task
            robot_tasks.append("Pick up the object and place it in the box.")
    
    return robot_tasks

def print_commands(commands: List[str]):
    """Print robot commands with appropriate icons."""
    if not commands:
        print(f"{colorize('No robot commands found in response.', Colors.YELLOW)}")
        return
    
    for i, command in enumerate(commands, 1):
        # Determine icon based on command content (simplified)
        if "straight scissor" in command.lower():
            icon = f"{Icons.SCISSORS}➡️"
        elif "tweezer" in command.lower():
            icon = Icons.TWEEZERS
        else:
            icon = Icons.ARROW
        
        print(f"{colorize(f'{i}.', Colors.CYAN)} {colorize(icon, Colors.YELLOW)} {colorize(command, Colors.WHITE)}")

def print_structured_response(response_text: str):
    """
    Parse and print the structured response from the server with enhanced reasoning display.
    
    Args:
        response_text (str): Raw response text from the server
    """
    # Extract structured sections
    sections = extract_structured_response(response_text)
    
    # Print reasoning section with enhanced formatting
    if sections["reasoning"]:
        print(f"\n{colorize('🔍 ANALYSIS & REASONING:', Colors.CYAN, bold=True)}")
        print(f"{colorize('─' * 70, Colors.CYAN)}")
        
        # Format reasoning text with proper indentation
        reasoning_lines = sections["reasoning"].split('\n')
        for line in reasoning_lines:
            if line.strip():
                print(f"  {colorize(line.strip(), Colors.WHITE)}")
            else:
                print()  # Preserve empty lines for readability
    
    # Print plan section with command extraction
    if sections["plan"]:
        print(f"\n{colorize('📋 EXECUTION PLAN:', Colors.CYAN, bold=True)}")
        print(f"{colorize('─' * 70, Colors.CYAN)}")
        
        # Format plan text
        plan_lines = sections["plan"].split('\n')
        for line in plan_lines:
            if line.strip():
                print(f"  {colorize(line.strip(), Colors.WHITE)}")
            else:
                print()
        
        # Extract and display commands separately
        commands = extract_plan_commands(sections["plan"])
        if commands:
            print(f"\n{colorize('🤖 EXTRACTED ROBOT COMMANDS:', Colors.MAGENTA, bold=True)}")
            print(f"{colorize('─' * 70, Colors.MAGENTA)}")
            print_commands(commands)
    
    # If no structured sections found, print raw response
    if not sections["reasoning"] and not sections["plan"]:
        print(f"\n{colorize('📄 RAW RESPONSE:', Colors.YELLOW, bold=True)}")
        print(f"{colorize('─' * 70, Colors.YELLOW)}")
        print(f"  {colorize(response_text.strip(), Colors.WHITE)}")

def print_analysis_summary(result: dict):
    """Print simplified analysis summary from server response."""
    if "analysis" not in result:
        return
    
    analysis = result["analysis"]
    
    # Only show a brief summary, no duplicate commands
    tool_count = analysis.get("surgical_tools_detected", 0)
    scissors_found = analysis.get("scissors_detected", False)
    tweezers_found = analysis.get("tweezers_detected", False)
    
    if tool_count > 0:
        print(f"\n{colorize('📋 DETECTION SUMMARY:', Colors.GREEN, bold=True)} {colorize(f'{tool_count} surgical tools detected', Colors.WHITE)}")
        
        # Brief tool breakdown without commands
        tool_details = []
        if scissors_found:
            tool_details.append(f"{colorize(Icons.SCISSORS, Colors.YELLOW)} Straight surgical scissors")
        if tweezers_found:
            tool_details.append(f"{colorize(Icons.TWEEZERS, Colors.YELLOW)} Surgical tweezers")
        
        if tool_details:
            print(f"  {colorize(' + ', Colors.GREEN).join(tool_details)}")
    else:
        print(f"\n{colorize('📋 DETECTION SUMMARY:', Colors.YELLOW, bold=True)} {colorize('No surgical tools detected', Colors.WHITE)}")
        print(f"  {colorize('→ Robot scrub nurse will remain idle', Colors.YELLOW)}")

def execute_robot_commands(robot: SO101Robot, commands: List[str], 
                          host: str = "localhost", port: int = 5555,
                          actions_per_task: int = 50, window_width: int = 320, 
                          window_height: int = 240) -> bool:
    """
    Execute robot commands using GR00T policy.
    
    Args:
        robot (SO101Robot): Connected robot instancewindow_
        commands (List[str]): List of task descriptions to execute
        host (str): GR00T server host
        port (int): GR00T server port
        actions_per_task (int): Number of actions to execute per task
        window_width (int): Camera window width
        window_height (int): Camera window height
        
    Returns:
        bool: True if execution completed successfully
    """
    if not commands:
        print(f"{colorize('No executable robot commands found.', Colors.YELLOW)}")
        return True
    
    print(f"\n{colorize(Icons.ROBOT, Colors.BLUE)} {colorize('EXECUTING ROBOT COMMANDS', Colors.BLUE, bold=True)}")
    print(f"{colorize('─' * 60, Colors.BLUE)}")
    print(f"{colorize('Press', Colors.CYAN)} {colorize('q', Colors.RED, bold=True)} {colorize('to skip to next task (early completion)', Colors.CYAN)}")
    
    try:
        # Initialize GR00T client
        client = Gr00tSO101InferenceClient(
            host=host,
            port=port,
            language_instruction=commands[0]  # Start with first command
        )
        
        for i, task_description in enumerate(commands, 1):
            print(f"\n{colorize(f'Task {i}/{len(commands)}:', Colors.CYAN, bold=True)} {colorize(task_description, Colors.WHITE)}")
            
            # Update task instruction
            client.set_language_instruction(task_description)
            
            # Execute task
            task_result = execute_single_task(robot, client, actions_per_task, window_width, window_height)
            
            if task_result == "skip":
                print(f"{colorize(f'✓ Task {i} completed early (skipped by user)', Colors.GREEN, bold=True)}")
                if i < len(commands):
                    print(f"{colorize(f'→ Moving to Task {i+1}...', Colors.CYAN)}")
                    time.sleep(1)  # Brief pause
                continue
            elif task_result == "stop":
                print(f"{colorize('Task execution stopped by user.', Colors.YELLOW)}")
                return False
            else:
                print(f"{colorize(f'✓ Task {i} completed successfully', Colors.GREEN, bold=True)}")
            
            # Brief pause between tasks
            if i < len(commands):
                print(f"{colorize('Preparing for next task...', Colors.CYAN)}")
                time.sleep(2)
        
        # All tasks completed - show completion options
        print(f"\n{colorize('🎉 ALL TASKS COMPLETED! 🎉', Colors.GREEN, bold=True)}")
        print(f"{colorize('═' * 50, Colors.GREEN)}")
        print(f"{colorize('Tray cleaning tasks finished successfully!', Colors.GREEN)}")
        print(f"\n{colorize('What would you like to do next?', Colors.WHITE, bold=True)}")
        print(f"  {colorize('n', Colors.RED, bold=True)} - Start new round (capture → analyze → execute)")
        print(f"  {colorize('x', Colors.RED, bold=True)} - Exit program")
        
        # Wait for user choice
        while True:
            live_img, live_img_room = robot.get_current_img()
            
            # Resize images to match window size
            live_img_resized = cv2.resize(live_img, (window_width, window_height))
            live_img_room_resized = cv2.resize(live_img_room, (window_width, window_height))
            
            img_wrist_bgr = cv2.cvtColor(live_img_resized, cv2.COLOR_RGB2BGR)
            img_room_bgr = cv2.cvtColor(live_img_room_resized, cv2.COLOR_RGB2BGR)
            
            # Display clean camera feeds
            cv2.imshow("Wrist Camera", img_wrist_bgr)
            cv2.imshow("Room Camera", img_room_bgr)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('n'):
                print(f"\n{colorize('🔄 Starting new round...', Colors.CYAN, bold=True)}")
                return "new_round"
            elif key == ord('x'):
                print(f"\n{colorize('👋 Exiting program...', Colors.CYAN)}")
                return "exit"
        
    except Exception as e:
        print(f"{colorize(Icons.ERROR, Colors.RED)} {colorize('Robot execution failed:', Colors.RED, bold=True)} {e}")
        return False

def execute_single_task(robot: SO101Robot, client: Gr00tSO101InferenceClient, 
                       actions_to_execute: int, window_width: int = 320, 
                       window_height: int = 240) -> str:
    """
    Execute a single robot task.
    
    Args:
        robot (SO101Robot): Connected robot instance
        client (Gr00tSO101InferenceClient): GR00T inference client
        actions_to_execute (int): Number of actions to execute
        window_width (int): Camera window width
        window_height (int): Camera window height
        
    Returns:
        str: "completed", "skip", or "stop"
    """
    MODALITY_KEYS = ["single_arm", "gripper"]
    action_horizon = 12  # Default action horizon
    
    print(f"  {colorize('Executing', Colors.CYAN)} {colorize(str(actions_to_execute), Colors.WHITE)} {colorize('action steps...', Colors.CYAN)}")
    print(f"  {colorize('Press', Colors.CYAN)} {colorize('q', Colors.RED, bold=True)} {colorize('to complete this task early and move to next', Colors.CYAN)}")
    
    for i in tqdm(range(actions_to_execute), desc="  Progress", leave=False):
        try:
            # Get current observation
            img, img_room = robot.get_current_img()
            state = robot.get_current_state()
            
            # Get action from policy
            action = client.get_action(img, img_room, state)
            
            # Execute action chunk
            for j in range(action_horizon):
                # Concatenate action components
                concat_action = np.concatenate([
                    np.atleast_1d(action[f"action.{key}"][j]) 
                    for key in MODALITY_KEYS
                ], axis=0)
                
                # Send to robot
                robot.set_target_state(torch.from_numpy(concat_action))
                time.sleep(0.1)  # Small delay between actions
                
                # Update existing camera windows (clean feeds, no overlays)
                live_img, live_img_room = robot.get_current_img()
                
                # Resize images to match window size
                live_img_resized = cv2.resize(live_img, (window_width, window_height))
                live_img_room_resized = cv2.resize(live_img_room, (window_width, window_height))
                
                img_wrist_bgr = cv2.cvtColor(live_img_resized, cv2.COLOR_RGB2BGR)
                img_room_bgr = cv2.cvtColor(live_img_room_resized, cv2.COLOR_RGB2BGR)
                
                # Display clean camera feeds without any overlays
                cv2.imshow("Wrist Camera", img_wrist_bgr)
                cv2.imshow("Room Camera", img_room_bgr)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print(f"\n{colorize('Task marked as completed (early finish)', Colors.YELLOW)}")
                    return "skip"
                elif key == ord('x'):  # Emergency stop
                    print(f"\n{colorize('Emergency stop requested', Colors.RED)}")
                    return "stop"
                    
        except Exception as e:
            print(f"\n{colorize(Icons.ERROR, Colors.RED)} Error during execution: {e}")
            return "stop"
    
    return "completed"

def send_request(image_path: Optional[str] = None,
                user_message: Optional[str] = None, system_message: Optional[str] = None, 
                server_url: str = "http://localhost:8000", max_tokens: int = 1024, 
                temperature: float = 0.0, use_base64: bool = False):
    """Send analysis request to the server."""
    
    # Print configuration
    print_config_info(max_tokens, temperature)
    
    # Prepare request data
    request_data = {
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    # Handle image source
    if image_path:
        # Handle file path
        if use_base64:
            # Read and encode image as base64
            try:
                with open(image_path, "rb") as image_file:
                    image_data = image_file.read()
                    base64_encoded = base64.b64encode(image_data).decode('utf-8')
                    request_data["image_base64"] = base64_encoded
                    print(f"{colorize(Icons.INFO, Colors.CYAN)} Image encoded as base64: {len(base64_encoded)} characters")
            except Exception as e:
                print(f"{colorize(Icons.ERROR, Colors.RED)} Failed to encode image: {e}")
                return
        else:
            # Use file path for local server
            request_data["image_path"] = image_path
    else:
        print(f"{colorize(Icons.ERROR, Colors.RED)} image_path must be provided")
        return
    
    # Use custom messages if provided
    if user_message:
        request_data["user_message"] = user_message
    if system_message:
        request_data["system_message"] = system_message
    
    print(f"\n{colorize(Icons.GEAR, Colors.CYAN)} {colorize('Sending surgical environment analysis request...', Colors.CYAN)}")
    
    try:
        # Send POST request
        response = requests.post(
            f"{server_url}/generate",
            json=request_data,
            headers={"Content-Type": "application/json"}
        )
        
        # Check response status
        if response.status_code == 200:
            result = response.json()
            
            # Print main header
            print_header("SURGICAL ENVIRONMENT ANALYSIS RESULT", Icons.MEDICAL, Colors.GREEN)
            print(f"\n{colorize(Icons.INFO, Colors.CYAN)} {colorize('Analyzed image:', Colors.WHITE, bold=True)} {colorize(image_path, Colors.YELLOW)}")
            
            # Print server configuration info if available
            if "config" in result:
                config = result["config"]
                print(f"\n{colorize(Icons.CONFIG, Colors.CYAN)} {colorize('Server Config:', Colors.WHITE, bold=True)} "
                      f"{colorize('max_tokens=', Colors.CYAN)}{colorize(str(config.get('max_tokens', 'unknown')), Colors.WHITE)} "
                      f"{colorize('temperature=', Colors.CYAN)}{colorize(str(config.get('temperature', 'unknown')), Colors.WHITE)} "
                      f"{colorize('tokens_used=', Colors.CYAN)}{colorize(str(config.get('actual_tokens', 'unknown')), Colors.WHITE)}")
            
            # Print detailed structured response (including reasoning)
            print_structured_response(result["response"])
            
            # Print enhanced analysis summary
            print_analysis_summary(result)
            
            # Print footer
            print(f"\n{colorize('═' * 80, Colors.GREEN)}")
            print(f"{colorize(Icons.SUCCESS, Colors.GREEN)} {colorize('Surgical environment analysis completed!', Colors.GREEN, bold=True)}")
            
        else:
            print(f"\n{colorize(Icons.ERROR, Colors.RED)} {colorize('Analysis request failed!', Colors.RED, bold=True)}")
            print(f"{colorize('Status code:', Colors.RED)} {response.status_code}")
            print(f"{colorize('Error message:', Colors.RED)} {response.text}")
            
    except requests.exceptions.ConnectionError:
        print(f"\n{colorize(Icons.ERROR, Colors.RED)} {colorize('Connection Error:', Colors.RED, bold=True)} Cannot connect to server.")
        print(f"{colorize('Please ensure the server is running at:', Colors.YELLOW)} {server_url}")
    except requests.exceptions.RequestException as e:
        print(f"\n{colorize(Icons.ERROR, Colors.RED)} {colorize('Request Error:', Colors.RED, bold=True)} {e}")

def load_system_message_from_file(file_path: str) -> Optional[str]:
    """
    Load system message from a text file.
    
    Args:
        file_path (str): Path to the text file containing system message
        
    Returns:
        str: Content of the file or None if file doesn't exist
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            print(f"{colorize(Icons.SUCCESS, Colors.GREEN)} System message loaded from: {file_path}")
            return content
    except FileNotFoundError:
        print(f"{colorize(Icons.ERROR, Colors.RED)} System message file not found: {file_path}")
        return None
    except Exception as e:
        print(f"{colorize(Icons.ERROR, Colors.RED)} Error reading system message file: {e}")
        return None

def run_robot_mode(args):
    """
    Run robot mode: monitor cameras and capture images for analysis.
    """
    print_header("ROBOT CAMERA MONITORING MODE", Icons.ROBOT, Colors.BLUE)
    
    # Initialize robot
    robot = SO101Robot(
        port_follower=args.port_follower,
        calibrate=args.calibrate,
        enable_camera=True
    )
    
    try:
        with robot.activate():
            while True:  # Main loop for multiple rounds
                print(f"\n{colorize(Icons.INFO, Colors.CYAN)} Robot connected successfully!")
                print(f"{colorize('Camera monitoring active. Controls:', Colors.WHITE, bold=True)}")
                print(f"  {colorize('c', Colors.RED, bold=True)} - Capture image from room camera and analyze")
                print(f"  {colorize('e', Colors.RED, bold=True)} - Execute last analyzed commands on robot")
                print(f"  {colorize('q', Colors.RED, bold=True)} - Quit application")
                
                last_commands = []  # Store commands from last analysis
                
                # Setup camera windows with specific positions and sizes
                window_width, window_height = 640, 480  # Smaller window size
                cv2.namedWindow("Wrist Camera", cv2.WINDOW_NORMAL)
                cv2.namedWindow("Room Camera", cv2.WINDOW_NORMAL)
                
                # Position windows on the left side of screen
                cv2.moveWindow("Wrist Camera", 50, 0)      # Top left
                cv2.moveWindow("Room Camera", 50, 580)       # Bottom left
                
                # Resize windows
                cv2.resizeWindow("Wrist Camera", window_width, window_height)
                cv2.resizeWindow("Room Camera", window_width, window_height)
                
                round_active = True
                while round_active:
                    try:
                        # Get current images from both cameras
                        img_wrist, img_room = robot.get_current_img()
                        
                        # Resize images to match window size
                        img_wrist_resized = cv2.resize(img_wrist, (window_width, window_height))
                        img_room_resized = cv2.resize(img_room, (window_width, window_height))
                        
                        # Convert to BGR for OpenCV display (no text overlays)
                        img_wrist_bgr = cv2.cvtColor(img_wrist_resized, cv2.COLOR_RGB2BGR)
                        img_room_bgr = cv2.cvtColor(img_room_resized, cv2.COLOR_RGB2BGR)
                        
                        # Display clean camera feeds without any text overlays
                        cv2.imshow("Wrist Camera", img_wrist_bgr)
                        cv2.imshow("Room Camera", img_room_bgr)
                        
                        # Handle key presses
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('c'):
                            print(f"\n{colorize(Icons.ANALYSIS, Colors.CYAN)} Capturing image from room camera...")
                            
                            # Convert room camera image to PIL format (use original size)
                            img_pil = Image.fromarray(img_room)
                            
                            # Convert to base64 for sending
                            img_buffer = io.BytesIO()
                            img_pil.save(img_buffer, format='JPEG')
                            img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
                            
                            # Send for analysis and get commands
                            commands = send_robot_image_request(
                                img_base64, 
                                server_url=args.server,
                                max_tokens=args.max_tokens,
                                temperature=args.temperature
                            )
                            
                            if commands:
                                last_commands = commands
                                print(f"{colorize('Ready to execute commands. Press', Colors.CYAN)} {colorize('e', Colors.RED, bold=True)} {colorize('to execute.', Colors.CYAN)}")
                            
                        elif key == ord('e') and last_commands:
                            print(f"\n{colorize(Icons.GEAR, Colors.BLUE)} Starting robot command execution...")
                            result = execute_robot_commands(
                                robot, 
                                last_commands,
                                host=args.groot_host,
                                port=args.groot_port,
                                actions_per_task=args.actions_per_task,
                                window_width=window_width,
                                window_height=window_height
                            )
                            
                            if result == "new_round":
                                # Return to home position and start new round
                                print(f"{colorize('Returning robot to home position...', Colors.CYAN)}")
                                robot.go_home()
                                last_commands = []  # Clear executed commands
                                round_active = False  # Exit inner loop to start new round
                                
                            elif result == "exit":
                                print(f"\n{colorize(Icons.INFO, Colors.CYAN)} Exiting robot mode...")
                                return  # Exit completely
                                
                            elif result:
                                # Normal completion - return to home and ready for next capture
                                print(f"{colorize('Returning robot to home position...', Colors.CYAN)}")
                                robot.go_home()
                                last_commands = []  # Clear executed commands
                                print(f"{colorize('Ready for next capture. Press', Colors.CYAN)} {colorize('c', Colors.RED, bold=True)} {colorize('to capture.', Colors.CYAN)}")
                            
                        elif key == ord('q'):
                            print(f"\n{colorize(Icons.INFO, Colors.CYAN)} Exiting robot mode...")
                            return
                            
                    except Exception as e:
                        print(f"{colorize(Icons.ERROR, Colors.RED)} Error in camera loop: {e}")
                        break
                        
    except Exception as e:
        print(f"{colorize(Icons.ERROR, Colors.RED)} Robot connection failed: {e}")
    finally:
        cv2.destroyAllWindows()

def send_robot_image_request(image_base64: str, server_url: str = "http://localhost:8000", 
                           max_tokens: int = 1024, temperature: float = 0.0) -> List[str]:
    """Send captured robot image for analysis and return extracted commands."""
    print_config_info(max_tokens, temperature)
    
    request_data = {
        "image_base64": image_base64,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    print(f"\n{colorize(Icons.GEAR, Colors.CYAN)} {colorize('Sending robot camera image for analysis...', Colors.CYAN)}")
    
    try:
        response = requests.post(
            f"{server_url}/generate",
            json=request_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            
            print_header("ROBOT CAMERA ANALYSIS RESULT", Icons.MEDICAL, Colors.GREEN)
            print(f"\n{colorize(Icons.INFO, Colors.CYAN)} {colorize('Source:', Colors.WHITE, bold=True)} {colorize('Robot Room Camera', Colors.YELLOW)}")
            
            if "config" in result:
                config = result["config"]
                print(f"\n{colorize(Icons.CONFIG, Colors.CYAN)} {colorize('Server Config:', Colors.WHITE, bold=True)} "
                      f"{colorize('max_tokens=', Colors.CYAN)}{colorize(str(config.get('max_tokens', 'unknown')), Colors.WHITE)} "
                      f"{colorize('temperature=', Colors.CYAN)}{colorize(str(config.get('temperature', 'unknown')), Colors.WHITE)} "
                      f"{colorize('tokens_used=', Colors.CYAN)}{colorize(str(config.get('actual_tokens', 'unknown')), Colors.WHITE)}")
            
            print_structured_response(result["response"])
            print_analysis_summary(result)
            
            # Extract robot commands
            commands = parse_robot_commands(result["response"])
            
            if commands:
                print(f"\n{colorize('🤖 EXTRACTED ROBOT COMMANDS FOR EXECUTION:', Colors.MAGENTA, bold=True)}")
                print(f"{colorize('─' * 70, Colors.MAGENTA)}")
                for i, cmd in enumerate(commands, 1):
                    print(f"  {colorize(f'{i}.', Colors.CYAN)} {colorize(cmd, Colors.WHITE)}")
                print(f"\n{colorize('Press', Colors.CYAN)} {colorize('e', Colors.RED, bold=True)} {colorize('to execute these commands on the robot.', Colors.CYAN)}")
            else:
                print(f"\n{colorize('No executable robot commands found in response.', Colors.YELLOW)}")
            
            print(f"\n{colorize('═' * 80, Colors.GREEN)}")
            print(f"{colorize(Icons.SUCCESS, Colors.GREEN)} {colorize('Robot camera analysis completed!', Colors.GREEN, bold=True)}")
            
            return commands
            
        else:
            print(f"\n{colorize(Icons.ERROR, Colors.RED)} {colorize('Analysis request failed!', Colors.RED, bold=True)}")
            print(f"{colorize('Status code:', Colors.RED)} {response.status_code}")
            print(f"{colorize('Error message:', Colors.RED)} {response.text}")
            return []
            
    except requests.exceptions.ConnectionError:
        print(f"\n{colorize(Icons.ERROR, Colors.RED)} {colorize('Connection Error:', Colors.RED, bold=True)} Cannot connect to server.")
        print(f"{colorize('Please ensure the server is running at:', Colors.YELLOW)} {server_url}")
        return []
    except requests.exceptions.RequestException as e:
        print(f"\n{colorize(Icons.ERROR, Colors.RED)} {colorize('Request Error:', Colors.RED, bold=True)} {e}")
        return []

def main():
    """Main function to parse arguments and send request."""
    parser = argparse.ArgumentParser(
        description=f"{Icons.ROBOT} Surgical Environment Analysis Client for SOARM 101",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
{colorize('ENVIRONMENT KNOWLEDGE:', Colors.CYAN, bold=True)}
  🔲 White foam board = surgical tray
  📦 Metal box = surgical tool box  
  🦾 White robot arm = robot scrub nurse (SOARM 101)

{colorize('SUPPORTED TOOLS:', Colors.YELLOW, bold=True)}
  ✂️  Metal surgical scissors (straight blades)
  🔧 Metal surgical tweezers

{colorize('Examples:', Colors.CYAN, bold=True)}
  {colorize('Analyze image file:', Colors.WHITE)}
    python robot_image_analyze.py image.jpg
    
  {colorize('Robot camera mode:', Colors.WHITE)}
    python robot_image_analyze.py --robot
    
  {colorize('Remote server analysis:', Colors.WHITE)}
    python robot_image_analyze.py image.jpg --server http://10.176.228.194:8000
        """
    )
    
    # Image source (mutually exclusive)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("image_path", nargs='?', help="Path to the image file to analyze")
    source_group.add_argument("--robot", action="store_true", help="Use robot camera for live monitoring and capture")

    # Robot-specific arguments
    parser.add_argument("--port_follower", type=str, default="/dev/ttyACM1", help="SO101 serial port (robot mode only)")
    parser.add_argument("--calibrate", action="store_true", help="Run robot calibration (robot mode only)")
    
    # GR00T server arguments (for robot command execution)
    parser.add_argument("--groot_host", type=str, default="localhost", help="GR00T server host for robot execution")
    parser.add_argument("--groot_port", type=int, default=5555, help="GR00T server port for robot execution")
    parser.add_argument("--actions_per_task", type=int, default=50, help="Number of actions to execute per robot task")

    # Optional arguments
    parser.add_argument(
        "--message", "-m", 
        help="Custom user message (optional)",
        default=None
    )
    
    parser.add_argument(
        "--system-message", "-sm",
        help="Custom system message (optional)",
        default=None
    )
    
    parser.add_argument(
        "--system-file", "-sf",
        help="Load system message from file (optional)",
        default=None
    )
    
    parser.add_argument(
        "--max-tokens", "-t",
        type=int,
        help="Maximum tokens for response generation (default: 1024, range: 128-8192)",
        default=1024
    )
    
    parser.add_argument(
        "--temperature", "-temp",
        type=float,
        help="Temperature for response generation (default: 0.0, range: 0.0-2.0)",
        default=0.0
    )
    
    parser.add_argument(
        "--server", "-s",
        help="Server URL (default: http://localhost:8000)",
        default="http://localhost:8000"
    )
    
    parser.add_argument(
        "--use-base64", "-b",
        action="store_true",
        help="Encode image as base64 and send in request body (for remote servers)"
    )
    
    args = parser.parse_args()

    # Check if robot mode is requested
    if args.robot:
        run_robot_mode(args)
        return

    # Print welcome header for file mode
    print_header("SURGICAL ENVIRONMENT ANALYSIS CLIENT", Icons.ROBOT, Colors.MAGENTA)
    
    # Validate parameter ranges
    max_tokens = min(max(args.max_tokens, 128), 8192)
    temperature = min(max(args.temperature, 0.0), 2.0)
    
    if max_tokens != args.max_tokens:
        print(f"{colorize(Icons.WARNING, Colors.YELLOW)} max_tokens clamped to valid range: {max_tokens}")
    if temperature != args.temperature:
        print(f"{colorize(Icons.WARNING, Colors.YELLOW)} temperature clamped to valid range: {temperature}")
    
    # Load system message from file if specified
    system_message = args.system_message
    if args.system_file:
        file_content = load_system_message_from_file(args.system_file)
        if file_content:
            system_message = file_content
        else:
            sys.exit(1)
    
    # Send request
    send_request(
        image_path=args.image_path,
        user_message=args.message, 
        system_message=system_message, 
        server_url=args.server,
        max_tokens=max_tokens, 
        temperature=temperature, 
        use_base64=args.use_base64,
    )

if __name__ == "__main__":
    main() 
