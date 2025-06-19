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
        # Look for grip commands
        if "grip" in line.lower() and ("scissor" in line.lower() or "tweezer" in line.lower()):
            commands.append(line)
    
    return commands

def print_commands(commands: List[str]):
    """Print robot commands with appropriate icons."""
    if not commands:
        print(f"{colorize('No robot commands found in response.', Colors.YELLOW)}")
        return
    
    for i, command in enumerate(commands, 1):
        # Determine icon based on command content
        if "scissor" in command.lower():
            icon = Icons.SCISSORS
        elif "tweezer" in command.lower():
            icon = Icons.TWEEZERS
        else:
            icon = Icons.ARROW
        
        print(f"{colorize(f'{i}.', Colors.CYAN)} {colorize(icon, Colors.YELLOW)} {colorize(command, Colors.WHITE)}")

def print_structured_response(response_text: str):
    """
    Parse and print the structured response from the server.
    
    Args:
        response_text (str): Raw response text from the server
    """
    # Extract structured sections
    sections = extract_structured_response(response_text)
    
    # Print reasoning section
    if sections["reasoning"]:
        print_section("REASONING & ANALYSIS", sections["reasoning"], Icons.ANALYSIS, Colors.CYAN)
    
    # Print plan section with command extraction
    if sections["plan"]:
        print_section("EXECUTION PLAN", sections["plan"], Icons.PLAN, Colors.CYAN)
        
        # Extract and display commands
        commands = extract_plan_commands(sections["plan"])
        if commands:
            print(f"\n{colorize('Robot Commands:', Colors.MAGENTA, bold=True)}")
            print_commands(commands)
    
    # If no structured sections found, print raw response
    if not sections["reasoning"] and not sections["plan"]:
        print_section("RAW RESPONSE", response_text, Icons.INFO, Colors.YELLOW)

def capture_camera_image(camera_index: int = 0, preview: bool = False, save_debug: bool = False) -> Optional[Image.Image]:
    """
    Capture an image from the camera.
    
    Args:
        camera_index (int): Camera index (0 for default camera)
        preview (bool): Show preview window before capture
        save_debug (bool): Save captured image to disk for debugging
        
    Returns:
        PIL.Image.Image: Captured image or None if failed
    """
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"{colorize(Icons.ERROR, Colors.RED)} Failed to open camera {camera_index}")
        return None
    
    # Set camera properties for better quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)
    
    print(f"{colorize(Icons.INFO, Colors.CYAN)} Camera {camera_index} opened successfully")
    print(f"{colorize(Icons.INFO, Colors.CYAN)} Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    
    # Let camera warm up and clear buffer
    print(f"{colorize(Icons.INFO, Colors.CYAN)} Warming up camera...")
    for _ in range(10):
        cap.read()
    
    if preview:
        print(f"{colorize(Icons.INFO, Colors.CYAN)} Preview mode: Press SPACE to capture, 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"{colorize(Icons.ERROR, Colors.RED)} Failed to read from camera")
                cap.release()
                return None
            
            # Display the frame
            cv2.imshow('Camera Preview - Press SPACE to capture, Q to quit', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # Space to capture
                # Capture a fresh frame
                ret, fresh_frame = cap.read()
                if not ret:
                    print(f"{colorize(Icons.ERROR, Colors.RED)} Failed to capture fresh frame")
                    cap.release()
                    cv2.destroyAllWindows()
                    return None
                
                # Convert BGR to RGB and then to PIL Image
                rgb_frame = cv2.cvtColor(fresh_frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                
                cap.release()
                cv2.destroyAllWindows()
                print(f"{colorize(Icons.SUCCESS, Colors.GREEN)} Image captured!")
                
                # Always save captured image with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                saved_filename = f"captured_cam{camera_index}_{timestamp}.jpg"
                pil_image.save(saved_filename, quality=95)
                print(f"{colorize(Icons.SUCCESS, Colors.GREEN)} Image saved: {saved_filename}")
                
                return pil_image
            elif key == ord('q'):  # Quit
                cap.release()
                cv2.destroyAllWindows()
                print(f"{colorize(Icons.WARNING, Colors.YELLOW)} Capture cancelled")
                return None
    else:
        # Capture immediately without preview
        print(f"{colorize(Icons.INFO, Colors.CYAN)} Capturing image directly...")
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print(f"{colorize(Icons.ERROR, Colors.RED)} Failed to capture image")
            return None
        
        # Convert BGR to RGB and then to PIL Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        # Always save captured image with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_filename = f"captured_cam{camera_index}_{timestamp}.jpg"
        pil_image.save(saved_filename, quality=95)
        print(f"{colorize(Icons.SUCCESS, Colors.GREEN)} Image captured and saved: {saved_filename}")
        
        # Save debug image if requested (additional debug info)
        if save_debug:
            debug_filename = f"debug_capture_{camera_index}_{timestamp}.jpg"
            pil_image.save(debug_filename, quality=95)
            print(f"{colorize(Icons.SUCCESS, Colors.GREEN)} Debug image also saved: {debug_filename}")
        
        return pil_image

def send_request(image_path: Optional[str] = None, camera_index: Optional[int] = None,
                user_message: Optional[str] = None, system_message: Optional[str] = None, 
                server_url: str = "http://localhost:8000", max_tokens: int = 1024, 
                temperature: float = 0.0, use_base64: bool = False, 
                camera_preview: bool = False, debug_mode: bool = False):
    """
    Send analysis request to the server.
    """
    # Print configuration
    print_config_info(max_tokens, temperature)
    
    # Prepare request data
    request_data = {
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    # Handle image source
    pil_image = None
    captured_filename = None
    
    if camera_index is not None:
        # Capture from camera
        pil_image = capture_camera_image(camera_index, camera_preview, debug_mode)
        if pil_image is None:
            return
        
        # Store the filename for reference
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        captured_filename = f"captured_cam{camera_index}_{timestamp}.jpg"
        
        # Always use base64 for camera images since there's no file path
        use_base64 = True
        print(f"{colorize(Icons.INFO, Colors.CYAN)} Using camera capture with base64 encoding")
        print(f"{colorize(Icons.INFO, Colors.CYAN)} Image size: {pil_image.size}, mode: {pil_image.mode}")
        
    elif image_path:
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
        print(f"{colorize(Icons.ERROR, Colors.RED)} Either image_path or camera_index must be provided")
        return
    
    # Handle camera image encoding
    if pil_image and use_base64:
        try:
            # Convert PIL image to base64
            buffer = io.BytesIO()
            pil_image.save(buffer, format='JPEG', quality=95)
            image_data = buffer.getvalue()
            base64_encoded = base64.b64encode(image_data).decode('utf-8')
            request_data["image_base64"] = base64_encoded
            print(f"{colorize(Icons.INFO, Colors.CYAN)} Camera image encoded as base64: {len(base64_encoded)} characters")
            
            # Debug: Save the exact image being sent (if debug mode)
            if debug_mode:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                debug_sent_filename = f"debug_sent_to_server_{timestamp}.jpg"
                pil_image.save(debug_sent_filename, quality=95)
                print(f"{colorize(Icons.SUCCESS, Colors.GREEN)} Image being sent saved as: {debug_sent_filename}")
                
        except Exception as e:
            print(f"{colorize(Icons.ERROR, Colors.RED)} Failed to encode camera image: {e}")
            return
    
    # Use custom messages if provided
    if user_message:
        request_data["user_message"] = user_message
    if system_message:
        request_data["system_message"] = system_message
    
    print(f"\n{colorize(Icons.GEAR, Colors.CYAN)} {colorize('Sending request to server...', Colors.CYAN)}")
    
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
            
            # Print header
            print_header("SURGICAL ENVIRONMENT ANALYSIS RESULT", Icons.MEDICAL, Colors.GREEN)
            
            # Show which image was analyzed
            if captured_filename:
                print(f"\n{colorize(Icons.INFO, Colors.CYAN)} {colorize('Analyzed image:', Colors.WHITE, bold=True)} {colorize(captured_filename, Colors.YELLOW)}")
            elif image_path:
                print(f"\n{colorize(Icons.INFO, Colors.CYAN)} {colorize('Analyzed image:', Colors.WHITE, bold=True)} {colorize(image_path, Colors.YELLOW)}")
            
            # Print server configuration info if available
            if "config" in result:
                config = result["config"]
                print(f"\n{colorize(Icons.CONFIG, Colors.CYAN)} {colorize('Server Config:', Colors.WHITE, bold=True)} "
                      f"{colorize('max_tokens=', Colors.CYAN)}{colorize(str(config.get('max_tokens', 'unknown')), Colors.WHITE)} "
                      f"{colorize('temperature=', Colors.CYAN)}{colorize(str(config.get('temperature', 'unknown')), Colors.WHITE)} "
                      f"{colorize('actual_tokens=', Colors.CYAN)}{colorize(str(config.get('actual_tokens', 'unknown')), Colors.WHITE)}")
            
            # Print analysis summary
            print_analysis_summary(result)
            
            # Print footer
            print(f"\n{colorize('=' * 80, Colors.GREEN)}")
            print(f"{colorize(Icons.SUCCESS, Colors.GREEN)} {colorize('Analysis completed successfully!', Colors.GREEN, bold=True)}")
            
        else:
            print(f"\n{colorize(Icons.ERROR, Colors.RED)} {colorize('Request failed!', Colors.RED, bold=True)}")
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

def print_analysis_summary(result: dict):
    """Print detailed analysis summary from server response."""
    if "analysis" not in result:
        return
    
    analysis = result["analysis"]
    
    print(f"\n{colorize('SURGICAL TOOL ANALYSIS:', Colors.MAGENTA, bold=True)}")
    print(f"{colorize('─' * 50, Colors.MAGENTA)}")
    
    tool_count = analysis.get("surgical_tools_detected", 0)
    scissors_found = analysis.get("scissors_detected", False)
    curved_scissors_found = analysis.get("curved_scissors_detected", False)
    straight_scissors_found = analysis.get("straight_scissors_detected", False)
    tweezers_found = analysis.get("tweezers_detected", False)
    
    if tool_count > 0:
        print(f"{colorize(Icons.SUCCESS, Colors.GREEN)} {colorize('Surgical tools detected:', Colors.WHITE, bold=True)} {colorize(str(tool_count), Colors.GREEN, bold=True)}")
        
        if curved_scissors_found:
            print(f"  {colorize(Icons.SCISSORS, Colors.YELLOW)} {colorize('Metal curved surgical scissors found', Colors.WHITE)}")
        if straight_scissors_found:
            print(f"  {colorize(Icons.SCISSORS, Colors.YELLOW)} {colorize('Metal straight surgical scissors found', Colors.WHITE)}")
        if tweezers_found:
            print(f"  {colorize(Icons.TWEEZERS, Colors.YELLOW)} {colorize('Metal surgical tweezers found', Colors.WHITE)}")
        
        print(f"\n{colorize('ROBOT COMMANDS TO EXECUTE:', Colors.GREEN, bold=True)}")
        commands = analysis.get("robot_commands", [])
        for i, cmd in enumerate(commands, 1):
            if "no actions needed" not in cmd.lower():
                if "curved scissor" in cmd.lower():
                    icon = f"{Icons.SCISSORS}🔄"  # Curved scissors icon
                elif "straight scissor" in cmd.lower():
                    icon = f"{Icons.SCISSORS}➡️"  # Straight scissors icon
                else:
                    icon = Icons.TWEEZERS
                print(f"  {colorize(f'{i}.', Colors.CYAN)} {colorize(icon, Colors.YELLOW)} {colorize(cmd, Colors.WHITE)}")
    else:
        print(f"{colorize(Icons.INFO, Colors.YELLOW)} {colorize('No surgical tools detected', Colors.WHITE, bold=True)}")
        print(f"  {colorize('→ Only non-surgical items or empty tray', Colors.YELLOW)}")
        print(f"  {colorize('→ Robot will remain idle', Colors.YELLOW)}")

def main():
    """Main function to parse arguments and send request."""
    parser = argparse.ArgumentParser(
        description=f"{Icons.ROBOT} Surgical Environment Analysis Client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
{colorize('Examples:', Colors.CYAN, bold=True)}
  {colorize('Camera capture with preview:', Colors.WHITE)}
    python env_analyze.py --camera 0 --preview
  
  {colorize('Direct camera capture:', Colors.WHITE)}
    python env_analyze.py --camera 0
  
  {colorize('Analyze image file:', Colors.WHITE)}
    python env_analyze.py image.jpg
    
  {colorize('Remote server analysis:', Colors.WHITE)}
    python env_analyze.py image.jpg --server http://10.176.228.194:8000
    
  {colorize('Custom parameters:', Colors.WHITE)}
    python env_analyze.py --camera 0 --max-tokens 2048 --temperature 0.0
        """
    )
    
    # Image source (mutually exclusive)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("image_path", nargs='?', help="Path to the image file to analyze")
    source_group.add_argument("--camera", "-c", type=int, help="Camera index to capture from (e.g., 0, 1, 2)")
    
    # Camera options
    parser.add_argument(
        "--preview", 
        action="store_true",
        help="Show camera preview before capture (default: direct capture)"
    )
    
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug mode (saves captured images to disk)"
    )
    
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

    # Print welcome header
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
        camera_index=args.camera,
        user_message=args.message, 
        system_message=system_message, 
        server_url=args.server,
        max_tokens=max_tokens, 
        temperature=temperature, 
        use_base64=args.use_base64,
        camera_preview=args.preview,
        debug_mode=args.debug
    )

if __name__ == "__main__":
    main() 
