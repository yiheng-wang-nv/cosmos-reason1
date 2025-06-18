import requests
import argparse
import json
import sys
import os
import re
from typing import List, Optional

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
    PHASE = "🔄"
    CONFIG = "🎛️"

def colorize(text: str, color: str, bold: bool = False) -> str:
    """Apply color and formatting to text"""
    formatting = Colors.BOLD if bold else ""
    return f"{formatting}{color}{text}{Colors.RESET}"

def print_header(title: str, icon: str = "", color: str = Colors.CYAN):
    """Print a formatted header with icon and color"""
    line = "=" * 80
    print(f"\n{colorize(line, color)}")
    header_text = f"{icon} {title}" if icon else title
    print(f"{colorize(header_text.center(80), color, bold=True)}")
    print(f"{colorize(line, color)}")

def print_section(title: str, content: str, icon: str = "", color: str = Colors.WHITE):
    """Print a formatted section with title and content"""
    section_title = f"{icon} {title}" if icon else title
    print(f"\n{colorize(section_title, color, bold=True)}")
    print(f"{colorize('-' * 60, Colors.DIM)}")
    print(content)

def print_phase_info(phase: str):
    """Print phase information with appropriate styling"""
    phase_colors = {
        "initial": Colors.BLUE,
        "verification": Colors.GREEN,
        "auto": Colors.YELLOW
    }
    
    phase_descriptions = {
        "initial": "INITIAL ANALYSIS - Tool identification and task planning",
        "verification": "COMPLETION VERIFICATION - Task verification and status assessment",
        "auto": "AUTO-DETECTION - System will determine the appropriate phase"
    }
    
    color = phase_colors.get(phase, Colors.WHITE)
    description = phase_descriptions.get(phase, "Unknown phase")
    
    print(f"{colorize(Icons.PHASE, color)} {colorize('Phase:', Colors.WHITE, bold=True)} {colorize(description, color)}")

def print_config_info(max_tokens: int, temperature: float):
    """Print configuration information"""
    print(f"{colorize(Icons.CONFIG, Colors.CYAN)} {colorize('Config:', Colors.WHITE, bold=True)} "
          f"{colorize('max_tokens=', Colors.CYAN)}{colorize(str(max_tokens), Colors.WHITE)} "
          f"{colorize('temperature=', Colors.CYAN)}{colorize(str(temperature), Colors.WHITE)}")

def extract_structured_response(response_text: str) -> dict:
    """
    Extract structured sections from the LLM response.
    
    Args:
        response_text (str): The full LLM response
        
    Returns:
        dict: Dictionary with think, reasoning, and plan sections
    """
    sections = {
        'think': '',
        'reasoning': '',
        'plan': ''
    }
    
    # Regular expressions to extract sections
    patterns = {
        'think': r'<think>(.*?)</think>',
        'reasoning': r'<reasoning>(.*?)</reasoning>',
        'plan': r'<plan>(.*?)</plan>'
    }
    
    for section, pattern in patterns.items():
        match = re.search(pattern, response_text, re.DOTALL | re.IGNORECASE)
        if match:
            sections[section] = match.group(1).strip()
    
    return sections

def extract_plan_commands(plan_text: str, phase: Optional[str] = None) -> List[str]:
    """
    Extract plan commands and status from the plan section.
    
    Args:
        plan_text (str): The plan section text
        phase (str, optional): Current phase to determine which commands to extract
        
    Returns:
        list: List of extracted commands and status messages
    """
    commands = []
    lines = plan_text.split('\n')
    
    # Determine if we're in initial or verification phase based on content
    is_initial_phase = True
    if phase == "verification":
        is_initial_phase = False
    elif phase is None:  # Auto-detection
        # Check if the response contains completion verification language
        verification_indicators = [
            "Task completed successfully", "Task partially completed", 
            "Task verification", "properly stored", "completion verification"
        ]
        if any(indicator.lower() in plan_text.lower() for indicator in verification_indicators):
            # Look for initial commands too
            initial_indicators = ["Grip a tweezer", "Grip a straight scissor", "Grip"]
            has_initial_commands = any(indicator in plan_text for indicator in initial_indicators)
            
            # If we have both, prioritize initial commands unless they're clearly about verification
            if has_initial_commands:
                is_initial_phase = True
            else:
                is_initial_phase = False
    
    for line in lines:
        line = line.strip()
        if line:
            if is_initial_phase:
                # Look for exact command patterns - be more flexible with line detection
                line_lower = line.lower()
                
                # Check for tweezer command
                if ("grip a tweezer" in line_lower and "put it in the box" in line_lower) or \
                   line.strip() == "Grip a tweezer and put it in the box.":
                    commands.append("Grip a tweezer and put it in the box.")
                
                # Check for scissor command  
                elif ("grip a straight scissor" in line_lower and "put it in the box" in line_lower) or \
                     line.strip() == "Grip a straight scissor and put it in the box.":
                    commands.append("Grip a straight scissor and put it in the box.")
                
                # Check for no tools detected
                elif "no surgical tools detected" in line_lower and "tray" in line_lower:
                    commands.append("No surgical tools detected on the tray.")
                
                # Handle cases where commands might be on one line separated by periods or other separators
                elif "grip a tweezer" in line_lower and "grip a straight scissor" in line_lower:
                    # Split commands that are on the same line
                    if "grip a tweezer" in line_lower:
                        commands.append("Grip a tweezer and put it in the box.")
                    if "grip a straight scissor" in line_lower:
                        commands.append("Grip a straight scissor and put it in the box.")
                
            else:
                # Extract verification status only
                verification_patterns = [
                    "Task completed successfully", "Task partially completed", 
                    "Task verification failed"
                ]
                if any(pattern in line for pattern in verification_patterns):
                    clean_cmd = line.strip('- "').strip()
                    if clean_cmd:
                        commands.append(clean_cmd)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_commands = []
    for cmd in commands:
        if cmd not in seen:
            seen.add(cmd)
            unique_commands.append(cmd)
    
    return unique_commands

def print_commands(commands: List[str]):
    """Print extracted commands with appropriate icons and colors"""
    if not commands:
        print(f"{colorize(Icons.INFO, Colors.YELLOW)} No actionable commands found in response")
        return
    
    print_section("EXTRACTED ROBOT COMMANDS", "", Icons.GEAR, Colors.MAGENTA)
    
    for i, cmd in enumerate(commands, 1):
        # Determine command type and appropriate styling
        if "Grip a tweezer" in cmd:
            icon = Icons.TWEEZERS
            color = Colors.BLUE
        elif "Grip a straight scissor" in cmd or "Grip a scissor" in cmd:
            icon = Icons.SCISSORS
            color = Colors.BLUE
        elif "Task completed successfully" in cmd:
            icon = Icons.SUCCESS
            color = Colors.GREEN
        elif "Task partially completed" in cmd:
            icon = Icons.WARNING
            color = Colors.YELLOW
        elif "Task verification failed" in cmd:
            icon = Icons.ERROR
            color = Colors.RED
        elif "No surgical tools" in cmd:
            icon = Icons.INFO
            color = Colors.CYAN
        else:
            icon = Icons.BULLET
            color = Colors.WHITE
        
        print(f"{colorize(f'{i}.', Colors.DIM)} {colorize(icon, color)} {colorize(cmd, color)}")

def print_structured_response(response_text: str, phase: Optional[str] = None):
    """Print the structured response with colors and formatting"""
    sections = extract_structured_response(response_text)
    
    # Print thinking section
    if sections['think']:
        print_section("THINKING PROCESS", sections['think'], Icons.THINKING, Colors.BLUE)
    
    # Print reasoning section
    if sections['reasoning']:
        print_section("REASONING & ANALYSIS", sections['reasoning'], Icons.ANALYSIS, Colors.YELLOW)
    
    # Print plan section
    if sections['plan']:
        print_section("EXECUTION PLAN", sections['plan'], Icons.PLAN, Colors.GREEN)
        
        # Extract and display commands with phase context
        commands = extract_plan_commands(sections['plan'], phase)
        if commands:
            print()  # Add spacing
            print_commands(commands)
    
    # If no structured sections found, print the raw response
    if not any(sections.values()):
        print_section("RAW RESPONSE", response_text, Icons.INFO, Colors.WHITE)

def send_request(image_path: str, user_message: Optional[str] = None, 
                system_message: Optional[str] = None, server_url: str = "http://localhost:8000", 
                phase: Optional[str] = None, max_tokens: int = 1024, temperature: float = 0.0):
    """
    Send a request to the LLM server for surgical tool analysis.
    
    Args:
        image_path (str): Path to the image file
        user_message (str, optional): Custom user message. Defaults to None.
        system_message (str, optional): Custom system message. Defaults to None.
        server_url (str): Server URL. Defaults to "http://localhost:8000".
        phase (str, optional): Operation phase - "initial", "verification", or None for auto-detection.
        max_tokens (int): Maximum tokens for response generation. Defaults to 1024.
        temperature (float): Temperature for response generation. Defaults to 0.0.
    """
    
    # Check if image file exists
    if not os.path.exists(image_path):
        print(f"{colorize(Icons.ERROR, Colors.RED)} {colorize('Error:', Colors.RED, bold=True)} Image file does not exist: {image_path}")
        sys.exit(1)
    
    # Generate phase-specific user message if phase is specified and no custom user message provided
    if phase and not user_message:
        if phase == "initial":
            user_message = (
                "Analyze this surgical room image for INITIAL TASK PLANNING. "
                "Identify surgical tools on the white foam board that need to be moved to the tool box. "
                "Generate appropriate movement commands for the SOARM 101 robotic arm."
            )
        elif phase == "verification":
            user_message = (
                "Analyze this surgical room image for COMPLETION VERIFICATION. "
                "Verify that surgical tools have been properly stored in the metal tool box "
                "and confirm the white foam board is clear of surgical instruments. "
                "Provide task completion status assessment."
            )
    
    # Print phase and configuration information
    if phase:
        print_phase_info(phase)
    else:
        print_phase_info("auto")
    
    print_config_info(max_tokens, temperature)
    
    # Prepare request data
    request_data = {
        "image_path": image_path,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    # Add phase if specified
    if phase:
        request_data["phase"] = phase
    
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
            print_header("SURGICAL SCRUB NURSE ANALYSIS RESULT", Icons.MEDICAL, Colors.GREEN)
            
            # Print server configuration info if available
            if "config" in result:
                config = result["config"]
                phase_info = f"{colorize('phase=', Colors.CYAN)}{colorize(str(config.get('phase', 'unknown')), Colors.WHITE)} " if 'phase' in config else ""
                print(f"\n{colorize(Icons.CONFIG, Colors.CYAN)} {colorize('Server Config:', Colors.WHITE, bold=True)} "
                      f"{phase_info}"
                      f"{colorize('max_tokens=', Colors.CYAN)}{colorize(str(config.get('max_tokens', 'unknown')), Colors.WHITE)} "
                      f"{colorize('temperature=', Colors.CYAN)}{colorize(str(config.get('temperature', 'unknown')), Colors.WHITE)} "
                      f"{colorize('actual_tokens=', Colors.CYAN)}{colorize(str(config.get('actual_tokens', 'unknown')), Colors.WHITE)}")
            
            # Print structured response with phase context
            print_structured_response(result["response"], phase)
            
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

def main():
    """Main function to parse arguments and send request."""
    parser = argparse.ArgumentParser(
        description=f"{Icons.ROBOT} Surgical Environment Analysis Client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
{colorize('Examples:', Colors.CYAN, bold=True)}
  {colorize('Basic usage:', Colors.WHITE)}
    python surgical_env_analyze.py image.jpg
  
  {colorize('Phase-specific analysis:', Colors.WHITE)}
    python surgical_env_analyze.py image.jpg --phase initial
    python surgical_env_analyze.py image.jpg --phase verification
  
  {colorize('Custom token limits:', Colors.WHITE)}
    python surgical_env_analyze.py image.jpg --max-tokens 2048
    python surgical_env_analyze.py image.jpg --max-tokens 4096 --temperature 0.2
  
  {colorize('Custom system message:', Colors.WHITE)}
    python surgical_env_analyze.py image.jpg --system-file custom_prompt.txt
  
  {colorize('Full configuration:', Colors.WHITE)}
    python surgical_env_analyze.py image.jpg --phase initial --max-tokens 2048 --temperature 0.1
        """
    )
    
    # Required argument: image path
    parser.add_argument("image_path", nargs='?', help="Path to the image file to analyze")
    
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
        "--phase", "-p",
        choices=["initial", "verification"],
        help="Operation phase: 'initial' for task planning, 'verification' for completion check",
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
    
    args = parser.parse_args()

    # Print welcome header
    print_header("SURGICAL ENVIRONMENT ANALYSIS CLIENT", Icons.ROBOT, Colors.MAGENTA)

    # Require image path for analysis
    if not args.image_path:
        print(f"{colorize(Icons.ERROR, Colors.RED)} {colorize('Error:', Colors.RED, bold=True)} Image path is required for analysis")
        parser.print_help()
        sys.exit(1)
    
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
    send_request(args.image_path, args.message, system_message, args.server, 
                args.phase, max_tokens, temperature)

if __name__ == "__main__":
    main() 
