"""
Surgical Analysis Client - Single image version with enhanced display
"""

import os
import requests
import re
from pathlib import Path
from typing import Optional


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
    """Unicode icons for surgical workflow representation"""
    ROBOT = "🤖"
    MEDICAL = "🏥"
    NEEDLE = "🪡"
    FORCEPS = "🔧"
    SUCCESS = "✅"
    WARNING = "⚠️"
    ERROR = "❌"
    INFO = "ℹ️"
    ANALYSIS = "🔍"
    THINKING = "💭"
    COMMAND = "⚡"
    ARROW = "➤"
    BULLET = "•"
    GEAR = "⚙️"
    CONFIG = "🎛️"
    PICK = "✋"
    HANDOVER = "🤝"
    THROW = "🎯"
    KNOT = "🔗"
    OTHER = "❓"


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


def print_config_info(image_path: str, max_tokens: int, temperature: float):
    """Print configuration information."""
    print(f"{colorize(Icons.CONFIG, Colors.CYAN)} {colorize('Config:', Colors.WHITE, bold=True)} "
          f"{colorize('image=', Colors.CYAN)}{colorize(Path(image_path).name, Colors.WHITE)} "
          f"{colorize('max_tokens=', Colors.CYAN)}{colorize(str(max_tokens), Colors.WHITE)} "
          f"{colorize('temperature=', Colors.CYAN)}{colorize(str(temperature), Colors.WHITE)}")


def extract_structured_response(response_text: str) -> dict:
    """
    Extract reasoning and robot_command sections from the response.
    
    Args:
        response_text (str): Raw response text from the server
        
    Returns:
        dict: Dictionary with 'reasoning' and 'robot_command' keys
    """
    result = {"reasoning": "", "robot_command": ""}
    
    # Extract reasoning section
    reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', response_text, re.DOTALL | re.IGNORECASE)
    if reasoning_match:
        result["reasoning"] = reasoning_match.group(1).strip()
    
    # Extract robot_command section
    command_match = re.search(r'<robot_command>(.*?)</robot_command>', response_text, re.DOTALL | re.IGNORECASE)
    if command_match:
        result["robot_command"] = command_match.group(1).strip()
    
    # Fallback: if no XML tags found, use the entire response as reasoning
    if not result["reasoning"] and not result["robot_command"]:
        result["reasoning"] = response_text.strip()
    
    return result


def get_command_icon(command: str) -> str:
    """Get appropriate icon for robot command."""
    command_icons = {
        "needle_pick": Icons.PICK,
        "needle_handover": Icons.HANDOVER,
        "needle_throw": Icons.THROW,
        "knot_tying": Icons.KNOT,
        "others": Icons.OTHER
    }
    return command_icons.get(command.lower(), Icons.COMMAND)


def get_command_color(command: str) -> str:
    """Get appropriate color for robot command."""
    command_colors = {
        "needle_pick": Colors.BLUE,
        "needle_handover": Colors.YELLOW,
        "needle_throw": Colors.RED,
        "knot_tying": Colors.GREEN,
        "others": Colors.MAGENTA
    }
    return command_colors.get(command.lower(), Colors.WHITE)


def print_structured_response(response_text: str):
    """
    Parse and print the structured response from the server with enhanced formatting.
    """
    # Extract structured sections
    sections = extract_structured_response(response_text)
    
    # Print reasoning section with enhanced formatting
    if sections["reasoning"]:
        print(f"\n{colorize(f'{Icons.THINKING} SURGICAL ANALYSIS & REASONING:', Colors.CYAN, bold=True)}")
        print(f"{colorize('─' * 70, Colors.CYAN)}")
        
        # Format reasoning text with proper indentation
        reasoning_lines = sections["reasoning"].split('\n')
        for line in reasoning_lines:
            if line.strip():
                print(f"  {colorize(line.strip(), Colors.WHITE)}")
            else:
                print()  # Preserve empty lines for readability
    else:
        # If no reasoning found in XML tags, print the full response as reasoning
        print(f"\n{colorize(f'{Icons.THINKING} SURGICAL ANALYSIS & REASONING:', Colors.CYAN, bold=True)}")
        print(f"{colorize('─' * 70, Colors.CYAN)}")
        
        # Look for any text before robot_command tag
        if "<robot_command>" in response_text:
            reasoning_part = response_text.split("<robot_command>")[0].strip()
            if reasoning_part:
                reasoning_lines = reasoning_part.split('\n')
                for line in reasoning_lines:
                    if line.strip():
                        print(f"  {colorize(line.strip(), Colors.WHITE)}")
        else:
            # Print full response if no tags found
            print(f"  {colorize(response_text.strip(), Colors.WHITE)}")
    
    # Print robot command section with highlighted command
    if sections["robot_command"]:
        command = sections["robot_command"].strip()
        command_icon = get_command_icon(command)
        command_color = get_command_color(command)
        
        print(f"\n{colorize(f'{Icons.COMMAND} ROBOT COMMAND:', Colors.MAGENTA, bold=True)}")
        print(f"{colorize('─' * 70, Colors.MAGENTA)}")
        
        # Highlight the command with icon and color
        print(f"  {colorize(command_icon, command_color)} {colorize(command.upper(), command_color, bold=True)}")
        
        # Add command description
        command_descriptions = {
            "needle_pick": "Robot should grasp the needle with needle driver",
            "needle_handover": "Robot should transfer needle from driver to forceps",
            "needle_throw": "Robot should drive needle through tissue using forceps",
            "knot_tying": "Robot should create and tighten surgical knots",
            "others": "No clear suturing task detected"
        }
        
        description = command_descriptions.get(command.lower(), "Unknown command")
        print(f"    {colorize('→', Colors.CYAN)} {colorize(description, Colors.WHITE)}")
    else:
        print(f"\n{colorize(f'{Icons.ERROR} NO ROBOT COMMAND FOUND', Colors.RED, bold=True)}")


def load_message_from_file(filename: str) -> Optional[str]:
    """Load message from file if it exists"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, filename)
    
    try:
        with open(file_path, "r") as f:
            content = f.read().strip()
            print(f"{colorize(Icons.SUCCESS, Colors.GREEN)} Message loaded from: {colorize(file_path, Colors.YELLOW)}")
            return content
    except FileNotFoundError:
        print(f"{colorize(Icons.ERROR, Colors.RED)} Message file not found: {colorize(file_path, Colors.YELLOW)}")
        return None


class SurgicalClient:
    """Client for surgical analysis API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def analyze_image(self, 
                     image_path: str,
                     system_message: Optional[str] = None,
                     user_message: Optional[str] = None,
                     max_tokens: int = 4096,
                     temperature: float = 0.0) -> dict:
        """Analyze single image from file path"""
        data = {
            "image_path": image_path,
            "system_message": system_message,
            "user_message": user_message,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        response = requests.post(f"{self.base_url}/analyze", json=data)
        response.raise_for_status()
        return response.json()
    
    def is_ready(self) -> bool:
        """Check if server is ready"""
        try:
            response = requests.get(f"{self.base_url}/health")
            return response.status_code == 200
        except:
            return False


def main():
    """Enhanced client with beautiful output formatting"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description=f"{Icons.NEEDLE} Surgical Suturing Workflow Analysis Client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
{colorize('SUTURING WORKFLOW:', Colors.CYAN, bold=True)}
  {colorize(f'{Icons.PICK} needle_pick', Colors.BLUE)} → {colorize(f'{Icons.HANDOVER} needle_handover', Colors.YELLOW)} → {colorize(f'{Icons.THROW} needle_throw', Colors.RED)} → {colorize(f'{Icons.KNOT} knot_tying', Colors.GREEN)}

{colorize('DA VINCI INSTRUMENTS:', Colors.YELLOW, bold=True)}
  🔧 Needle Driver (left): Picks up needle initially
  🔧 Bipolar Forceps (right): Receives needle and performs throwing

{colorize('Examples:', Colors.CYAN, bold=True)}
  {colorize('Analyze image:', Colors.WHITE)}
    python client.py image.jpg
    
  {colorize('Remote server:', Colors.WHITE)}
    python client.py image.jpg --server 10.176.228.194:8000
        """
    )
    
    parser.add_argument("image", help="Image file to analyze")
    parser.add_argument("--server", default="http://localhost:8000", help="Server URL")
    parser.add_argument("--system", help="Custom system message")
    parser.add_argument("--user", help="Custom user message")
    parser.add_argument("--system-file", help="Load system message from file")
    parser.add_argument("--user-file", help="Load user message from file")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Max tokens")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature")
    
    args = parser.parse_args()
    
    # Print welcome header
    print_header("SURGICAL SUTURING WORKFLOW ANALYSIS", Icons.MEDICAL, Colors.MAGENTA)
    
    # Check image exists
    if not Path(args.image).exists():
        print(f"{colorize(Icons.ERROR, Colors.RED)} Image not found: {colorize(args.image, Colors.YELLOW)}")
        return 1
    
    client = SurgicalClient(args.server)
    
    # Check server
    print(f"\n{colorize(Icons.GEAR, Colors.CYAN)} {colorize('Checking server connection...', Colors.CYAN)}")
    if not client.is_ready():
        print(f"{colorize(Icons.ERROR, Colors.RED)} Server not ready at {colorize(args.server, Colors.YELLOW)}")
        print(f"{colorize('Please ensure the server is running:', Colors.RED)} python server.py")
        return 1
    
    print(f"{colorize(Icons.SUCCESS, Colors.GREEN)} Server is ready")
    
    # Load messages from files if specified
    system_message = args.system
    user_message = args.user
    
    if args.system_file:
        system_message = load_message_from_file(args.system_file)
    
    if args.user_file:
        user_message = load_message_from_file(args.user_file)
    
    # Print configuration
    print_config_info(args.image, args.max_tokens, args.temperature)
    
    print(f"\n{colorize(Icons.ANALYSIS, Colors.CYAN)} {colorize('Analyzing surgical image...', Colors.CYAN)}")
    print(f"{colorize('Image:', Colors.CYAN)} {colorize(Path(args.image).name, Colors.WHITE, bold=True)}")
    
    try:
        result = client.analyze_image(
            args.image,
            system_message, 
            user_message,
            args.max_tokens,
            args.temperature
        )
        
        if result["success"]:
            # Print main analysis result
            print_header("SUTURING WORKFLOW ANALYSIS RESULT", Icons.NEEDLE, Colors.GREEN)
            
            # Print server configuration info if available
            if "config" in result:
                config = result["config"]
                print(f"\n{colorize(Icons.CONFIG, Colors.CYAN)} {colorize('Server Config:', Colors.WHITE, bold=True)} "
                      f"{colorize('max_tokens=', Colors.CYAN)}{colorize(str(config.get('max_tokens', 'unknown')), Colors.WHITE)} "
                      f"{colorize('temperature=', Colors.CYAN)}{colorize(str(config.get('temperature', 'unknown')), Colors.WHITE)} "
                      f"{colorize('tokens_used=', Colors.CYAN)}{colorize(str(config.get('actual_tokens', 'unknown')), Colors.WHITE)}")
            
            # Print detailed structured response
            print_structured_response(result["analysis"])
            
            # Print footer
            print(f"\n{colorize('═' * 80, Colors.GREEN)}")
            print(f"{colorize(Icons.SUCCESS, Colors.GREEN)} {colorize('Surgical workflow analysis completed!', Colors.GREEN, bold=True)}")
            
        else:
            print(f"\n{colorize(Icons.ERROR, Colors.RED)} {colorize('Analysis failed!', Colors.RED, bold=True)}")
            print(f"{colorize('Error:', Colors.RED)} {result['error']}")
            return 1
            
    except requests.exceptions.ConnectionError:
        print(f"\n{colorize(Icons.ERROR, Colors.RED)} {colorize('Connection Error:', Colors.RED, bold=True)} Cannot connect to server.")
        print(f"{colorize('Please ensure the server is running at:', Colors.YELLOW)} {args.server}")
        return 1
    except requests.exceptions.RequestException as e:
        print(f"\n{colorize(Icons.ERROR, Colors.RED)} {colorize('Request Error:', Colors.RED, bold=True)} {e}")
        return 1
    except Exception as e:
        print(f"{colorize(Icons.ERROR, Colors.RED)} Request failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 