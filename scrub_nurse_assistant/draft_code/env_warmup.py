#!/usr/bin/env python3
"""
LLM Warm-up Script for Surgical Environment Analysis

This script sends a warm-up request to teach the LLM about:
1. What surgical tools look like (scissors and tweezers)
2. The environment setup (tray, tool box, robotic arm)
3. The task context (scrub nurse simulation)
"""

import argparse
import requests
import sys
import base64
import os
from typing import Optional

# Terminal colors (reuse from main script)
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'

class Icons:
    ROBOT = "🤖"
    MEDICAL = "🏥"
    SUCCESS = "✅"
    ERROR = "❌"
    INFO = "ℹ️"
    THINKING = "💭"
    GEAR = "⚙️"
    TEACHING = "🎓"

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

def create_warmup_messages():
    """Create the warm-up system and user messages based on the actual reference image content."""
    
    system_message = """You are learning about a surgical environment simulation for SOARM 101 robotic scrub nurse tasks.

STUDY THIS REFERENCE IMAGE TO LEARN:

ENVIRONMENT COMPONENTS:
- LEFT SIDE: Stainless steel metal box = surgical tool storage container (target location)
- RIGHT SIDE: White foam board = surgical tray (current tool location)  
- TOP: White robotic arm = SOARM 101 scrub nurse robot (performs the cleaning task)

SURGICAL TOOLS TO RECOGNIZE (study these carefully):
- SURGICAL SCISSORS: Metal cutting instruments with two blades and circular finger holes (can be curved or straight design)
- SURGICAL TWEEZERS: Thin metal grasping instruments with pointed tips

NON-SURGICAL ITEMS TO IGNORE:
- Office supplies: Pens, markers (non-medical design)
- Plastic tools: Pink plastic scissors, colored plastic tweezers (plastic material, non-metallic)
- Personal items: Watches, jewelry

MATERIAL DISTINCTION:
- Metal surgical tools: Shiny metallic appearance, professional medical-grade finish
- Plastic tools: Colored plastic (pink, blue, etc.), non-metallic appearance
- Office supplies: Pens with non-medical design and function"""

    user_message = """This is your TRAINING IMAGE for surgical environment analysis.

LEARNING OBJECTIVES:
1. Recognize metal vs plastic materials
2. Distinguish surgical scissors (curved/straight) from plastic scissors
3. Identify surgical tweezers vs office supplies
4. Understand left-to-right analysis approach

Please analyze this reference image with 5 items on the white foam board (left to right):
1. Item 1 (leftmost): Identify material, type, surgical vs non-surgical
2. Item 2: Identify material, type, surgical vs non-surgical  
3. Item 3 (center): Identify material, type, surgical vs non-surgical
4. Item 4: Identify material, type, surgical vs non-surgical
5. Item 5 (rightmost): Identify material, type, surgical vs non-surgical

Focus on material distinction and medical vs non-medical classification."""

    return system_message, user_message

def send_warmup_request(server_url: str, reference_image_path: str, 
                       max_tokens: int = 1024, temperature: float = 0.0) -> bool:
    """
    Send warm-up request to teach the LLM about the surgical environment.
    
    Args:
        server_url (str): Server URL
        reference_image_path (str): Path to reference image
        max_tokens (int): Token limit
        temperature (float): Temperature setting
        
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"{colorize(Icons.GEAR, Colors.CYAN)} Preparing warm-up request...")
    
    # Create warm-up messages
    system_message, user_message = create_warmup_messages()
    
    # Prepare request data
    request_data = {
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system_message": system_message,
        "user_message": user_message
    }
    
    # Load and encode reference image
    if not os.path.exists(reference_image_path):
        print(f"{colorize(Icons.ERROR, Colors.RED)} Reference image not found: {reference_image_path}")
        return False
    
    try:
        with open(reference_image_path, "rb") as image_file:
            image_data = image_file.read()
            base64_encoded = base64.b64encode(image_data).decode('utf-8')
            request_data["image_base64"] = base64_encoded
            print(f"{colorize(Icons.INFO, Colors.CYAN)} Reference image loaded: {reference_image_path}")
            print(f"{colorize(Icons.INFO, Colors.CYAN)} Image encoded: {len(base64_encoded)} characters")
    except Exception as e:
        print(f"{colorize(Icons.ERROR, Colors.RED)} Failed to load reference image: {e}")
        return False
    
    print(f"{colorize(Icons.GEAR, Colors.CYAN)} Sending warm-up request to server...")
    
    try:
        # Send warm-up request
        response = requests.post(
            f"{server_url}/generate",
            json=request_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            
            print_header("LLM WARM-UP RESULT", Icons.TEACHING, Colors.GREEN)
            
            # Show configuration
            if "config" in result:
                config = result["config"]
                print(f"\n{colorize(Icons.INFO, Colors.CYAN)} Server Config: "
                      f"max_tokens={config.get('max_tokens', 'unknown')} "
                      f"temperature={config.get('temperature', 'unknown')} "
                      f"actual_tokens={config.get('actual_tokens', 'unknown')}")
            
            # Show response
            print(f"\n{colorize(Icons.THINKING, Colors.CYAN)} {colorize('LLM Learning Response:', Colors.WHITE, bold=True)}")
            print(f"{colorize('-' * 60, Colors.CYAN)}")
            print(result["response"])
            
            print(f"\n{colorize('=' * 80, Colors.GREEN)}")
            print(f"{colorize(Icons.SUCCESS, Colors.GREEN)} {colorize('Warm-up completed successfully!', Colors.GREEN, bold=True)}")
            print(f"{colorize('The LLM has learned about:', Colors.GREEN)}")
            print(f"{colorize('• White foam board = surgical tray', Colors.GREEN)}")
            print(f"{colorize('• Metal box = surgical tool box', Colors.GREEN)}")
            print(f"{colorize('• White robotic arm = Arm 101 scrub nurse', Colors.GREEN)}")
            print(f"{colorize('• Surgical scissors and tweezers recognition', Colors.GREEN)}")
            
            return True
        else:
            print(f"{colorize(Icons.ERROR, Colors.RED)} Warm-up request failed!")
            print(f"{colorize('Status code:', Colors.RED)} {response.status_code}")
            print(f"{colorize('Error message:', Colors.RED)} {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"{colorize(Icons.ERROR, Colors.RED)} Connection Error: Cannot connect to server.")
        print(f"{colorize('Please ensure the server is running at:', Colors.YELLOW)} {server_url}")
        return False
    except Exception as e:
        print(f"{colorize(Icons.ERROR, Colors.RED)} Request failed: {e}")
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description=f"{Icons.TEACHING} LLM Warm-up for Surgical Environment Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
{colorize('This script teaches the LLM about:', Colors.CYAN, bold=True)}
  • White foam board = surgical tray
  • Metal box = surgical tool box  
  • White robotic arm = Arm 101 scrub nurse
  • What surgical scissors and tweezers look like

{colorize('Examples:', Colors.CYAN, bold=True)}
  {colorize('Basic warm-up:', Colors.WHITE)}
    python warmup_llm.py reference_image.jpg
  
  {colorize('Remote server:', Colors.WHITE)}
    python warmup_llm.py reference_image.jpg --server http://10.176.228.194:8000
        """
    )
    
    # Required argument
    parser.add_argument("reference_image", help="Path to reference image showing surgical environment")
    
    # Optional arguments
    parser.add_argument(
        "--server", "-s",
        help="Server URL (default: http://localhost:8000)",
        default="http://localhost:8000"
    )
    
    parser.add_argument(
        "--max-tokens", "-t",
        type=int,
        help="Maximum tokens for response (default: 1024)",
        default=1024
    )
    
    parser.add_argument(
        "--temperature", "-temp",
        type=float,
        help="Temperature for response generation (default: 0.0)",
        default=0.0
    )
    
    args = parser.parse_args()
    
    # Print welcome header
    print_header("LLM WARM-UP FOR SURGICAL ENVIRONMENT", Icons.TEACHING, Colors.MAGENTA)
    
    # Validate parameters
    max_tokens = min(max(args.max_tokens, 128), 8192)
    temperature = min(max(args.temperature, 0.0), 2.0)
    
    # Send warm-up request
    success = send_warmup_request(
        server_url=args.server,
        reference_image_path=args.reference_image,
        max_tokens=max_tokens,
        temperature=temperature
    )
    
    if success:
        print(f"\n{colorize(Icons.SUCCESS, Colors.GREEN)} {colorize('LLM is now ready for surgical environment analysis!', Colors.GREEN, bold=True)}")
        print(f"{colorize('You can now run the main analysis script.', Colors.CYAN)}")
        sys.exit(0)
    else:
        print(f"\n{colorize(Icons.ERROR, Colors.RED)} {colorize('Warm-up failed. Please check the error messages above.', Colors.RED, bold=True)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 