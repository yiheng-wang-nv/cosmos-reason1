#!/usr/bin/env python3
"""
Test script for Surgical Scrub Nurse LLM System

This script helps test different prompt configurations with your surgical images
for both initial task planning and completion verification phases.
"""

import argparse
import sys
import os
sys.path.append(os.path.dirname(__file__))

from surgical_prompts import get_prompt, list_prompts, PROMPT_CONFIGS
from request_llm import send_request

def test_with_prompt_config(image_path: str, prompt_name: str, server_url: str = "http://localhost:8000", 
                           phase: str = "auto"):
    """
    Test the system with a specific prompt configuration.
    
    Args:
        image_path (str): Path to the test image
        prompt_name (str): Name of the prompt configuration to use
        server_url (str): Server URL
        phase (str): Operation phase - "initial", "verification", or "auto"
    """
    prompt = get_prompt(prompt_name)
    if not prompt:
        print(f"Error: Prompt configuration '{prompt_name}' not found.")
        print("Available configurations:")
        list_prompts()
        return
    
    print(f"Testing with '{prompt_name}' configuration...")
    
    # Add phase-specific instructions if specified
    user_message = None
    if phase == "initial":
        user_message = (
            "Analyze this surgical room image for INITIAL TASK PLANNING. "
            "Identify surgical tools on the white foam board that need to be moved to the tool box. "
            "Generate appropriate movement commands for the SOARM 101 robotic arm."
        )
        print("Phase: INITIAL ANALYSIS - Tool identification and task planning")
    elif phase == "verification":
        user_message = (
            "Analyze this surgical room image for COMPLETION VERIFICATION. "
            "Verify that surgical tools have been properly stored in the metal tool box "
            "and confirm the white foam board is clear of surgical instruments. "
            "Provide task completion status assessment."
        )
        print("Phase: COMPLETION VERIFICATION - Task verification and status assessment")
    else:
        print("Phase: AUTO-DETECTION - System will determine the appropriate phase")
    
    print("=" * 80)
    
    # Send request with custom system message and optional user message
    send_request(
        image_path=image_path,
        user_message=user_message,
        system_message=prompt,
        server_url=server_url
    )

def run_phase_comparison(image_path: str, prompt_name: str = "enhanced", 
                        server_url: str = "http://localhost:8000"):
    """
    Run comparison tests for both initial and verification phases.
    
    Args:
        image_path (str): Path to the test image
        prompt_name (str): Prompt configuration to use
        server_url (str): Server URL
    """
    print(f"Running phase comparison test with image: {image_path}")
    print("=" * 80)
    
    phases = [
        ("initial", "INITIAL ANALYSIS - Task Planning"),
        ("verification", "COMPLETION VERIFICATION - Status Assessment"),
        ("auto", "AUTO-DETECTION - System Decision")
    ]
    
    for phase_key, phase_desc in phases:
        print(f"\n🔬 TESTING PHASE: {phase_desc}")
        print("-" * 60)
        test_with_prompt_config(image_path, prompt_name, server_url, phase_key)
        print("\n" + "=" * 80)

def run_all_configs_test(image_path: str, phase: str = "auto", 
                        server_url: str = "http://localhost:8000"):
    """
    Run tests with all available prompt configurations for a specific phase.
    
    Args:
        image_path (str): Path to the test image
        phase (str): Operation phase to test
        server_url (str): Server URL
    """
    print(f"Running all configurations test with image: {image_path}")
    print(f"Phase: {phase.upper()}")
    print("=" * 80)
    
    for prompt_name in PROMPT_CONFIGS.keys():
        print(f"\n🔬 TESTING CONFIG: {prompt_name.upper()}")
        print("-" * 60)
        test_with_prompt_config(image_path, prompt_name, server_url, phase)
        print("\n" + "=" * 80)

def extract_plan_commands(response_text: str):
    """
    Extract plan commands and status from the LLM response.
    
    Args:
        response_text (str): The full LLM response
        
    Returns:
        list: List of extracted commands and status messages
    """
    commands = []
    lines = response_text.split('\n')
    in_plan_section = False
    
    for line in lines:
        line = line.strip()
        if line.startswith('<plan>'):
            in_plan_section = True
            continue
        elif line.startswith('</plan>'):
            in_plan_section = False
            break
        elif in_plan_section and line:
            # Look for command patterns and status messages
            command_patterns = [
                "Grip a tweezer", "Grip a straight scissor", "No surgical tools",
                "Task completed successfully", "Task partially completed", "Task verification failed"
            ]
            if any(pattern in line for pattern in command_patterns):
                commands.append(line.strip('- "'))
    
    return commands

def analyze_response_type(response_text: str):
    """
    Analyze the response to determine if it's initial planning or verification.
    
    Args:
        response_text (str): The full LLM response
        
    Returns:
        str: Response type - "initial_planning", "completion_verification", or "unknown"
    """
    commands = extract_plan_commands(response_text)
    
    initial_patterns = ["Grip a tweezer", "Grip a straight scissor", "No surgical tools detected"]
    verification_patterns = ["Task completed", "Task partially completed", "Task verification"]
    
    has_initial = any(any(pattern in cmd for pattern in initial_patterns) for cmd in commands)
    has_verification = any(any(pattern in cmd for pattern in verification_patterns) for cmd in commands)
    
    if has_verification:
        return "completion_verification"
    elif has_initial:
        return "initial_planning"
    else:
        return "unknown"

def main():
    parser = argparse.ArgumentParser(description="Test Surgical Scrub Nurse LLM System")
    
    # Test mode options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--image", "-i",
        help="Path to test image file"
    )
    group.add_argument(
        "--list-prompts", "-l",
        action="store_true",
        help="List available prompt configurations"
    )
    
    # Prompt configuration
    parser.add_argument(
        "--prompt", "-p",
        help="Prompt configuration to use (default: enhanced)",
        default="enhanced"
    )
    
    # Testing modes
    parser.add_argument(
        "--all-prompts", "-a",
        action="store_true",
        help="Test with all available prompt configurations"
    )
    
    parser.add_argument(
        "--phase-comparison", "-pc",
        action="store_true",
        help="Compare initial and verification phases with same image"
    )
    
    # Phase specification
    parser.add_argument(
        "--phase",
        choices=["initial", "verification", "auto"],
        default="auto",
        help="Specify operation phase (default: auto-detection)"
    )
    
    # Server configuration
    parser.add_argument(
        "--server", "-s",
        help="Server URL (default: http://localhost:8000)",
        default="http://localhost:8000"
    )
    
    args = parser.parse_args()
    
    if args.list_prompts:
        list_prompts()
        return
    
    if not os.path.exists(args.image):
        print(f"Error: Image file does not exist: {args.image}")
        sys.exit(1)
    
    # Execute appropriate test mode
    if args.phase_comparison:
        run_phase_comparison(args.image, args.prompt, args.server)
    elif args.all_prompts:
        run_all_configs_test(args.image, args.phase, args.server)
    else:
        test_with_prompt_config(args.image, args.prompt, args.server, args.phase)

if __name__ == "__main__":
    main() 