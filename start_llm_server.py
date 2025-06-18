from fastapi import FastAPI
from pydantic import BaseModel
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
import uvicorn
from PIL import Image
import os
from typing import Optional

MODEL_PATH = "nvidia/Cosmos-Reason1-7B"

def main():
    """Main function that initializes everything"""
    print(f"Initializing model: {MODEL_PATH}")
    
    # Initialize model and processor in main
    llm = LLM(
        model=MODEL_PATH,
        limit_mm_per_prompt={"image": 1, "video": 0},
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    
    print("Model loaded successfully!")
    
    # Create FastAPI app
    app = FastAPI()
    
    # Load default system message from file based on phase
    def load_default_system_message(phase="initial"):
        if phase == "verification":
            filename = "system_message_verification.txt"
            fallback = (
                "Surgical scrub nurse assistant using SOARM 101 robotic arm. Analyze surgical environment IMAGE. "
                "Verify tray cleaning completion after robot operations.\n\n"
                "VERIFICATION TASK: Check if SOARM 101 successfully cleaned the surgical tray by moving surgical tools to proper storage.\n\n"
                "ANALYSIS AREAS:\n"
                "1. Surgical tray (white foam board): Should be CLEAN of metal surgical tools\n"
                "2. Surgical tool box (metal box): Should CONTAIN the organized surgical tools\n\n"
                "SURGICAL TOOLS TO VERIFY:\n"
                "- Metal surgical scissors → Should be IN the surgical tool box\n"
                "- Metal tweezers → Should be IN the surgical tool box\n"
                "- IGNORE: Plastic scissors, watches, office items (these can remain on tray)\n\n"
                "FORMAT (STRICT):\n"
                "<reasoning>\n"
                "Verify surgical tray cleaning: check if tray is clean of surgical tools, confirm tools are properly stored in surgical tool box, assess SOARM 101 task completion.\n"
                "</reasoning>\n\n"
                "<plan>\n"
                "Tray cleaning completed successfully. All surgical tools organized in surgical tool box.\n"
                "Surgical tools still remain on tray: [list any remaining tools]\n"
                "Surgical tools not properly stored in box: [list any missing tools]\n"
                "</plan>\n\n"
                "RULES:\n"
                "- REASONING: Describe tray cleaning verification, SOARM 101 performance assessment\n"
                "- PLAN: Status message based on surgical tray cleaning results\n"
                "- ONLY verify surgical tools organization (ignore office items)\n"
                "- BE CONSISTENT: Same input = same output"
            )
        else:  # phase == "initial" or default
            filename = "system_message_analyze.txt"
            fallback = (
                "Surgical scrub nurse assistant using SOARM 101 robotic arm. Analyze surgical environment IMAGE. "
                "Task: Clean the surgical tray by moving surgical tools to proper storage.\n\n"
                "ENVIRONMENT:\n"
                "- White foam board = Surgical tray (needs to be cleaned)\n"
                "- Metal box = Surgical tool box (destination for surgical tools)\n"
                "- SOARM 101 = Robotic scrub nurse arm\n\n"
                "SURGICAL TOOLS: Metal scissors, metal tweezers only\n"
                "IGNORE: Plastic scissors, watches, office items (leave unchanged)\n\n"
                "FORMAT (STRICT):\n"
                "<reasoning>\n"
                "Analyze surgical tray, identify metal surgical tools that need cleaning/organizing, determine SOARM 101 actions to transfer tools to surgical tool box.\n"
                "</reasoning>\n\n"
                "<plan>\n"
                "Grip a straight scissor and put it in the box.\n"
                "Grip a tweezer and put it in the box.\n"
                "</plan>\n\n"
                "RULES:\n"
                "- REASONING: Describe surgical tray cleaning task, tools found, SOARM 101 actions needed\n"
                "- PLAN: Only list grip commands for surgical tools found ON tray\n"
                "- Each command on separate line (no \\n characters)\n"
                "- Focus on tray cleaning: surgical tools → surgical tool box\n"
                "- BE CONSISTENT: Same input = same output"
            )
        
        try:
            # Get the directory where this script is located
            script_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(script_dir, filename)
            with open(file_path, "r") as f:
                content = f.read()
                print(f"Loaded {filename} from: {file_path}")
                return content
        except FileNotFoundError:
            print(f"File not found: {filename}, using fallback")
            return fallback

    # Load default user message from file based on phase
    def load_default_user_message(phase="initial"):
        if phase == "verification":
            filename = "user_message_verification.txt"
            fallback = (
                "Verify SOARM 101 surgical tray cleaning completion. Check if surgical tools properly organized from tray to surgical tool box.\n\n"
                "IGNORE: plastic scissors, watches, office items (these can remain on tray).\n\n"
                "REQUIRED:\n"
                "- Reasoning: Describe tray cleaning verification, SOARM 101 performance assessment\n"
                "- Plan: Status message indicating tray cleaning completion or remaining tasks\n\n"
                "Status messages:\n"
                "\"Tray cleaning completed successfully. All surgical tools organized in surgical tool box.\"\n"
                "\"Surgical tools still remain on tray: [specific tools]\"\n"
                "\"Surgical tools not properly stored in box: [specific tools]\"\n\n"
                "FOCUS: Only verify surgical tools organization (scissors, tweezers).\n\n"
                "BE EXTREMELY BRIEF."
            )
        else:  # phase == "initial" or default
            filename = "user_message_analyze.txt"
            fallback = (
                "Analyze surgical environment IMAGE. SOARM 101 scrub nurse task: Clean the surgical tray by organizing surgical tools.\n\n"
                "TASK: Move ONLY metal surgical tools from surgical tray (white foam board) to surgical tool box (metal box).\n"
                "IGNORE: plastic scissors, watches, office items (leave on tray).\n\n"
                "REQUIRED:\n"
                "- Reasoning: Describe surgical tray cleaning task, tools found, SOARM 101 actions needed\n"
                "- Plan: ONLY grip commands for surgical tools found on tray, each on separate line\n\n"
                "Commands (only for surgical tools ON tray):\n"
                "\"Grip a straight scissor and put it in the box.\"\n"
                "\"Grip a tweezer and put it in the box.\"\n\n"
                "BE EXTREMELY BRIEF."
            )
        
        try:
            # Get the directory where this script is located
            script_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(script_dir, filename)
            with open(file_path, "r") as f:
                content = f.read()
                print(f"Loaded {filename} from: {file_path}")
                return content
        except FileNotFoundError:
            print(f"File not found: {filename}, using fallback")
            return fallback
    
    class PromptRequest(BaseModel):
        image_path: str
        user_message: Optional[str] = None
        system_message: Optional[str] = None
        phase: Optional[str] = "initial"  # "initial" or "verification" - determines default messages
        max_tokens: Optional[int] = 1024  # Default max_tokens, can be overridden by client
        temperature: Optional[float] = 0.0  # Default temperature for deterministic responses

    @app.post("/generate")
    async def generate(request: PromptRequest):
        """
        Generate surgical tool analysis and task planning/verification response.
        
        Args:
            request: PromptRequest containing:
                - image_path: Path to surgical environment image
                - phase: "initial" (tool identification/planning) or "verification" (completion check)
                - user_message/system_message: Optional custom messages (overrides phase defaults)
                - max_tokens/temperature: Sampling parameters
            
        Returns:
            dict: Analysis response with structured reasoning/plan format
        """
        # Validate image file exists
        if not os.path.exists(request.image_path):
            return {"error": f"Image file does not exist: {request.image_path}"}
        
        # Validate max_tokens range (reasonable limits)
        max_tokens = min(max(request.max_tokens, 128), 8192)  # Clamp between 128 and 8192
        temperature = min(max(request.temperature, 0.0), 2.0)  # Clamp between 0.0 and 2.0
        
        try:
            image = Image.open(request.image_path)
            
            # Use custom messages if provided, otherwise use phase-appropriate defaults
            system_text = request.system_message if request.system_message else load_default_system_message(request.phase)
            user_text = request.user_message if request.user_message else load_default_user_message(request.phase)
            
            messages = [
                {"role": "system", "content": system_text},
                {"role": "user", "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image", "image": image}
                ]},
            ]

            # Create sampling parameters for this specific request
            sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
            
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            llm_inputs = {"prompt": prompt, "multi_modal_data": {"image": image}}
            outputs = llm.generate([llm_inputs], sampling_params)
            
            return {
                "response": outputs[0].outputs[0].text,
                "config": {
                    "phase": request.phase,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "actual_tokens": len(outputs[0].outputs[0].token_ids) if hasattr(outputs[0].outputs[0], 'token_ids') else "unknown"
                }
            }
        except Exception as e:
            return {"error": f"Generation failed: {str(e)}"}

    @app.get("/health")
    async def health_check():
        """Check server health status."""
        return {"status": "healthy", "model": MODEL_PATH}
    
    # Start the server
    print("Starting Surgical Scrub Nurse LLM Server...")
    print("Ready to analyze surgical environment images and generate SOARM 101 commands!")
    print("Features:")
    print("- Phase-aware defaults: 'initial' (analysis) or 'verification' (completion check)")
    print("- Auto-loads appropriate message files per phase:")
    print("  * initial: system_message_analyze.txt + user_message_analyze.txt")
    print("  * verification: system_message_verification.txt + user_message_verification.txt")
    print("- Supports custom message overrides via API parameters")
    print("- Dynamic sampling: max_tokens and temperature adjustable per request")
    print("- Focus: Surgical tray cleaning and tool organization")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)

if __name__ == "__main__":
    # HF_HUB_OFFLINE=1 python start_llm_server.py
    main() 