from fastapi import FastAPI
from pydantic import BaseModel
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
import uvicorn
from PIL import Image
import os
import base64
import io
from typing import Optional

def main():
    MODEL_PATH = "nvidia/Cosmos-Reason1-7B"
    
    # Initialize model and processor
    print("Loading model and processor...")
    llm = LLM(model=MODEL_PATH, max_model_len=8192)
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    
    app = FastAPI()
    
    # Load default system message from file
    def load_default_system_message():
        filename = "system_message_analyze.txt"
        fallback = (
            "You are analyzing a surgical environment for a scrub nurse task.\n\n"
            "Task: Clean the surgical tray (white foam board) by moving surgical tools to the tool box (metal box). Leave non-surgical items unchanged.\n\n"
            "Output format:\n"
            "<reasoning>\n"
            "This task simulates a surgical scrub nurse cleaning the tray by organizing surgical tools. I see [describe the metal scissors and tweezers you observe in the image].\n"
            "</reasoning>\n\n"
            "<plan>\n"
            "Grip a straight scissor and put it in the box.\n"
            "Grip a tweezer and put it in the box.\n"
            "</plan>\n\n"
            "CRITICAL: Use EXACTLY these commands in plan:\n"
            "- \"Grip a straight scissor and put it in the box.\" (for scissors)\n"
            "- \"Grip a tweezer and put it in the box.\" (for tweezers)\n"
            "- If no surgical tools found: \"No actions needed.\""
        )
        
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(script_dir, filename)
            with open(file_path, "r") as f:
                content = f.read()
                print(f"Loaded {filename} from: {file_path}")
                return content
        except FileNotFoundError:
            print(f"File not found: {filename}, using fallback")
            return fallback

    # Load default user message from file
    def load_default_user_message():
        filename = "user_message_analyze.txt"
        fallback = (
            "What metal surgical tools (scissors or tweezers) do you see in this image?\n\n"
            "Use exact robot commands in plan section:\n"
            "\"Grip a straight scissor and put it in the box.\"\n"
            "\"Grip a tweezer and put it in the box.\"\n\n"
            "Use XML format."
        )
        
        try:
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
        image_path: Optional[str] = None  # For local file path (server-side files)
        image_base64: Optional[str] = None  # For base64-encoded image data (remote clients)
        user_message: Optional[str] = None
        system_message: Optional[str] = None
        max_tokens: Optional[int] = 1024  # Default max_tokens, can be overridden by client
        temperature: Optional[float] = 0.0  # Default temperature for deterministic responses

    @app.post("/generate")
    async def generate(request: PromptRequest):
        """
        Generate surgical tool analysis and task planning response.
        
        Args:
            request: PromptRequest containing:
                - image_path: Path to surgical environment image (for server-side files)
                - image_base64: Base64-encoded image data (for remote clients)
                - user_message/system_message: Optional custom messages (overrides defaults)
                - max_tokens/temperature: Sampling parameters
            
        Returns:
            dict: Analysis response with structured reasoning/plan format
        """
        # Validate that either image_path or image_base64 is provided
        if not request.image_path and not request.image_base64:
            return {"error": "Either image_path or image_base64 must be provided"}
        
        if request.image_path and request.image_base64:
            return {"error": "Provide either image_path OR image_base64, not both"}
        
        # Validate max_tokens range (reasonable limits)
        max_tokens = min(max(request.max_tokens, 128), 8192)  # Clamp between 128 and 8192
        temperature = min(max(request.temperature, 0.0), 2.0)  # Clamp between 0.0 and 2.0
        
        try:
            # Load image from either file path or base64 data
            if request.image_path:
                # Server-side file path
                if not os.path.exists(request.image_path):
                    return {"error": f"Image file does not exist: {request.image_path}"}
                image = Image.open(request.image_path)
            else:
                # Base64-encoded image from remote client
                try:
                    # Remove data URL prefix if present (e.g., "data:image/jpeg;base64,")
                    base64_data = request.image_base64
                    if base64_data.startswith('data:'):
                        base64_data = base64_data.split(',', 1)[1]
                    
                    # Decode base64 to bytes
                    image_bytes = base64.b64decode(base64_data)
                    # Create PIL Image from bytes
                    image = Image.open(io.BytesIO(image_bytes))
                except Exception as e:
                    return {"error": f"Failed to decode base64 image: {str(e)}"}
            
            # Convert to RGB if necessary (some formats like RGBA or P mode can cause issues)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Use custom messages if provided, otherwise use defaults
            system_text = request.system_message if request.system_message else load_default_system_message()
            user_text = request.user_message if request.user_message else load_default_user_message()
            
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
    print("Starting Surgical Environment Analysis Server...")
    print("Ready to analyze surgical environment images and generate robot commands!")
    print("Features:")
    print("- Analyzes surgical environments for scrub nurse tasks")
    print("- Generates precise robot commands for tool organization")
    print("- Supports both local file paths and base64 image encoding")
    print("- Auto-loads system_message_analyze.txt and user_message_analyze.txt")
    print("- Supports custom message overrides via API parameters")
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main() 