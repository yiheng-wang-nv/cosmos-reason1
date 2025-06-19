from fastapi import FastAPI
from pydantic import BaseModel
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
import uvicorn
from PIL import Image
import os
import base64
import io
import re
from typing import Optional, List

def main():
    MODEL_PATH = "nvidia/Cosmos-Reason1-7B"
    
    # Initialize model and processor
    print("Loading model and processor...")
    llm = LLM(model=MODEL_PATH, max_model_len=8192)
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    
    app = FastAPI()
    
    # Load default system message from file
    def load_default_system_message():
        filename = "system_message.txt"
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, filename)
        
        with open(file_path, "r") as f:
            content = f.read()
            print(f"Loaded system message from: {file_path}")
            return content

    # Load default user message from file
    def load_default_user_message():
        filename = "user_message.txt"
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, filename)
        
        with open(file_path, "r") as f:
            content = f.read()
            print(f"Loaded user message from: {file_path}")
            return content
    
    # Extract and analyze surgical commands (simplified for straight scissors + tweezers only)
    def extract_surgical_commands(response_text: str) -> dict:
        """Extract and analyze surgical commands from response."""
        result = {
            "commands": [],
            "tool_count": 0,
            "has_scissors": False,
            "has_tweezers": False
        }
        
        # Extract plan section
        plan_match = re.search(r'<plan>(.*?)</plan>', response_text, re.DOTALL | re.IGNORECASE)
        if not plan_match:
            return result
        
        plan_text = plan_match.group(1).strip()
        lines = [line.strip() for line in plan_text.split('\n') if line.strip()]
        
        for line in lines:
            line_lower = line.lower()
            if "grip a straight scissor and put it in the box" in line_lower:
                result["commands"].append(line)
                result["has_scissors"] = True
                result["tool_count"] += 1
            elif "grip a tweezer and put it in the box" in line_lower:
                result["commands"].append(line)
                result["has_tweezers"] = True
                result["tool_count"] += 1
            elif "no actions needed" in line_lower:
                result["commands"].append(line)
        
        return result
    
    class PromptRequest(BaseModel):
        image_path: Optional[str] = None
        image_base64: Optional[str] = None
        user_message: Optional[str] = None
        system_message: Optional[str] = None
        max_tokens: Optional[int] = 1024
        temperature: Optional[float] = 0.0

    @app.post("/generate")
    async def generate(request: PromptRequest):
        """Generate surgical tool analysis and robot commands."""
        
        # Validate image input
        if not request.image_path and not request.image_base64:
            return {"error": "Either image_path or image_base64 must be provided"}
        
        if request.image_path and request.image_base64:
            return {"error": "Provide either image_path OR image_base64, not both"}
        
        # Validate parameters
        max_tokens = min(max(request.max_tokens, 128), 8192)
        temperature = min(max(request.temperature, 0.0), 2.0)
        
        try:
            # Load image
            if request.image_path:
                if not os.path.exists(request.image_path):
                    return {"error": f"Image file does not exist: {request.image_path}"}
                image = Image.open(request.image_path)
            else:
                try:
                    base64_data = request.image_base64
                    if base64_data.startswith('data:'):
                        base64_data = base64_data.split(',', 1)[1]
                    
                    image_bytes = base64.b64decode(base64_data)
                    image = Image.open(io.BytesIO(image_bytes))
                except Exception as e:
                    return {"error": f"Failed to decode base64 image: {str(e)}"}
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Load messages
            system_text = request.system_message if request.system_message else load_default_system_message()
            user_text = request.user_message if request.user_message else load_default_user_message()
            
            # Prepare messages
            messages = [
                {"role": "system", "content": system_text},
                {"role": "user", "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image", "image": image}
                ]},
            ]

            # Generate response
            sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            llm_inputs = {"prompt": prompt, "multi_modal_data": {"image": image}}
            outputs = llm.generate([llm_inputs], sampling_params)
            
            response_text = outputs[0].outputs[0].text
            
            # Analyze commands
            command_analysis = extract_surgical_commands(response_text)
            
            return {
                "response": response_text,
                "analysis": {
                    "surgical_tools_detected": command_analysis["tool_count"],
                    "scissors_detected": command_analysis["has_scissors"],
                    "tweezers_detected": command_analysis["has_tweezers"],
                    "robot_commands": command_analysis["commands"]
                },
                "config": {
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "actual_tokens": len(outputs[0].outputs[0].token_ids) if hasattr(outputs[0].outputs[0], 'token_ids') else "unknown"
                }
            }
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}

    @app.get("/health")
    async def health_check():
        """Check server health status."""
        return {"status": "healthy", "model": MODEL_PATH}
    
    # Start the server
    print("=" * 60)
    print("🏥 SURGICAL ENVIRONMENT ANALYSIS SERVER")
    print("=" * 60)
    print("Ready to analyze surgical environments and generate robot commands!")
    print("")
    print("📋 Supported Tools:")
    print("- Metal surgical scissors (straight blades)")
    print("- Metal surgical tweezers") 
    print("")
    print("🎯 Environment Components:")
    print("- White foam board = surgical tray")
    print("- Metal box = surgical tool box")
    print("- White robot arm = robot scrub nurse (SOARM 101)")
    print("")
    print("🚀 Server starting on http://0.0.0.0:8000")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main() 