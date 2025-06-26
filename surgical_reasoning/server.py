"""
Surgical Analysis Server - Single image version for suturing workflow analysis
"""

import os
from typing import Optional
from datetime import datetime

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info


class AnalysisRequest(BaseModel):
    image_path: str
    system_message: Optional[str] = None
    user_message: Optional[str] = None
    max_tokens: Optional[int] = 4096
    temperature: Optional[float] = 0.0


class AnalysisResponse(BaseModel):
    success: bool
    analysis: Optional[str] = None
    error: Optional[str] = None
    timestamp: str
    config: dict


def load_default_system_message():
    """Load default system message from file"""
    filename = "system_message.txt"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, filename)
    
    with open(file_path, "r") as f:
        content = f.read()
        print(f"Loaded system message from: {file_path}")
        return content


def load_default_user_message():
    """Load default user message from file"""
    filename = "user_message.txt"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, filename)
    
    with open(file_path, "r") as f:
        content = f.read()
        print(f"Loaded user message from: {file_path}")
        return content


class SurgicalAnalyzer:
    """Core surgical analyzer using Cosmos Reason1"""
    
    def __init__(self, model_path: str = "nvidia/Cosmos-Reason1-7B"):
        print(f"Loading Cosmos Reason1: {model_path}")
        
        self.llm = LLM(
            model=model_path,
            limit_mm_per_prompt={"image": 10},
            gpu_memory_utilization=0.8,
            max_model_len=8192
        )
        
        self.processor = AutoProcessor.from_pretrained(model_path)
        print("✅ Analyzer ready")

    def analyze(self, 
                image_path: str, 
                system_message: Optional[str] = None,
                user_message: Optional[str] = None,
                max_tokens: int = 4096,
                temperature: float = 0.0) -> dict:
        """Analyze single surgical image"""
        
        # Verify image exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Use default messages if not provided
        system_text = system_message if system_message else load_default_system_message()
        user_text = user_message if user_message else load_default_user_message()
        
        # Build message with single image
        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": [
                {"type": "text", "text": user_text},
                {"type": "image", "image": image_path}
            ]}
        ]
        
        # Generate response
        try:
            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=1.0,
                max_tokens=max_tokens
            )
            
            prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, _, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
            
            llm_inputs = {
                "prompt": prompt,
                "multi_modal_data": {"image": image_inputs},
                "mm_processor_kwargs": video_kwargs,
            }
            
            outputs = self.llm.generate([llm_inputs], sampling_params)
            analysis = outputs[0].outputs[0].text
            
            return {
                "success": True,
                "analysis": analysis,
                "timestamp": datetime.now().isoformat(),
                "config": {
                    "image_path": image_path,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "actual_tokens": len(outputs[0].outputs[0].token_ids) if hasattr(outputs[0].outputs[0], 'token_ids') else "unknown"
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "config": {
                    "image_path": image_path,
                    "max_tokens": max_tokens,
                    "temperature": temperature
                }
            }


# FastAPI Server
app = FastAPI(title="Surgical Analysis API", version="1.0.0")
analyzer = None


@app.on_event("startup")
async def startup():
    global analyzer
    print("🏥 Starting Surgical Analysis System...")
    analyzer = SurgicalAnalyzer()


@app.get("/")
async def root():
    return {"message": "Surgical Analysis API Ready"}


@app.get("/health")
async def health_check():
    """Check server health status"""
    if analyzer is None:
        raise HTTPException(status_code=503, detail="Analyzer not ready")
    return {"status": "healthy", "model": "nvidia/Cosmos-Reason1-7B"}


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_image(request: AnalysisRequest):
    """Analyze single surgical image from file path"""
    if analyzer is None:
        raise HTTPException(status_code=503, detail="Analyzer not ready")
    
    try:
        result = analyzer.analyze(
            request.image_path,
            request.system_message,
            request.user_message,
            request.max_tokens or 4096,
            request.temperature if request.temperature is not None else 0.0
        )
        return AnalysisResponse(**result)
    except Exception as e:
        return AnalysisResponse(
            success=False,
            error=str(e),
            timestamp=datetime.now().isoformat(),
            config={"error": True}
        )


def main():
    """Start the server"""
    print("=" * 60)
    print("🏥 SURGICAL SUTURING ANALYSIS SERVER")
    print("=" * 60)
    print("Ready to analyze surgical suturing workflow!")
    print("")
    print("📸 Image Support: Single image per analysis")
    print("🎯 Analysis Focus: Suturing workflow step recognition")
    print("🔒 Output: Deterministic (temperature=0.0)")
    print("")
    print("🚀 Server starting on http://0.0.0.0:8000")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main() 