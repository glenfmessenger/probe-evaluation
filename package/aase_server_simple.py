#!/usr/bin/env python3
"""
AASE Server - Simple Version for vLLM v1

This version avoids the apply_model() pickling issues in vLLM v1 by using
a separate activation extraction pass before the main generation.

Usage:
    VLLM_ALLOW_INSECURE_SERIALIZATION=1 python aase_server_simple.py \
        --model google/gemma-2-2b-it \
        --probes-dir pretrained/ \
        --port 8000
"""

import os
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

import sys
import argparse
import json
import time
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("aase-server")


# ============================================================================
# AASE PROBES
# ============================================================================

@dataclass
class ProbeResult:
    probe_type: str
    score: float
    threshold: float
    is_flagged: bool


class AASEProbe:
    def __init__(self, probe_type: str, direction: np.ndarray, threshold: float):
        self.probe_type = probe_type
        self.direction = direction / np.linalg.norm(direction)
        self.threshold = threshold
    
    def evaluate(self, activation: np.ndarray) -> ProbeResult:
        normalized = activation / np.linalg.norm(activation)
        score = float(np.dot(normalized, self.direction))
        return ProbeResult(
            probe_type=self.probe_type,
            score=score,
            threshold=self.threshold,
            is_flagged=score > self.threshold,
        )
    
    @classmethod
    def load(cls, path: str, probe_type: str) -> "AASEProbe":
        npy_path = path if path.endswith('.npy') else f"{path}.npy"
        json_path = npy_path.replace('.npy', '.json')
        
        direction = np.load(npy_path)
        threshold = 0.0
        if os.path.exists(json_path):
            with open(json_path) as f:
                meta = json.load(f)
                threshold = meta.get("threshold", 0.0)
        
        return cls(probe_type, direction, threshold)


# ============================================================================
# ACTIVATION EXTRACTOR (uses transformers directly, not vLLM hooks)
# ============================================================================

class ActivationExtractor:
    """
    Extract activations using transformers directly.
    This avoids vLLM's apply_model() pickling issues.
    """
    
    def __init__(self, model_name: str, layer_index: int, device: str = "cuda"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info(f"Loading tokenizer for activation extraction...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"Loading model for activation extraction...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True,
            output_hidden_states=True,
        )
        self.model.eval()
        
        self.layer_index = layer_index
        self.device = device
        logger.info(f"Activation extractor ready (layer {layer_index})")
    
    @torch.no_grad()
    def extract(self, text: str) -> np.ndarray:
        """Extract activation for a single prompt."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(self.device)
        
        outputs = self.model(**inputs, output_hidden_states=True)
        
        # Get hidden state at target layer, last token position
        hidden = outputs.hidden_states[self.layer_index]
        last_pos = inputs.attention_mask.sum(dim=1) - 1
        activation = hidden[0, last_pos[0], :].float().cpu().numpy()
        
        return activation


# ============================================================================
# SAFETY GATE
# ============================================================================

class SafetyGate:
    def __init__(self, extractor: ActivationExtractor, block_unsafe: bool = True):
        self.extractor = extractor
        self.probes: Dict[str, AASEProbe] = {}
        self.block_unsafe = block_unsafe
    
    def add_probe(self, probe: AASEProbe):
        self.probes[probe.probe_type] = probe
        logger.info(f"Loaded probe: {probe.probe_type} (threshold={probe.threshold:.4f})")
    
    def check(self, prompt: str) -> Dict:
        """Check if prompt is safe."""
        start = time.perf_counter()
        
        activation = self.extractor.extract(prompt)
        
        results = {}
        flagged = []
        
        for name, probe in self.probes.items():
            result = probe.evaluate(activation)
            results[name] = asdict(result)
            if result.is_flagged:
                flagged.append(name)
        
        latency = (time.perf_counter() - start) * 1000
        
        is_safe = len(flagged) == 0
        
        if not is_safe:
            logger.warning(f"UNSAFE: {flagged} - '{prompt[:50]}...'")
        
        return {
            "is_safe": is_safe,
            "flagged_by": flagged,
            "results": results,
            "latency_ms": latency,
        }


# ============================================================================
# SERVER
# ============================================================================

def create_app(safety_gate: SafetyGate, vllm_model: str):
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    from typing import List, Union
    from vllm import LLM, SamplingParams
    
    app = FastAPI(title="AASE Safety Server")
    
    # Load vLLM for generation (separate from activation extraction)
    logger.info("Loading vLLM for generation...")
    llm = LLM(
        model=vllm_model,
        trust_remote_code=True,
        gpu_memory_utilization=0.4,  # Share GPU with transformers model
        max_model_len=2048,
    )
    logger.info("vLLM ready")
    
    class CompletionRequest(BaseModel):
        model: str
        prompt: Union[str, List[str]]
        max_tokens: int = 100
        temperature: float = 0.7
    
    class ChatMessage(BaseModel):
        role: str
        content: str
    
    class ChatRequest(BaseModel):
        model: str
        messages: List[ChatMessage]
        max_tokens: int = 100
        temperature: float = 0.7
    
    @app.get("/health")
    def health():
        return {"status": "ok", "aase_enabled": True}
    
    @app.get("/v1/safety/status")
    def safety_status():
        return {
            "probes": list(safety_gate.probes.keys()),
            "block_unsafe": safety_gate.block_unsafe,
        }
    
    @app.post("/v1/completions")
    def completions(req: CompletionRequest):
        prompt = req.prompt if isinstance(req.prompt, str) else req.prompt[0]
        
        # Safety check
        safety = safety_gate.check(prompt)
        
        if not safety["is_safe"] and safety_gate.block_unsafe:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "blocked_by_safety",
                    "flagged_by": safety["flagged_by"],
                    "scores": safety["results"],
                }
            )
        
        # Generate
        params = SamplingParams(max_tokens=req.max_tokens, temperature=req.temperature)
        outputs = llm.generate([prompt], params)
        
        return {
            "id": f"cmpl-{int(time.time())}",
            "model": req.model,
            "choices": [{
                "text": outputs[0].outputs[0].text,
                "index": 0,
                "finish_reason": "stop",
            }],
            "safety": safety,
        }
    
    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatRequest):
        # Format as prompt
        prompt = "\n".join([f"{m.role}: {m.content}" for m in req.messages])
        prompt += "\nassistant:"
        
        # Safety check
        safety = safety_gate.check(prompt)
        
        if not safety["is_safe"] and safety_gate.block_unsafe:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "blocked_by_safety",
                    "flagged_by": safety["flagged_by"],
                }
            )
        
        # Generate
        params = SamplingParams(max_tokens=req.max_tokens, temperature=req.temperature)
        outputs = llm.generate([prompt], params)
        
        return {
            "id": f"chat-{int(time.time())}",
            "model": req.model,
            "choices": [{
                "message": {"role": "assistant", "content": outputs[0].outputs[0].text},
                "index": 0,
                "finish_reason": "stop",
            }],
            "safety": safety,
        }
    
    @app.post("/v1/safety/check")
    def safety_check(req: CompletionRequest):
        """Check safety without generating."""
        prompt = req.prompt if isinstance(req.prompt, str) else req.prompt[0]
        return safety_gate.check(prompt)
    
    return app


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-2-2b-it")
    parser.add_argument("--probes-dir", default="pretrained/")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--layer", type=int, default=13)
    parser.add_argument("--no-block", action="store_true")
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info(" AASE Safety Server")
    logger.info("=" * 60)
    
    # Create extractor
    extractor = ActivationExtractor(args.model, args.layer)
    
    # Create safety gate
    gate = SafetyGate(extractor, block_unsafe=not args.no_block)
    
    # Load probes
    model_safe = args.model.replace("/", "_").replace("-", "_")
    probes_dir = Path(args.probes_dir)
    
    probe_files = [
        ("af", probes_dir / "af" / model_safe),
        ("aag", probes_dir / "aag" / model_safe),
        ("apc", probes_dir / "apc" / f"{model_safe}_medical_advice"),
    ]
    
    for probe_type, path in probe_files:
        npy_path = f"{path}.npy"
        if os.path.exists(npy_path):
            gate.add_probe(AASEProbe.load(str(path), probe_type))
        else:
            logger.warning(f"Probe not found: {npy_path}")
    
    # Create and run app
    app = create_app(gate, args.model)
    
    import uvicorn
    logger.info(f"Starting server on port {args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
