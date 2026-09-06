#!/usr/bin/env python3
"""
AASE Server - Transformers Only (Memory Efficient)

Uses only transformers for both activation extraction AND generation.
This saves GPU memory by not loading the model twice.

Usage:
    python aase_server_transformers.py --model meta-llama/Llama-3.1-8B-Instruct --probes-dir pretrained/ --port 8000
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger("aase-server")


@dataclass
class ProbeResult:
    probe_type: str
    score: float
    threshold: float
    is_flagged: bool


class AASEProbe:
    def __init__(self, name: str, direction: np.ndarray, threshold: float, layer: int):
        self.name = name
        self.direction = direction / np.linalg.norm(direction)
        self.threshold = threshold
        self.layer = layer
    
    def evaluate(self, activation: np.ndarray) -> ProbeResult:
        normalized = activation / np.linalg.norm(activation)
        score = float(np.dot(normalized, self.direction))
        return ProbeResult(
            probe_type=self.name,
            score=score,
            threshold=self.threshold,
            is_flagged=score > self.threshold,
        )
    
    @classmethod
    def load(cls, path: str, name: str) -> "AASEProbe":
        npy_path = path if path.endswith('.npy') else f"{path}.npy"
        json_path = npy_path.replace('.npy', '.json')
        
        direction = np.load(npy_path)
        threshold = 0.0
        layer = 13
        if os.path.exists(json_path):
            with open(json_path) as f:
                meta = json.load(f)
                threshold = meta.get("threshold", 0.0)
                layer = meta.get("layer_index", 13)
        
        return cls(name, direction, threshold, layer)


class TransformersModel:
    """Single model for both activation extraction and generation."""
    
    def __init__(self, model_name: str, device: str = "cuda"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info(f"Loading {model_name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        self.device = device
        self.num_layers = self.model.config.num_hidden_layers
        
        logger.info(f"Model loaded: {self.num_layers} layers")
    
    @torch.no_grad()
    def extract_activations(self, text: str, layers: List[int]) -> Dict[int, np.ndarray]:
        """Extract activations from specified layers."""
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=2048
        ).to(self.model.device)
        
        outputs = self.model(**inputs, output_hidden_states=True)
        last_pos = inputs.attention_mask.sum(dim=1) - 1
        
        result = {}
        for layer in layers:
            hidden = outputs.hidden_states[layer]
            result[layer] = hidden[0, last_pos[0], :].float().cpu().numpy()
        
        return result
    
    @torch.no_grad()
    def generate(self, prompt: str, max_tokens: int = 100, temperature: float = 0.7) -> str:
        """Generate text."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        
        generated = outputs[0][inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)


class SafetyGate:
    def __init__(self, model: TransformersModel, block_unsafe: bool = True):
        self.model = model
        self.probes: Dict[str, AASEProbe] = {}
        self.block_unsafe = block_unsafe
        self._layers_needed: set = set()
    
    def add_probe(self, probe: AASEProbe):
        self.probes[probe.name] = probe
        self._layers_needed.add(probe.layer)
        logger.info(f"Loaded: {probe.name} (layer={probe.layer}, threshold={probe.threshold:.4f})")
    
    def check(self, prompt: str) -> Dict:
        start = time.perf_counter()
        
        activations = self.model.extract_activations(prompt, list(self._layers_needed))
        
        results = {}
        flagged = []
        
        for name, probe in self.probes.items():
            act = activations[probe.layer]
            result = probe.evaluate(act)
            results[name] = asdict(result)
            if result.is_flagged:
                flagged.append(name)
        
        latency = (time.perf_counter() - start) * 1000
        
        if flagged:
            logger.warning(f"BLOCKED: {flagged} - '{prompt[:50]}...'")
        
        return {
            "is_safe": len(flagged) == 0,
            "flagged_by": flagged,
            "results": results,
            "latency_ms": latency,
        }


def create_app(model: TransformersModel, safety_gate: SafetyGate):
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    from typing import List, Union
    
    app = FastAPI(title="AASE Safety Server (Transformers)")
    
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
        return {"status": "ok", "probes": list(safety_gate.probes.keys())}
    
    @app.get("/v1/safety/probes")
    def list_probes():
        return {
            name: {"layer": p.layer, "threshold": p.threshold}
            for name, p in safety_gate.probes.items()
        }
    
    @app.post("/v1/safety/check")
    def safety_check(req: CompletionRequest):
        prompt = req.prompt if isinstance(req.prompt, str) else req.prompt[0]
        return safety_gate.check(prompt)
    
    @app.post("/v1/completions")
    def completions(req: CompletionRequest):
        prompt = req.prompt if isinstance(req.prompt, str) else req.prompt[0]
        
        safety = safety_gate.check(prompt)
        
        if not safety["is_safe"] and safety_gate.block_unsafe:
            return JSONResponse(status_code=400, content={
                "error": "blocked_by_safety",
                "flagged_by": safety["flagged_by"],
                "scores": safety["results"],
            })
        
        text = model.generate(prompt, req.max_tokens, req.temperature)
        
        return {
            "id": f"cmpl-{int(time.time())}",
            "model": req.model,
            "choices": [{"text": text, "index": 0, "finish_reason": "stop"}],
            "safety": safety,
        }
    
    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatRequest):
        prompt = "\n".join([f"{m.role}: {m.content}" for m in req.messages])
        prompt += "\nassistant:"
        
        safety = safety_gate.check(prompt)
        
        if not safety["is_safe"] and safety_gate.block_unsafe:
            return JSONResponse(status_code=400, content={
                "error": "blocked_by_safety",
                "flagged_by": safety["flagged_by"],
            })
        
        text = model.generate(prompt, req.max_tokens, req.temperature)
        
        return {
            "id": f"chat-{int(time.time())}",
            "model": req.model,
            "choices": [{
                "message": {"role": "assistant", "content": text},
                "index": 0,
                "finish_reason": "stop",
            }],
            "safety": safety,
        }
    
    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-2-2b-it")
    parser.add_argument("--probes-dir", default="pretrained")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-block", action="store_true")
    parser.add_argument("--enable-af", action="store_true", default=True)
    parser.add_argument("--enable-aag", action="store_true", default=True)
    parser.add_argument("--enable-medical", action="store_true", default=True)
    parser.add_argument("--enable-legal", action="store_true", default=False)
    parser.add_argument("--enable-financial", action="store_true", default=False)
    parser.add_argument("--enable-crisis", action="store_true", default=True)
    parser.add_argument("--enable-all-apc", action="store_true")
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info(" AASE Safety Server (Transformers Only)")
    logger.info("=" * 60)
    
    # Load single model
    model = TransformersModel(args.model)
    
    # Create safety gate
    gate = SafetyGate(model, block_unsafe=not args.no_block)
    
    # Load probes
    model_safe = args.model.replace("/", "_").replace("-", "_")
    probes_dir = Path(args.probes_dir)
    
    if args.enable_af:
        af_path = probes_dir / "af" / model_safe
        if (probes_dir / "af" / f"{model_safe}.npy").exists():
            gate.add_probe(AASEProbe.load(str(af_path), "af"))
    
    if args.enable_aag:
        aag_path = probes_dir / "aag" / model_safe
        if (probes_dir / "aag" / f"{model_safe}.npy").exists():
            gate.add_probe(AASEProbe.load(str(aag_path), "aag"))
    
    apc_policies = []
    if args.enable_all_apc:
        apc_policies = ["medical", "legal", "financial", "crisis"]
    else:
        if args.enable_medical:
            apc_policies.append("medical")
        if args.enable_legal:
            apc_policies.append("legal")
        if args.enable_financial:
            apc_policies.append("financial")
        if args.enable_crisis:
            apc_policies.append("crisis")
    
    for policy in apc_policies:
        apc_path = probes_dir / "apc" / f"{model_safe}_{policy}"
        if (probes_dir / "apc" / f"{model_safe}_{policy}.npy").exists():
            gate.add_probe(AASEProbe.load(str(apc_path), f"apc_{policy}"))
    
    if not gate.probes:
        logger.error("No probes loaded! Check probes directory.")
        sys.exit(1)
    
    # Create and run
    app = create_app(model, gate)
    
    import uvicorn
    logger.info(f"Starting on port {args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
