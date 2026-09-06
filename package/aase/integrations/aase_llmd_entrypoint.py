#!/usr/bin/env python3
"""
AASE-Enabled llm-d Entrypoint

This script serves as a drop-in entrypoint for llm-d that adds AASE safety
checking via activation hooks. No proxy needed - hooks are registered at
startup and safety checking happens inline.

Usage:
    # As llm-d entrypoint
    python aase_llmd_entrypoint.py --model google/gemma-2-2b-it --port 8000
    
    # With custom probe paths
    python aase_llmd_entrypoint.py --model google/gemma-2-2b-it \
        --af-probe /models/af_gemma.npy \
        --aag-probe /models/aag_gemma.npy
    
    # Disable specific probes
    python aase_llmd_entrypoint.py --model google/gemma-2-2b-it --disable-aag

Environment variables:
    AASE_PROBES_DIR: Directory containing pretrained probes
    AASE_BLOCK_UNSAFE: Set to "false" to log but not block (default: true)
    AASE_LOG_ACTIVATIONS: Set to "true" to log activation scores

Kubernetes deployment:
    Add to your llm-d container spec:
    
    containers:
    - name: llm-d
      command: ["python", "/app/aase_llmd_entrypoint.py"]
      args: ["--model", "$(MODEL_NAME)", "--port", "8000"]
      env:
      - name: AASE_PROBES_DIR
        value: "/models/probes"

Author: Glen Messenger
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
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, asdict
from threading import Lock
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("aase-llmd")


# ============================================================================
# AASE PROBE CLASSES (Simplified inline versions)
# ============================================================================

@dataclass
class ProbeResult:
    """Result from a single probe evaluation."""
    probe_type: str
    score: float
    threshold: float
    is_flagged: bool
    latency_ms: float


@dataclass 
class SafetyResult:
    """Combined result from all probes."""
    is_safe: bool
    flagged_probes: List[str]
    results: Dict[str, ProbeResult]
    total_latency_ms: float
    
    def to_dict(self) -> Dict:
        return {
            "is_safe": self.is_safe,
            "flagged_probes": self.flagged_probes,
            "results": {k: asdict(v) for k, v in self.results.items()},
            "total_latency_ms": self.total_latency_ms,
        }


class AASEProbe:
    """Base class for AASE probes."""
    
    def __init__(self, probe_type: str, direction: np.ndarray, threshold: float):
        self.probe_type = probe_type
        self.direction = direction / np.linalg.norm(direction)
        self.threshold = threshold
    
    def score(self, activation: np.ndarray) -> float:
        """Score an activation."""
        normalized = activation / np.linalg.norm(activation)
        return float(np.dot(normalized, self.direction))
    
    def evaluate(self, activation: np.ndarray) -> ProbeResult:
        """Evaluate an activation."""
        start = time.perf_counter()
        score = self.score(activation)
        latency = (time.perf_counter() - start) * 1000
        
        return ProbeResult(
            probe_type=self.probe_type,
            score=score,
            threshold=self.threshold,
            is_flagged=score > self.threshold,
            latency_ms=latency,
        )
    
    @classmethod
    def load(cls, path: str, probe_type: str) -> "AASEProbe":
        """Load probe from .npy and .json files."""
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
# AASE SAFETY GATE
# ============================================================================

class AASESafetyGate:
    """
    Safety gate that evaluates activations against multiple probes.
    
    Thread-safe for concurrent request handling.
    """
    
    def __init__(self, block_unsafe: bool = True):
        self.probes: Dict[str, AASEProbe] = {}
        self.block_unsafe = block_unsafe
        self._lock = Lock()
        
        # Activation storage (written by hooks, read by evaluate)
        self._current_activation: Optional[np.ndarray] = None
        self._activation_file = "/tmp/aase_activation.npy"
    
    def add_probe(self, probe: AASEProbe) -> None:
        """Add a probe to the gate."""
        with self._lock:
            self.probes[probe.probe_type] = probe
            logger.info(f"Added probe: {probe.probe_type} (threshold={probe.threshold:.4f})")
    
    def load_probe(self, path: str, probe_type: str) -> None:
        """Load and add a probe from file."""
        probe = AASEProbe.load(path, probe_type)
        self.add_probe(probe)
    
    def get_activation(self) -> Optional[np.ndarray]:
        """Get the current activation (set by hook)."""
        if os.path.exists(self._activation_file):
            return np.load(self._activation_file)
        return self._current_activation
    
    def set_activation(self, activation: np.ndarray) -> None:
        """Set current activation (called by hook)."""
        self._current_activation = activation
        np.save(self._activation_file, activation)
    
    def evaluate(self, activation: Optional[np.ndarray] = None) -> SafetyResult:
        """Evaluate activation against all probes."""
        start = time.perf_counter()
        
        if activation is None:
            activation = self.get_activation()
        
        if activation is None:
            logger.warning("No activation available for evaluation")
            return SafetyResult(
                is_safe=True,
                flagged_probes=[],
                results={},
                total_latency_ms=0,
            )
        
        results = {}
        flagged = []
        
        with self._lock:
            for probe_type, probe in self.probes.items():
                result = probe.evaluate(activation)
                results[probe_type] = result
                if result.is_flagged:
                    flagged.append(probe_type)
        
        total_latency = (time.perf_counter() - start) * 1000
        
        return SafetyResult(
            is_safe=len(flagged) == 0,
            flagged_probes=flagged,
            results=results,
            total_latency_ms=total_latency,
        )
    
    def check_and_log(self, prompt: str) -> SafetyResult:
        """Check safety and log results."""
        result = self.evaluate()
        
        if not result.is_safe:
            logger.warning(
                f"UNSAFE content detected: flagged_by={result.flagged_probes}, "
                f"prompt='{prompt[:50]}...'"
            )
            for probe_type, probe_result in result.results.items():
                if probe_result.is_flagged:
                    logger.warning(
                        f"  {probe_type}: score={probe_result.score:.4f}, "
                        f"threshold={probe_result.threshold:.4f}"
                    )
        
        return result


# ============================================================================
# VLLM INTEGRATION
# ============================================================================

# Global safety gate instance
_safety_gate: Optional[AASESafetyGate] = None


def create_activation_hook(layer_index: int, safety_gate: AASESafetyGate):
    """Create a hook function for activation extraction."""
    
    def register_hook(model):
        """Register forward hook on specified layer."""
        layers = model.model.layers
        
        def hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            # Take last token activation
            activation = hidden[-1, :].detach().float().cpu().numpy()
            safety_gate.set_activation(activation)
        
        layers[layer_index].register_forward_hook(hook_fn)
        logger.info(f"Registered activation hook on layer {layer_index}")
        return {"layer": layer_index, "total_layers": len(layers)}
    
    return register_hook


def setup_aase(
    llm,
    probes_dir: str,
    model_name: str,
    layer_index: Optional[int] = None,
    enable_af: bool = True,
    enable_aag: bool = True,
    enable_apc: bool = True,
    block_unsafe: bool = True,
) -> AASESafetyGate:
    """
    Set up AASE safety gate with hooks and probes.
    
    Args:
        llm: vLLM LLM instance
        probes_dir: Directory containing pretrained probes
        model_name: Model name for finding correct probes
        layer_index: Layer for activation extraction (default: 50% depth)
        enable_af: Enable Activation Fingerprinting
        enable_aag: Enable Agent Action Gating
        enable_apc: Enable Activation Policy Compliance
        block_unsafe: Block unsafe requests (vs just logging)
    
    Returns:
        Configured AASESafetyGate
    """
    global _safety_gate
    
    # Get model config
    hf_config = llm.llm_engine.model_config.hf_config
    num_layers = hf_config.num_hidden_layers
    
    if layer_index is None:
        layer_index = num_layers // 2  # 50% depth default
    
    logger.info(f"Setting up AASE for {model_name}")
    logger.info(f"  Layers: {num_layers}, Hook at: {layer_index}")
    
    # Create safety gate
    _safety_gate = AASESafetyGate(block_unsafe=block_unsafe)
    
    # Register activation hook
    hook_fn = create_activation_hook(layer_index, _safety_gate)
    llm.apply_model(hook_fn)
    
    # Load probes
    probes_path = Path(probes_dir)
    model_safe_name = model_name.replace("/", "_").replace("-", "_")
    
    probe_configs = []
    if enable_af:
        probe_configs.append(("af", probes_path / "af" / model_safe_name))
    if enable_aag:
        probe_configs.append(("aag", probes_path / "aag" / model_safe_name))
    if enable_apc:
        probe_configs.append(("apc", probes_path / "apc" / f"{model_safe_name}_medical_advice"))
    
    for probe_type, probe_path in probe_configs:
        npy_path = f"{probe_path}.npy"
        if os.path.exists(npy_path):
            _safety_gate.load_probe(str(probe_path), probe_type)
        else:
            logger.warning(f"Probe not found: {npy_path}")
    
    logger.info(f"AASE setup complete. Active probes: {list(_safety_gate.probes.keys())}")
    
    return _safety_gate


def get_safety_gate() -> Optional[AASESafetyGate]:
    """Get the global safety gate instance."""
    return _safety_gate


# ============================================================================
# OPENAI-COMPATIBLE SERVER WITH AASE
# ============================================================================

def create_aase_server(
    llm,
    safety_gate: AASESafetyGate,
    host: str = "0.0.0.0",
    port: int = 8000,
):
    """
    Create an OpenAI-compatible server with AASE safety checking.
    
    This wraps vLLM's serving logic to add safety checks.
    """
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    from typing import List, Optional, Union
    import uvicorn
    
    app = FastAPI(title="AASE-Enabled LLM Server")
    
    # Store LLM reference
    app.state.llm = llm
    app.state.safety_gate = safety_gate
    
    class CompletionRequest(BaseModel):
        model: str
        prompt: Union[str, List[str]]
        max_tokens: int = 100
        temperature: float = 0.7
        top_p: float = 1.0
        stream: bool = False
    
    class ChatMessage(BaseModel):
        role: str
        content: str
    
    class ChatCompletionRequest(BaseModel):
        model: str
        messages: List[ChatMessage]
        max_tokens: int = 100
        temperature: float = 0.7
        top_p: float = 1.0
        stream: bool = False
    
    @app.get("/health")
    async def health():
        return {"status": "healthy", "aase_enabled": True}
    
    @app.get("/v1/models")
    async def list_models():
        model_name = llm.llm_engine.model_config.model
        return {
            "data": [{"id": model_name, "object": "model"}]
        }
    
    @app.post("/v1/completions")
    async def completions(request: CompletionRequest):
        from vllm import SamplingParams
        
        prompts = [request.prompt] if isinstance(request.prompt, str) else request.prompt
        
        # Generate to capture activation (uses hook)
        check_params = SamplingParams(max_tokens=1, temperature=0.0)
        llm.generate(prompts[:1], check_params)  # Check first prompt
        
        # Evaluate safety
        safety_result = safety_gate.check_and_log(prompts[0])
        
        if not safety_result.is_safe and safety_gate.block_unsafe:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": "Request blocked by safety filter",
                        "type": "safety_violation",
                        "flagged_by": safety_result.flagged_probes,
                    }
                }
            )
        
        # Generate response
        params = SamplingParams(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )
        outputs = llm.generate(prompts, params)
        
        choices = []
        for i, output in enumerate(outputs):
            choices.append({
                "index": i,
                "text": output.outputs[0].text,
                "finish_reason": "stop",
            })
        
        return {
            "id": f"cmpl-{int(time.time())}",
            "object": "text_completion",
            "model": request.model,
            "choices": choices,
            "safety": safety_result.to_dict(),
        }
    
    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        from vllm import SamplingParams
        
        # Format messages as prompt
        prompt = "\n".join([f"{m.role}: {m.content}" for m in request.messages])
        prompt += "\nassistant:"
        
        # Check safety
        check_params = SamplingParams(max_tokens=1, temperature=0.0)
        llm.generate([prompt], check_params)
        
        safety_result = safety_gate.check_and_log(prompt)
        
        if not safety_result.is_safe and safety_gate.block_unsafe:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": "Request blocked by safety filter",
                        "type": "safety_violation",
                        "flagged_by": safety_result.flagged_probes,
                    }
                }
            )
        
        # Generate
        params = SamplingParams(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )
        outputs = llm.generate([prompt], params)
        
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": outputs[0].outputs[0].text,
                },
                "finish_reason": "stop",
            }],
            "safety": safety_result.to_dict(),
        }
    
    @app.get("/v1/safety/status")
    async def safety_status():
        """Get AASE safety gate status."""
        return {
            "enabled": True,
            "block_unsafe": safety_gate.block_unsafe,
            "probes": list(safety_gate.probes.keys()),
            "probe_details": {
                name: {"threshold": probe.threshold}
                for name, probe in safety_gate.probes.items()
            }
        }
    
    return app, host, port


# ============================================================================
# MAIN ENTRYPOINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="AASE-Enabled llm-d Server")
    
    # Model args
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-model-len", type=int, default=4096)
    
    # Server args
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    
    # AASE args
    parser.add_argument("--probes-dir", 
                       default=os.environ.get("AASE_PROBES_DIR", "/models/probes"))
    parser.add_argument("--layer", type=int, default=None,
                       help="Layer for activation extraction (default: 50% depth)")
    parser.add_argument("--disable-af", action="store_true")
    parser.add_argument("--disable-aag", action="store_true")
    parser.add_argument("--disable-apc", action="store_true")
    parser.add_argument("--no-block", action="store_true",
                       help="Log unsafe content but don't block")
    
    # Probe file overrides
    parser.add_argument("--af-probe", help="Path to AF probe file")
    parser.add_argument("--aag-probe", help="Path to AAG probe file")
    parser.add_argument("--apc-probe", help="Path to APC probe file")
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info(" AASE-Enabled llm-d Server")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Probes: {args.probes_dir}")
    logger.info(f"Port: {args.port}")
    
    # Load vLLM
    from vllm import LLM
    
    logger.info("Loading model...")
    llm = LLM(
        model=args.model,
        trust_remote_code=args.trust_remote_code,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=True,  # Required for hooks
    )
    
    # Setup AASE
    block_unsafe = not args.no_block and os.environ.get("AASE_BLOCK_UNSAFE", "true").lower() != "false"
    
    safety_gate = setup_aase(
        llm=llm,
        probes_dir=args.probes_dir,
        model_name=args.model,
        layer_index=args.layer,
        enable_af=not args.disable_af,
        enable_aag=not args.disable_aag,
        enable_apc=not args.disable_apc,
        block_unsafe=block_unsafe,
    )
    
    # Load override probes if specified
    if args.af_probe:
        safety_gate.load_probe(args.af_probe, "af")
    if args.aag_probe:
        safety_gate.load_probe(args.aag_probe, "aag")
    if args.apc_probe:
        safety_gate.load_probe(args.apc_probe, "apc")
    
    # Create and run server
    app, host, port = create_aase_server(llm, safety_gate, args.host, args.port)
    
    import uvicorn
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
