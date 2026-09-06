#!/usr/bin/env python3
"""
llm-d Model Instantiation Probe

This script investigates how llm-d instantiates its vLLM model to determine
if we can use apply_model() hooks for AASE integration.

Run this script inside an llm-d pod or container to inspect the runtime.

Usage:
    # Option 1: Run inside llm-d container
    kubectl exec -it <llm-d-pod> -- python /path/to/probe_llmd.py
    
    # Option 2: Run locally against llm-d source
    python probe_llmd.py --source-only
    
    # Option 3: Connect to running llm-d process
    python probe_llmd.py --pid <llm-d-pid>

Author: Glen Messenger
"""

import os
import sys
import argparse
import subprocess
import json
from pathlib import Path
from typing import Optional, Dict, Any


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'='*70}")
    print(f" {title}")
    print('='*70)


def check_environment():
    """Check the runtime environment."""
    print_section("ENVIRONMENT CHECK")
    
    env_vars = [
        "VLLM_*",
        "MODEL_*", 
        "LLM_*",
        "CUDA_*",
        "KUBERNETES_*",
    ]
    
    print("\nRelevant environment variables:")
    for key, value in sorted(os.environ.items()):
        for pattern in env_vars:
            if pattern.endswith("*"):
                if key.startswith(pattern[:-1]):
                    print(f"  {key}={value[:100]}{'...' if len(value) > 100 else ''}")
                    break
            elif key == pattern:
                print(f"  {key}={value[:100]}{'...' if len(value) > 100 else ''}")
                break
    
    print("\nPython path:")
    for p in sys.path[:10]:
        print(f"  {p}")
    
    print(f"\nWorking directory: {os.getcwd()}")
    print(f"Python executable: {sys.executable}")


def check_installed_packages():
    """Check installed packages related to vLLM/llm-d."""
    print_section("INSTALLED PACKAGES")
    
    packages = ["vllm", "llm-d", "llmd", "torch", "transformers"]
    
    for pkg in packages:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "show", pkg],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                info = {l.split(': ')[0]: l.split(': ')[1] if ': ' in l else '' 
                       for l in lines if ': ' in l}
                print(f"\n{pkg}:")
                print(f"  Version: {info.get('Version', 'N/A')}")
                print(f"  Location: {info.get('Location', 'N/A')}")
        except Exception as e:
            print(f"\n{pkg}: Not found or error ({e})")


def find_llmd_source():
    """Try to locate llm-d source code."""
    print_section("LLM-D SOURCE LOCATION")
    
    # Common locations
    search_paths = [
        "/app",
        "/opt/llm-d",
        "/home/llm-d",
        "/workspace",
        os.getcwd(),
    ]
    
    # Also check pip package locations
    try:
        import vllm
        search_paths.append(os.path.dirname(vllm.__file__))
    except ImportError:
        pass
    
    llmd_files = []
    for base_path in search_paths:
        if not os.path.exists(base_path):
            continue
        for root, dirs, files in os.walk(base_path):
            # Skip common non-code directories
            dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git', 'node_modules', 'venv']]
            for f in files:
                if f.endswith('.py'):
                    filepath = os.path.join(root, f)
                    try:
                        with open(filepath, 'r') as file:
                            content = file.read()
                            # Look for llm-d specific patterns
                            if any(pattern in content for pattern in [
                                'llm-d', 'llmd', 'LLMEngine', 'apply_model',
                                'vllm.LLM', 'from vllm import'
                            ]):
                                llmd_files.append(filepath)
                    except:
                        pass
    
    if llmd_files:
        print("\nFound relevant Python files:")
        for f in llmd_files[:20]:
            print(f"  {f}")
        if len(llmd_files) > 20:
            print(f"  ... and {len(llmd_files) - 20} more")
    else:
        print("\nNo llm-d specific source files found")
    
    return llmd_files


def analyze_vllm_usage(files: list):
    """Analyze how vLLM is used in the found files."""
    print_section("VLLM USAGE ANALYSIS")
    
    patterns = {
        "LLM class instantiation": ["LLM(", "vllm.LLM("],
        "LLMEngine usage": ["LLMEngine", "AsyncLLMEngine"],
        "apply_model hook": ["apply_model(", ".apply_model"],
        "Server mode": ["api_server", "openai.api_server", "vllm serve"],
        "Model loading": ["from_pretrained", "AutoModel"],
        "Forward hooks": ["register_forward_hook", "forward_hook"],
    }
    
    results = {k: [] for k in patterns}
    
    for filepath in files[:50]:  # Limit to avoid long processing
        try:
            with open(filepath, 'r') as f:
                content = f.read()
                lines = content.split('\n')
                
                for pattern_name, pattern_list in patterns.items():
                    for pattern in pattern_list:
                        if pattern in content:
                            # Find the line numbers
                            for i, line in enumerate(lines):
                                if pattern in line:
                                    results[pattern_name].append({
                                        "file": filepath,
                                        "line": i + 1,
                                        "content": line.strip()[:100]
                                    })
        except:
            pass
    
    for pattern_name, matches in results.items():
        print(f"\n{pattern_name}:")
        if matches:
            for match in matches[:5]:
                print(f"  {match['file']}:{match['line']}")
                print(f"    {match['content']}")
            if len(matches) > 5:
                print(f"  ... and {len(matches) - 5} more matches")
        else:
            print("  No matches found")


def check_running_processes():
    """Check for running vLLM/llm-d processes."""
    print_section("RUNNING PROCESSES")
    
    try:
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True, text=True
        )
        
        relevant = []
        for line in result.stdout.split('\n'):
            if any(p in line.lower() for p in ['vllm', 'llm-d', 'llmd', 'python']):
                relevant.append(line)
        
        if relevant:
            print("\nRelevant processes:")
            for line in relevant[:20]:
                print(f"  {line[:120]}")
        else:
            print("\nNo relevant processes found")
    except Exception as e:
        print(f"\nCould not list processes: {e}")


def inspect_vllm_internals():
    """Inspect vLLM internals if available."""
    print_section("VLLM INTERNALS")
    
    try:
        import vllm
        print(f"\nvLLM version: {vllm.__version__}")
        print(f"vLLM location: {vllm.__file__}")
        
        # Check for LLM class
        from vllm import LLM
        print(f"\nLLM class location: {LLM.__module__}")
        
        # Check for apply_model method
        if hasattr(LLM, 'apply_model'):
            print("✓ LLM.apply_model() method EXISTS")
            import inspect
            sig = inspect.signature(LLM.apply_model)
            print(f"  Signature: apply_model{sig}")
        else:
            print("✗ LLM.apply_model() method NOT FOUND")
        
        # Check for engine classes
        try:
            from vllm.engine.llm_engine import LLMEngine
            print("✓ LLMEngine available")
        except ImportError:
            print("✗ LLMEngine not available")
        
        try:
            from vllm.engine.async_llm_engine import AsyncLLMEngine
            print("✓ AsyncLLMEngine available")
        except ImportError:
            print("✗ AsyncLLMEngine not available")
        
        # Check entrypoints
        print("\nEntrypoints:")
        entrypoints_dir = os.path.join(os.path.dirname(vllm.__file__), "entrypoints")
        if os.path.exists(entrypoints_dir):
            for f in os.listdir(entrypoints_dir):
                if f.endswith('.py'):
                    print(f"  {f}")
        
    except ImportError as e:
        print(f"\nvLLM not available: {e}")


def test_apply_model_capability():
    """Test if apply_model works in current environment."""
    print_section("APPLY_MODEL CAPABILITY TEST")
    
    try:
        from vllm import LLM
        
        print("\nChecking if we can use apply_model() for activation extraction...")
        print("(This requires a GPU and will load a small model)")
        
        # Check for GPU
        try:
            import torch
            if torch.cuda.is_available():
                print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            else:
                print("✗ CUDA not available - skipping model load test")
                return
        except ImportError:
            print("✗ PyTorch not available - skipping model load test")
            return
        
        # Ask user before loading model
        response = input("\nLoad a small model to test apply_model? (y/N): ")
        if response.lower() != 'y':
            print("Skipping model load test")
            return
        
        print("\nLoading small test model (google/gemma-2-2b-it)...")
        
        os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
        
        llm = LLM(
            model="google/gemma-2-2b-it",
            trust_remote_code=True,
            gpu_memory_utilization=0.5,
            max_model_len=512,
            enforce_eager=True,
        )
        
        # Test apply_model
        print("\nTesting apply_model()...")
        
        def inspect_model(model):
            """Inspect model structure."""
            info = {
                "model_type": type(model).__name__,
                "has_layers": hasattr(model, 'model') and hasattr(model.model, 'layers'),
            }
            if info["has_layers"]:
                info["num_layers"] = len(model.model.layers)
                info["layer_type"] = type(model.model.layers[0]).__name__
            return info
        
        result = llm.apply_model(inspect_model)
        print(f"\n✓ apply_model() works!")
        print(f"  Model info: {result}")
        
        # Test hook registration
        print("\nTesting hook registration...")
        
        import numpy as np
        activation_captured = [False]
        
        def register_hook(model):
            layers = model.model.layers
            target_layer = len(layers) // 2  # Middle layer
            
            def hook_fn(module, input, output):
                hidden = output[0] if isinstance(output, tuple) else output
                activation_captured[0] = True
                print(f"    Hook fired! Activation shape: {hidden.shape}")
            
            layers[target_layer].register_forward_hook(hook_fn)
            return {"registered_at_layer": target_layer}
        
        result = llm.apply_model(register_hook)
        print(f"  Hook registered: {result}")
        
        # Trigger the hook
        from vllm import SamplingParams
        params = SamplingParams(max_tokens=1, temperature=0.0)
        llm.generate(["Hello"], params)
        
        if activation_captured[0]:
            print("\n✓ Activation extraction via hooks WORKS!")
            print("\n" + "="*70)
            print(" CONCLUSION: AASE can integrate directly with llm-d")
            print(" if llm-d exposes the LLM instance or allows startup hooks")
            print("="*70)
        else:
            print("\n✗ Hook did not fire")
        
    except Exception as e:
        print(f"\nError during test: {e}")
        import traceback
        traceback.print_exc()


def check_llmd_github():
    """Fetch and analyze llm-d source from GitHub."""
    print_section("LLM-D GITHUB SOURCE ANALYSIS")
    
    print("\nFetching llm-d repository structure...")
    
    try:
        import urllib.request
        
        # GitHub API for repo contents
        api_url = "https://api.github.com/repos/llm-d/llm-d/contents"
        
        req = urllib.request.Request(api_url)
        req.add_header('User-Agent', 'AASE-Probe/1.0')
        
        with urllib.request.urlopen(req, timeout=10) as response:
            contents = json.loads(response.read().decode())
        
        print("\nRepository root contents:")
        for item in contents:
            print(f"  [{item['type']}] {item['name']}")
        
        # Look for key files
        key_files = ['main.py', 'server.py', 'entrypoint.py', 'app.py', 'run.py']
        for item in contents:
            if item['name'].lower() in [f.lower() for f in key_files]:
                print(f"\n  Found potential entry point: {item['name']}")
                # Fetch content
                file_url = item['download_url']
                with urllib.request.urlopen(file_url, timeout=10) as response:
                    file_content = response.read().decode()
                    # Show first 50 lines
                    lines = file_content.split('\n')[:50]
                    print("  First 50 lines:")
                    for line in lines:
                        print(f"    {line[:100]}")
        
    except Exception as e:
        print(f"\nCould not fetch from GitHub: {e}")
        print("This is expected if running in an isolated environment.")


def generate_integration_recommendation():
    """Generate integration recommendations based on findings."""
    print_section("INTEGRATION RECOMMENDATIONS")
    
    print("""
Based on the analysis, here are the integration paths for AASE:

1. IF llm-d uses LLM class directly:
   - Modify llm-d startup to call apply_model() with AASE hooks
   - Add /v1/activations endpoint to return captured activations
   - Zero additional infrastructure needed
   
2. IF llm-d uses AsyncLLMEngine:
   - Similar approach but need to hook into engine initialization
   - May require subclassing the engine
   
3. IF llm-d uses vllm serve (CLI):
   - Need to create custom entrypoint that:
     a) Loads model with LLM class
     b) Registers AASE hooks via apply_model()
     c) Starts serving with the hooked model
   - OR: Fork vllm's api_server.py to add hooks
   
4. FALLBACK - Proxy architecture:
   - Only needed if llm-d is a black box we can't modify
   - Deploy separate AASE service with its own vLLM
   
Next steps:
   a) Determine which pattern llm-d uses (run this script in llm-d container)
   b) Implement the appropriate integration
   c) Test activation extraction works correctly
""")


def main():
    parser = argparse.ArgumentParser(description="Probe llm-d model instantiation")
    parser.add_argument("--source-only", action="store_true", 
                       help="Only analyze source code, don't test runtime")
    parser.add_argument("--pid", type=int, 
                       help="PID of running llm-d process to inspect")
    parser.add_argument("--github", action="store_true",
                       help="Analyze llm-d GitHub repository")
    parser.add_argument("--test-hooks", action="store_true",
                       help="Test apply_model hook capability (requires GPU)")
    
    args = parser.parse_args()
    
    print("="*70)
    print(" LLM-D MODEL INSTANTIATION PROBE")
    print(" Determining how to integrate AASE with llm-d")
    print("="*70)
    
    # Always run these
    check_environment()
    check_installed_packages()
    check_running_processes()
    inspect_vllm_internals()
    
    # Find and analyze source
    llmd_files = find_llmd_source()
    if llmd_files:
        analyze_vllm_usage(llmd_files)
    
    # Optional: GitHub analysis
    if args.github:
        check_llmd_github()
    
    # Optional: Test hooks
    if args.test_hooks:
        test_apply_model_capability()
    
    # Always show recommendations
    generate_integration_recommendation()
    
    print("\n" + "="*70)
    print(" PROBE COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
