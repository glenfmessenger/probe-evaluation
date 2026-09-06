#!/usr/bin/env python3
"""Fix for Gemma 3 models - different config attribute names."""

import os
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

import sys
import argparse
import json
import numpy as np
import torch
from pathlib import Path
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Import training data from pretrain_robust
from pretrain_robust import (
    AF_HARMFUL, AF_BENIGN,
    AAG_INJECTION, AAG_SAFE,
    APC_MEDICAL_VIOLATION, APC_MEDICAL_COMPLIANT,
    APC_LEGAL_VIOLATION, APC_LEGAL_COMPLIANT,
    APC_FINANCIAL_VIOLATION, APC_FINANCIAL_COMPLIANT,
    APC_CRISIS_VIOLATION, APC_CRISIS_COMPLIANT,
    train_direction_vector, save_probe
)


class Gemma3Extractor:
    """Extractor that handles Gemma 3's different config structure."""
    
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
        
        # Gemma 3 uses different config attributes
        config = self.model.config
        
        # Try different attribute names
        if hasattr(config, 'num_hidden_layers'):
            self.num_layers = config.num_hidden_layers
        elif hasattr(config, 'text_config') and hasattr(config.text_config, 'num_hidden_layers'):
            self.num_layers = config.text_config.num_hidden_layers
        elif hasattr(config, 'num_layers'):
            self.num_layers = config.num_layers
        else:
            # Count layers manually
            self.num_layers = len([n for n in self.model.named_modules() if 'layers.' in n[0] and '.self_attn' in n[0]])
            logger.info(f"Counted {self.num_layers} layers manually")
        
        if hasattr(config, 'hidden_size'):
            self.hidden_dim = config.hidden_size
        elif hasattr(config, 'text_config') and hasattr(config.text_config, 'hidden_size'):
            self.hidden_dim = config.text_config.hidden_size
        else:
            self.hidden_dim = None
        
        self.device = device
        logger.info(f"Model loaded: {self.num_layers} layers, {self.hidden_dim} hidden dim")
    
    @torch.no_grad()
    def extract(self, text: str, layer_index: int) -> np.ndarray:
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=2048
        ).to(self.model.device)
        
        outputs = self.model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[layer_index]
        last_pos = inputs.attention_mask.sum(dim=1) - 1
        return hidden[0, last_pos[0], :].float().cpu().numpy()
    
    def extract_batch(self, texts, layer_index: int) -> np.ndarray:
        activations = []
        for text in texts:
            activations.append(self.extract(text, layer_index))
        return np.array(activations)


def pretrain_gemma3(model_name: str, output_dir: str):
    """Pretrain probes for Gemma 3 models."""
    
    logger.info("=" * 70)
    logger.info(f" AASE Pretraining for {model_name}")
    logger.info("=" * 70)
    
    extractor = Gemma3Extractor(model_name)
    num_layers = extractor.num_layers
    
    af_layer = int(0.50 * num_layers)
    aag_layer = int(0.55 * num_layers)
    apc_layer = int(0.60 * num_layers)
    
    logger.info(f"Layers: AF={af_layer}, AAG={aag_layer}, APC={apc_layer}")
    
    model_safe = model_name.replace("/", "_").replace("-", "_")
    results = {}
    
    # AF
    logger.info("\nTraining AF...")
    af_harmful_acts = extractor.extract_batch(AF_HARMFUL, af_layer)
    af_benign_acts = extractor.extract_batch(AF_BENIGN, af_layer)
    af_direction, af_metrics = train_direction_vector(af_harmful_acts, af_benign_acts)
    logger.info(f"  AF: {af_metrics.separation_sigma:.1f}σ, {af_metrics.accuracy:.0%}")
    save_probe(af_direction, af_metrics, f"{output_dir}/af/{model_safe}", "af", model_name, af_layer, 0.50)
    results["af"] = af_metrics.accuracy
    
    # AAG
    logger.info("\nTraining AAG...")
    aag_inj_acts = extractor.extract_batch(AAG_INJECTION, aag_layer)
    aag_safe_acts = extractor.extract_batch(AAG_SAFE, aag_layer)
    aag_direction, aag_metrics = train_direction_vector(aag_inj_acts, aag_safe_acts)
    logger.info(f"  AAG: {aag_metrics.separation_sigma:.1f}σ, {aag_metrics.accuracy:.0%}")
    save_probe(aag_direction, aag_metrics, f"{output_dir}/aag/{model_safe}", "aag", model_name, aag_layer, 0.55)
    results["aag"] = aag_metrics.accuracy
    
    # APC policies
    apc_data = [
        ("medical", APC_MEDICAL_VIOLATION, APC_MEDICAL_COMPLIANT),
        ("legal", APC_LEGAL_VIOLATION, APC_LEGAL_COMPLIANT),
        ("financial", APC_FINANCIAL_VIOLATION, APC_FINANCIAL_COMPLIANT),
        ("crisis", APC_CRISIS_VIOLATION, APC_CRISIS_COMPLIANT),
    ]
    
    for policy, violations, compliant in apc_data:
        logger.info(f"\nTraining APC-{policy}...")
        v_acts = extractor.extract_batch(violations, apc_layer)
        c_acts = extractor.extract_batch(compliant, apc_layer)
        direction, metrics = train_direction_vector(v_acts, c_acts)
        logger.info(f"  {policy}: {metrics.separation_sigma:.1f}σ, {metrics.accuracy:.0%}")
        save_probe(direction, metrics, f"{output_dir}/apc/{model_safe}_{policy}", "apc", model_name, apc_layer, 0.60,
                   extra_meta={"policy_name": policy})
        results[f"apc_{policy}"] = metrics.accuracy
    
    logger.info("\n" + "=" * 70)
    logger.info(" COMPLETE")
    logger.info("=" * 70)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-3-4b-it")
    parser.add_argument("--output", default="pretrained")
    args = parser.parse_args()
    
    pretrain_gemma3(args.model, args.output)
