"""
Activation Policy Compliance (APC) - Enterprise Policy Enforcement

APC enforces enterprise-specific policies by detecting policy violations
in model outputs. Uses contrastive pair training to isolate the "speech act"
signal from topic contamination.

Key characteristics:
- Uses CONTRASTIVE PAIR training (critical for policy detection)
- Optimal layer depth: 15-40% of model (earlier layers)
- Captures speech act, not topic
- Supports arbitrary enterprise policies

Usage:
    from aase import APC
    
    # Define policy pairs (violation, compliant) on same topic
    pairs = [
        ("Take ibuprofen 400mg", "Consult a doctor about ibuprofen"),
        ("Buy AAPL stock now!", "Consider consulting a financial advisor"),
    ]
    
    # Train
    apc = APC(policy_name="medical_advice")
    apc.train_from_pairs(pairs, extractor)
    
    # Evaluate
    result = apc.evaluate(activation)
    if result.is_flagged:
        print(f"Policy violation detected!")
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

from aase.core.probe import ActivationProbe, ProbeResult, TrainingMetrics
from aase.core.extraction import ActivationExtractor


class APC(ActivationProbe):
    """
    Activation Policy Compliance for enterprise policy enforcement.
    
    Detects:
    - Medical advice (when policy prohibits)
    - Financial advice
    - Legal advice
    - Unauthorized commitments
    - PII disclosure
    - Custom enterprise policies
    
    KEY INNOVATION: Contrastive pair training
    
    Standard approach: direction = mean(violations) - mean(compliant)
        Problem: Captures TOPIC, not SPEECH ACT
        
    Contrastive approach: direction = mean(violation_i - compliant_i)
        Solution: Topic cancels out, speech act signal remains
        
    This is why APC requires matched pairs during training.
    """
    
    PROBE_TYPE = "apc"
    PROBE_NAME = "ActivationPolicyCompliance"
    DEFAULT_LAYER_DEPTH = 0.25  # 25% depth - earlier layers best for APC
    POSITIVE_LABEL = "VIOLATION"
    NEGATIVE_LABEL = "COMPLIANT"
    
    def __init__(
        self,
        direction: Optional[np.ndarray] = None,
        threshold: float = 0.0,
        layer_index: Optional[int] = None,
        layer_depth: Optional[float] = None,
        model_name: Optional[str] = None,
        policy_name: str = "default",
        policy_description: str = "",
    ):
        """
        Initialize APC probe.
        
        Args:
            policy_name: Name of the policy being enforced
            policy_description: Human-readable policy description
        """
        super().__init__(direction, threshold, layer_index, layer_depth, model_name)
        self.policy_name = policy_name
        self.policy_description = policy_description
    
    # Example medical advice policy pairs
    MEDICAL_ADVICE_PAIRS = [
        (
            "You should take ibuprofen for your headache. 400mg every 6 hours should help.",
            "Ibuprofen is commonly used for headaches. However, I can't recommend specific medications - please consult a doctor."
        ),
        (
            "For your cold, take some vitamin C and zinc supplements. Also try echinacea.",
            "Vitamin C, zinc, and echinacea are sometimes used for colds. A healthcare provider can advise what's appropriate for you."
        ),
        (
            "You should try melatonin for your insomnia. Start with 3mg before bed.",
            "Melatonin is a common sleep aid. I'd recommend discussing sleep issues with a doctor who can evaluate your situation."
        ),
        (
            "Apply ice for the first 48 hours, then switch to heat. Take naproxen for the pain.",
            "Ice and heat therapy are common approaches for back pain. A medical professional can provide personalized advice."
        ),
        (
            "You should try deep breathing exercises and consider asking your doctor about SSRIs.",
            "Deep breathing and medications like SSRIs are used for anxiety. A mental health professional can assess what's right for you."
        ),
        (
            "Take cetirizine daily for your allergies. It's safe for long-term use.",
            "Cetirizine is an antihistamine used for allergies. An allergist can recommend the best treatment for your specific symptoms."
        ),
        (
            "You should take omeprazole for your acid reflux. 20mg in the morning works well.",
            "Omeprazole is commonly prescribed for acid reflux. A gastroenterologist can determine if it's appropriate for you."
        ),
        (
            "Apply hydrocortisone cream to that rash twice daily. It should clear up in a week.",
            "Hydrocortisone cream is used for various skin conditions. A dermatologist can properly diagnose and treat your rash."
        ),
        (
            "Take acetaminophen for your fever. 500mg every 4-6 hours until it breaks.",
            "Acetaminophen can help reduce fever. If you have a persistent fever, please seek medical attention."
        ),
        (
            "Glucosamine supplements will help your joint pain. Take 1500mg daily.",
            "Glucosamine is sometimes used for joint health. A rheumatologist can evaluate your joint pain and recommend treatment."
        ),
    ]
    
    # Example financial advice policy pairs
    FINANCIAL_ADVICE_PAIRS = [
        (
            "You should buy Tesla stock now, it's going to double.",
            "Tesla is a popular stock. A financial advisor can help assess if it fits your investment goals."
        ),
        (
            "Put all your savings in Bitcoin immediately.",
            "Bitcoin is a cryptocurrency investment option. Consider consulting a financial advisor about crypto investments."
        ),
        (
            "Refinance your mortgage now before rates go up.",
            "Mortgage refinancing can be beneficial in some situations. A mortgage advisor can evaluate your specific case."
        ),
        (
            "Max out your credit cards and invest in this opportunity.",
            "Borrowing to invest is a high-risk strategy. A financial planner can help you understand the risks."
        ),
        (
            "You should withdraw from your 401k to pay off debt.",
            "Early 401k withdrawals have tax implications. A tax professional can help you understand the tradeoffs."
        ),
    ]
    
    def train(
        self,
        positive_examples: List[str],
        negative_examples: List[str],
        extractor: ActivationExtractor,
        verbose: bool = True,
    ) -> TrainingMetrics:
        """
        Train APC from unpaired examples.
        
        WARNING: This uses mean difference which may capture topic signal.
        For better results, use train_from_pairs() with matched pairs.
        """
        if verbose:
            print("[APC] Warning: Using unpaired training. "
                  "Consider train_from_pairs() for better results.")
        
        # Extract activations
        violation_acts = []
        for text in positive_examples:
            prompt = f"User: What should I do?\nAssistant: {text}"
            act = extractor.extract(prompt)
            violation_acts.append(act)
        
        compliant_acts = []
        for text in negative_examples:
            prompt = f"User: What should I do?\nAssistant: {text}"
            act = extractor.extract(prompt)
            compliant_acts.append(act)
        
        return self.train_from_activations(
            np.array(violation_acts),
            np.array(compliant_acts),
            method="mean_diff"
        )
    
    def train_from_pairs(
        self,
        pairs: List[Tuple[str, str]],
        extractor: ActivationExtractor,
        context_template: str = "User: What should I do about my health concern?\nAssistant: {text}",
        verbose: bool = True,
    ) -> TrainingMetrics:
        """
        Train APC using CONTRASTIVE PAIR methodology.
        
        This is the recommended training method for APC. Each pair
        contains (violation, compliant) responses on the SAME topic,
        allowing the direction vector to capture the speech act
        difference rather than topic.
        
        Args:
            pairs: List of (violation, compliant) text pairs
            extractor: Activation extractor
            context_template: Template with {text} placeholder
            verbose: Print progress
            
        Returns:
            Training metrics
        """
        if verbose:
            print(f"[APC] Training policy '{self.policy_name}' with {len(pairs)} pairs")
            print("[APC] Using contrastive pair methodology")
        
        violation_acts = []
        compliant_acts = []
        
        for i, (violation, compliant) in enumerate(pairs):
            # Format with context template
            viol_prompt = context_template.format(text=violation)
            comp_prompt = context_template.format(text=compliant)
            
            # Extract activations
            viol_act = extractor.extract(viol_prompt)
            comp_act = extractor.extract(comp_prompt)
            
            violation_acts.append(viol_act)
            compliant_acts.append(comp_act)
            
            if verbose and (i + 1) % 5 == 0:
                print(f"  {i + 1}/{len(pairs)} pairs processed")
        
        # Train using contrastive method
        metrics = self.train_from_activations(
            np.array(violation_acts),
            np.array(compliant_acts),
            method="contrastive"
        )
        
        if verbose:
            print(f"[APC] Contrastive training complete:")
            print(f"  Separation: {metrics.separation:.2f}σ")
            print(f"  Train accuracy: {metrics.train_accuracy:.1%}")
        
        return metrics
    
    def train_medical_advice(
        self,
        extractor: ActivationExtractor,
        additional_pairs: Optional[List[Tuple[str, str]]] = None,
        verbose: bool = True,
    ) -> TrainingMetrics:
        """
        Train APC for medical advice prohibition policy.
        
        Convenience method using built-in medical advice pairs.
        """
        self.policy_name = "medical_advice"
        self.policy_description = "Prohibit giving specific medical recommendations"
        
        pairs = self.MEDICAL_ADVICE_PAIRS.copy()
        if additional_pairs:
            pairs.extend(additional_pairs)
        
        return self.train_from_pairs(
            pairs,
            extractor,
            context_template="User: What should I do about my health concern?\nAssistant: {text}",
            verbose=verbose
        )
    
    def train_financial_advice(
        self,
        extractor: ActivationExtractor,
        additional_pairs: Optional[List[Tuple[str, str]]] = None,
        verbose: bool = True,
    ) -> TrainingMetrics:
        """
        Train APC for financial advice prohibition policy.
        """
        self.policy_name = "financial_advice"
        self.policy_description = "Prohibit giving specific financial recommendations"
        
        pairs = self.FINANCIAL_ADVICE_PAIRS.copy()
        if additional_pairs:
            pairs.extend(additional_pairs)
        
        return self.train_from_pairs(
            pairs,
            extractor,
            context_template="User: What should I do with my money?\nAssistant: {text}",
            verbose=verbose
        )
    
    def evaluate(self, activation: np.ndarray) -> ProbeResult:
        """Evaluate with policy metadata."""
        result = super().evaluate(activation)
        result.metadata["policy_name"] = self.policy_name
        result.metadata["policy_description"] = self.policy_description
        return result
    
    def save(self, path) -> None:
        """Save with policy info."""
        super().save(path)
        
        # Update metadata with policy info
        import json
        meta_path = f"{path}.json"
        with open(meta_path) as f:
            meta = json.load(f)
        
        meta["policy_name"] = self.policy_name
        meta["policy_description"] = self.policy_description
        
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
    
    @classmethod
    def load(cls, path) -> "APC":
        """Load with policy info."""
        probe = super().load(path)
        
        # Load policy info from metadata
        import json
        with open(f"{path}.json") as f:
            meta = json.load(f)
        
        probe.policy_name = meta.get("policy_name", "default")
        probe.policy_description = meta.get("policy_description", "")
        
        return probe
    
    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        policy: str = "medical_advice"
    ) -> "APC":
        """Load pre-trained APC probe for a policy."""
        import os
        pretrained_dir = os.path.join(
            os.path.dirname(__file__),
            "..", "pretrained", "apc"
        )
        
        safe_name = model_name.replace("/", "_").replace("-", "_")
        vector_path = os.path.join(pretrained_dir, f"{safe_name}_{policy}")
        
        if os.path.exists(f"{vector_path}.npy"):
            return cls.load(vector_path)
        else:
            raise FileNotFoundError(
                f"No pre-trained APC probe found for {model_name} policy {policy}"
            )
