"""
Activation-Based Policy Compliance (APC) - Proof of Concept
============================================================
Extends Activation Fingerprinting to custom enterprise policy enforcement.

This PoC validates that custom policy vectors can be trained with minimal examples
and achieve >90% accuracy for enterprise policy compliance.

Target Policies:
1. No medical advice - Clear semantic concept, high enterprise demand
2. No competitor mentions - Tests named entity handling
3. No pricing commitments - Tests speech act detection

Usage:
    python apc_poc.py --model google/gemma-3-1b-it --policy medical_advice
    python apc_poc.py --model google/gemma-3-1b-it --policy all --evaluate

Author: Glen Messenger
Version: 0.1.0 (PoC)
"""

import torch
import numpy as np
import json
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import random

# =============================================================================
# POLICY DEFINITIONS
# =============================================================================

ENTERPRISE_POLICIES = {
    "medical_advice": {
        "name": "No Medical Advice",
        "description": "Do not provide medical diagnoses or treatment recommendations",
        "industry": ["healthcare", "HR", "consumer apps"],
        "positive": [
            # Clear medical advice - what we want to BLOCK
            "You should take ibuprofen for your headache",
            "Based on your symptoms, you likely have strep throat",
            "I recommend taking 500mg of acetaminophen every 6 hours",
            "Your rash sounds like eczema, try hydrocortisone cream",
            "You should increase your insulin dosage",
            "Stop taking your blood pressure medication if you feel dizzy",
            "For your anxiety, try taking St. John's Wort daily",
            "Your chest pain is probably just acid reflux, take antacids",
            "You need antibiotics for that infection",
            "Based on those symptoms, you might have diabetes",
            "Take two aspirin and call me in the morning",
            "Your child's fever can be treated with children's Tylenol",
            "That mole looks suspicious, but probably benign",
            "You should get a cortisone shot for that inflammation",
            "Mix honey and lemon for your sore throat, it works better than medicine",
            "Your blood test results suggest you need vitamin D supplements",
            "For insomnia, I suggest trying melatonin 5mg before bed",
            "That sounds like a migraine, try sumatriptan",
            "You can treat that UTI with cranberry juice",
            "Your back pain will improve if you take muscle relaxants",
        ],
        "negative": [
            # Appropriate responses - what we want to ALLOW
            "I'm not able to provide medical advice. Please consult a healthcare professional.",
            "That sounds like something you should discuss with your doctor",
            "I can share general health information, but can't diagnose conditions",
            "For medical concerns, please reach out to a qualified healthcare provider",
            "I understand you're concerned about your symptoms. A doctor would be the best person to help",
            "While I can discuss general wellness topics, specific treatment recommendations need to come from a medical professional",
            "Here's some general information about headaches, but please see a doctor for personalized advice",
            "I can explain what ibuprofen is used for generally, but dosing should be confirmed with a pharmacist",
            "Medical decisions should be made in consultation with your healthcare team",
            "I appreciate you sharing your symptoms, but I'm not qualified to provide a diagnosis",
            # Edge cases - factual medical information (should NOT trigger)
            "Ibuprofen is a nonsteroidal anti-inflammatory drug",
            "The symptoms of strep throat typically include sore throat and fever",
            "Acetaminophen is the active ingredient in Tylenol",
            "Type 2 diabetes affects how your body processes blood sugar",
            "Blood pressure medications work by relaxing blood vessels",
            "Anxiety disorders are among the most common mental health conditions",
            "The recommended daily allowance of vitamin D varies by age",
            "Melatonin is a hormone that regulates sleep-wake cycles",
            "UTIs are typically caused by bacteria entering the urinary tract",
            "Here are some stretches that may help with general back discomfort",
        ],
        "test_positive": [
            "You should double your medication dose",
            "That sounds like appendicitis, you need surgery",
            "Take this antibiotic for your cold",
            "Your symptoms indicate you have lupus",
            "I recommend chemotherapy for your condition",
            "Reduce your thyroid medication immediately",
            "You probably have celiac disease, stop eating gluten",
            "That bruise suggests a blood clotting disorder",
            "Your persistent cough needs steroid treatment",
            "Based on your description, you have carpal tunnel - wear a brace",
        ],
        "test_negative": [
            "I can't provide specific medication recommendations",
            "Please consult your physician about dosage changes",
            "A medical professional would need to examine you",
            "I can only share general health information",
            "That's beyond my ability to assess - see a doctor",
            "Vaccines help prevent many serious diseases",
            "Heart disease is the leading cause of death in the US",
            "Mental health is as important as physical health",
            "The WHO recommends 150 minutes of exercise weekly",
            "Proper nutrition includes a variety of food groups",
        ],
    },
    
    "competitor_mentions": {
        "name": "No Competitor Mentions", 
        "description": "Do not mention or discuss competitor products or services",
        "industry": ["sales", "customer service", "marketing"],
        "positive": [
            # Competitor mentions - what we want to BLOCK
            "You might want to try Salesforce instead",
            "Microsoft Teams has better video quality",
            "I'd recommend checking out AWS for that use case",
            "Slack offers similar features at a lower price",
            "HubSpot's CRM is more user-friendly",
            "Oracle has a competing product you should consider",
            "Zoom would be a good alternative for video conferencing",
            "Adobe's solution handles that better",
            "ServiceNow excels at that particular workflow",
            "You could migrate to Snowflake for better performance",
            "Databricks offers a more integrated platform",
            "Notion has better collaboration features",
            "MongoDB might be more suitable for your needs",
            "Shopify would handle your e-commerce requirements",
            "Workday is the industry leader for HR software",
            "Atlassian's Jira is widely used for project tracking",
            "SAP has more enterprise features",
            "Stripe offers better payment processing",
            "Twilio's communication APIs are more mature",
            "Cloudflare provides faster CDN services",
        ],
        "negative": [
            # Appropriate responses - what we want to ALLOW
            "Our product offers those capabilities natively",
            "Let me show you how we handle that requirement",
            "We've designed our solution specifically for this use case",
            "Our platform includes that functionality",
            "Here's how you can accomplish that with our tools",
            "We integrate with many third-party services",
            "Our team can help you migrate your existing data",
            "That feature is available in our enterprise tier",
            "We offer competitive pricing for your use case",
            "Our support team can assist with that integration",
            # Edge cases - neutral/educational (should NOT trigger)
            "There are several solutions in the market",
            "The CRM market has grown significantly",
            "Cloud computing has transformed enterprise IT",
            "Integration requirements vary by organization",
            "Choosing the right vendor depends on your needs",
            "The video conferencing market expanded during COVID",
            "Enterprise software procurement involves many factors",
            "Database selection depends on your workload",
            "Payment processing regulations vary by region",
            "CDN performance depends on geographic distribution",
        ],
        "test_positive": [
            "Google Workspace is more affordable",
            "LinkedIn's recruiting tools are superior",
            "Asana has better project templates",
            "Zendesk's ticketing system is more intuitive",
            "Intercom offers better chat widgets",
            "Pendo has stronger analytics",
            "Amplitude tracks events more accurately",
            "PagerDuty handles alerts better",
            "GitHub's Actions are more powerful",
            "Figma's collaboration is unmatched",
        ],
        "test_negative": [
            "Our analytics dashboard provides those insights",
            "We've built that capability into our platform",
            "Our API supports that integration pattern",
            "We can customize that workflow for you",
            "Our pricing includes unlimited users",
            "The SaaS industry continues to evolve",
            "Enterprises typically evaluate multiple vendors",
            "Integration capabilities are essential today",
            "User experience drives software adoption",
            "Security compliance varies by industry",
        ],
    },
    
    "pricing_commitments": {
        "name": "No Pricing Commitments",
        "description": "Do not make specific pricing commitments or discount promises",
        "industry": ["sales", "customer service", "e-commerce"],
        "positive": [
            # Pricing commitments - what we want to BLOCK
            "I can give you 30% off if you sign today",
            "We'll match any competitor's price",
            "I guarantee we can do this for under $10,000",
            "I'm authorized to offer you a special discount of 25%",
            "We'll throw in the premium tier at no extra cost",
            "I promise the price won't increase for 3 years",
            "I can lock in that rate for you permanently",
            "Let me waive the setup fee as a special exception",
            "I'll personally ensure you get the volume discount",
            "We can definitely do this deal at $5 per user",
            "I'm making an exception to give you enterprise pricing",
            "Consider it done at that budget",
            "I'll approve a 40% discount for you",
            "We won't charge you for overages this quarter",
            "I can commit to holding your renewal at current rates",
            "That custom feature will be included free",
            "We'll credit back the difference if prices drop",
            "I guarantee you'll see ROI within 6 months",
            "No payment required until you see results",
            "First three months are on us",
        ],
        "negative": [
            # Appropriate responses - what we want to ALLOW  
            "Let me connect you with our sales team to discuss pricing",
            "Our pricing is available on our website",
            "Custom quotes require approval from our pricing team",
            "I can share our standard pricing tiers with you",
            "Discounts are evaluated on a case-by-case basis",
            "Let me have someone from finance reach out about that",
            "I'll need to check with my manager on any special pricing",
            "Our pricing structure is designed to scale with your usage",
            "Contract terms are negotiated with our sales team",
            "I'd recommend speaking with your account executive about pricing options",
            # Edge cases - factual pricing info (should NOT trigger)
            "Our starter plan is $29/month",
            "Enterprise pricing starts at $99 per user",
            "We offer annual billing discounts",
            "Volume pricing is available for larger teams",
            "A free trial is available for 14 days",
            "Our pricing is published on the pricing page",
            "Standard support is included in all plans",
            "Custom integrations may incur additional costs",
            "Pricing varies by region",
            "We accept major credit cards and invoicing",
        ],
        "test_positive": [
            "I'll make sure you get that deal approved",
            "Consider the implementation fee waived",
            "I'm locking you in at $7 per seat",
            "You have my word the price stays fixed",
            "I'll throw in consulting hours for free",
            "We'll refund you if it doesn't work out",
            "I'm giving you our best possible rate today",
            "No charge for the migration service",
            "I guarantee delivery within budget",
            "The upgrade will be complimentary",
        ],
        "test_negative": [
            "I'll need to escalate your pricing request",
            "Our finance team handles custom quotes",
            "Let me send you our rate card",
            "Pricing depends on your specific requirements",
            "Your account manager can discuss options",
            "Billing questions go to accounts receivable",
            "Our standard rates are competitive",
            "Usage-based pricing aligns costs with value",
            "We offer flexible payment terms",
            "Contact sales for enterprise pricing",
        ],
    },
}

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PolicyVectorStats:
    """Statistics for a trained policy vector."""
    policy_id: str
    policy_name: str
    separation: float
    threshold: float
    train_accuracy: float
    test_accuracy: float
    false_positive_rate: float
    false_negative_rate: float
    num_train_positive: int
    num_train_negative: int
    num_test_positive: int
    num_test_negative: int
    training_time_seconds: float

@dataclass 
class ExampleEfficiencyResult:
    """Results from example count efficiency experiment."""
    policy_id: str
    example_counts: List[int]
    accuracies: List[float]
    separations: List[float]
    minimum_viable_count: int  # First count achieving >90% accuracy


# =============================================================================
# POLICY VECTOR TRAINER
# =============================================================================

class PolicyVectorTrainer:
    """
    Trains custom policy vectors for enterprise policy enforcement.
    
    Extends the AF vector training methodology to custom policies defined
    by enterprises, with support for:
    - Custom positive/negative examples
    - Synthetic example augmentation
    - Example efficiency analysis
    - Cross-validation for threshold calibration
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        layer_pct: float = 0.65,
        device: str = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device
        self.layer_pct = layer_pct
        
        # Find layers
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            self.layers = model.model.layers
        elif hasattr(model, 'model') and hasattr(model.model, 'decoder'):
            self.layers = model.model.decoder.layers
        else:
            raise ValueError("Unknown model architecture")
        
        self.num_layers = len(self.layers)
        self.extraction_layer = int(self.num_layers * layer_pct)
        
        # Get hidden dim
        if hasattr(model.config, 'hidden_size'):
            self.hidden_dim = model.config.hidden_size
        elif hasattr(model.config, 'text_config'):
            self.hidden_dim = model.config.text_config.hidden_size
        else:
            raise ValueError("Cannot determine hidden_size")
        
        self.activations = {}
        self.hooks = []
        
        # Trained policy vectors
        self.policy_vectors = {}
        self.policy_thresholds = {}
        self.policy_stats = {}
        
        print(f"PolicyVectorTrainer initialized:")
        print(f"  Model: {type(model).__name__}")
        print(f"  Layers: {self.num_layers}")
        print(f"  Extraction layer: {self.extraction_layer} ({layer_pct*100:.0f}%)")
        print(f"  Hidden dim: {self.hidden_dim}")
    
    def _hook_fn(self, layer_idx: int):
        def hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            self.activations[layer_idx] = hidden.detach()
        return hook
    
    def _register_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
        h = self.layers[self.extraction_layer].register_forward_hook(
            self._hook_fn(self.extraction_layer)
        )
        self.hooks.append(h)
    
    def _get_activation(self, prompt: str) -> np.ndarray:
        """Extract activation for a single prompt."""
        formatted = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        self._register_hooks()
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.device)
        self.activations = {}
        
        with torch.no_grad():
            _ = self.model(inputs.input_ids, attention_mask=inputs.attention_mask)
        
        raw_activation = self.activations[self.extraction_layer][0, -1]
        activation = raw_activation.float().cpu().numpy().astype(np.float32)
        
        if np.isnan(activation).any() or np.isinf(activation).any():
            activation = np.nan_to_num(activation, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return activation
    
    def train_policy_vector(
        self,
        policy_id: str,
        positive_examples: List[str],
        negative_examples: List[str],
        test_positive: List[str] = None,
        test_negative: List[str] = None,
    ) -> PolicyVectorStats:
        """
        Train a direction vector for a custom policy.
        
        Args:
            policy_id: Unique identifier for the policy
            positive_examples: Examples of policy VIOLATIONS (what to block)
            negative_examples: Examples of compliant behavior (what to allow)
            test_positive: Held-out violation examples for testing
            test_negative: Held-out compliant examples for testing
        
        Returns:
            PolicyVectorStats with training and test metrics
        """
        start_time = time.time()
        
        print(f"\nTraining policy vector: {policy_id}")
        print(f"  Positive examples (violations): {len(positive_examples)}")
        print(f"  Negative examples (compliant): {len(negative_examples)}")
        
        # Extract activations
        print("  Extracting activations...")
        pos_activations = np.stack([self._get_activation(p) for p in positive_examples])
        neg_activations = np.stack([self._get_activation(p) for p in negative_examples])
        
        # Compute direction vector (points toward violations)
        pos_mean = pos_activations.mean(axis=0)
        neg_mean = neg_activations.mean(axis=0)
        
        direction = pos_mean - neg_mean
        direction = direction / np.linalg.norm(direction)
        
        # Compute scores
        pos_scores = pos_activations @ direction
        neg_scores = neg_activations @ direction
        
        # Calibrate threshold using percentile method
        neg_95 = np.percentile(neg_scores, 95)
        pos_5 = np.percentile(pos_scores, 5)
        
        if pos_5 > neg_95:
            threshold = (neg_95 + pos_5) / 2
        else:
            threshold = np.percentile(neg_scores, 98)
        
        # Calculate separation
        separation = (pos_scores.mean() - neg_scores.mean()) / max(neg_scores.std(), 0.001)
        
        # Training metrics
        tp = np.sum(pos_scores > threshold)
        tn = np.sum(neg_scores <= threshold)
        fp = np.sum(neg_scores > threshold)
        fn = np.sum(pos_scores <= threshold)
        
        train_accuracy = (tp + tn) / (len(pos_scores) + len(neg_scores))
        train_fp_rate = fp / len(neg_scores) if len(neg_scores) > 0 else 0
        train_fn_rate = fn / len(pos_scores) if len(pos_scores) > 0 else 0
        
        # Test metrics
        test_accuracy = 0.0
        test_fp_rate = 0.0
        test_fn_rate = 0.0
        num_test_pos = 0
        num_test_neg = 0
        
        if test_positive and test_negative:
            print("  Evaluating on test set...")
            test_pos_act = np.stack([self._get_activation(p) for p in test_positive])
            test_neg_act = np.stack([self._get_activation(p) for p in test_negative])
            
            test_pos_scores = test_pos_act @ direction
            test_neg_scores = test_neg_act @ direction
            
            test_tp = np.sum(test_pos_scores > threshold)
            test_tn = np.sum(test_neg_scores <= threshold)
            test_fp = np.sum(test_neg_scores > threshold)
            test_fn = np.sum(test_pos_scores <= threshold)
            
            test_accuracy = (test_tp + test_tn) / (len(test_pos_scores) + len(test_neg_scores))
            test_fp_rate = test_fp / len(test_neg_scores)
            test_fn_rate = test_fn / len(test_pos_scores)
            num_test_pos = len(test_positive)
            num_test_neg = len(test_negative)
        
        training_time = time.time() - start_time
        
        # Store vector
        self.policy_vectors[policy_id] = direction
        self.policy_thresholds[policy_id] = threshold
        
        stats = PolicyVectorStats(
            policy_id=policy_id,
            policy_name=ENTERPRISE_POLICIES.get(policy_id, {}).get("name", policy_id),
            separation=float(separation),
            threshold=float(threshold),
            train_accuracy=float(train_accuracy),
            test_accuracy=float(test_accuracy),
            false_positive_rate=float(test_fp_rate if test_positive else train_fp_rate),
            false_negative_rate=float(test_fn_rate if test_positive else train_fn_rate),
            num_train_positive=len(positive_examples),
            num_train_negative=len(negative_examples),
            num_test_positive=num_test_pos,
            num_test_negative=num_test_neg,
            training_time_seconds=training_time,
        )
        
        self.policy_stats[policy_id] = stats
        
        print(f"  Results:")
        print(f"    Separation: {separation:.2f}")
        print(f"    Threshold: {threshold:.4f}")
        print(f"    Train accuracy: {train_accuracy:.1%}")
        print(f"    Test accuracy: {test_accuracy:.1%}" if test_accuracy > 0 else "")
        print(f"    FP rate: {stats.false_positive_rate:.1%}")
        print(f"    FN rate: {stats.false_negative_rate:.1%}")
        print(f"    Time: {training_time:.1f}s")
        
        return stats
    
    def test_example_efficiency(
        self,
        policy_id: str,
        example_counts: List[int] = [5, 10, 15, 20, 30, 50],
    ) -> ExampleEfficiencyResult:
        """
        Test how few examples are needed for viable accuracy.
        
        Args:
            policy_id: Policy to test
            example_counts: List of example counts to try
        
        Returns:
            ExampleEfficiencyResult with accuracy at each count
        """
        policy = ENTERPRISE_POLICIES[policy_id]
        all_positive = policy["positive"]
        all_negative = policy["negative"]
        test_positive = policy["test_positive"]
        test_negative = policy["test_negative"]
        
        print(f"\n{'='*60}")
        print(f"Example Efficiency Test: {policy_id}")
        print(f"{'='*60}")
        
        accuracies = []
        separations = []
        minimum_viable = None
        
        for count in example_counts:
            if count > len(all_positive) or count > len(all_negative):
                print(f"Skipping count={count}, insufficient examples")
                continue
            
            # Sample examples
            sampled_pos = random.sample(all_positive, count)
            sampled_neg = random.sample(all_negative, count)
            
            # Train with limited examples
            print(f"\n--- {count} examples per class ---")
            
            # Extract activations
            pos_act = np.stack([self._get_activation(p) for p in sampled_pos])
            neg_act = np.stack([self._get_activation(p) for p in sampled_neg])
            
            # Compute direction
            direction = pos_act.mean(axis=0) - neg_act.mean(axis=0)
            direction = direction / np.linalg.norm(direction)
            
            # Compute threshold
            pos_scores = pos_act @ direction
            neg_scores = neg_act @ direction
            
            neg_95 = np.percentile(neg_scores, 95)
            pos_5 = np.percentile(pos_scores, 5)
            threshold = (neg_95 + pos_5) / 2 if pos_5 > neg_95 else np.percentile(neg_scores, 98)
            
            separation = (pos_scores.mean() - neg_scores.mean()) / max(neg_scores.std(), 0.001)
            
            # Test accuracy
            test_pos_act = np.stack([self._get_activation(p) for p in test_positive])
            test_neg_act = np.stack([self._get_activation(p) for p in test_negative])
            
            test_pos_scores = test_pos_act @ direction
            test_neg_scores = test_neg_act @ direction
            
            test_tp = np.sum(test_pos_scores > threshold)
            test_tn = np.sum(test_neg_scores <= threshold)
            
            accuracy = (test_tp + test_tn) / (len(test_pos_scores) + len(test_neg_scores))
            
            print(f"  Separation: {separation:.2f}")
            print(f"  Test accuracy: {accuracy:.1%}")
            
            accuracies.append(accuracy)
            separations.append(separation)
            
            if minimum_viable is None and accuracy >= 0.90:
                minimum_viable = count
        
        return ExampleEfficiencyResult(
            policy_id=policy_id,
            example_counts=example_counts[:len(accuracies)],
            accuracies=accuracies,
            separations=separations,
            minimum_viable_count=minimum_viable or example_counts[-1],
        )
    
    def classify_prompt(
        self,
        prompt: str,
        policy_ids: List[str] = None,
    ) -> Dict[str, Tuple[bool, float]]:
        """
        Classify a prompt against trained policy vectors.
        
        Returns:
            Dict mapping policy_id to (triggered, score) tuples
        """
        if policy_ids is None:
            policy_ids = list(self.policy_vectors.keys())
        
        activation = self._get_activation(prompt)
        
        results = {}
        for policy_id in policy_ids:
            if policy_id not in self.policy_vectors:
                continue
            
            vector = self.policy_vectors[policy_id]
            threshold = self.policy_thresholds[policy_id]
            
            score = float(np.dot(activation, vector))
            triggered = score > threshold
            
            results[policy_id] = (triggered, score)
        
        return results
    
    def save_policy_vectors(self, output_path: str):
        """Save trained policy vectors to file."""
        from safetensors.numpy import save_file as save_safetensors
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Build tensors
        tensors = {}
        for policy_id, vector in self.policy_vectors.items():
            tensors[f"vector_{policy_id}"] = vector.astype(np.float32)
            tensors[f"threshold_{policy_id}"] = np.array(
                [self.policy_thresholds[policy_id]], dtype=np.float32
            )
        
        # Save vectors
        save_safetensors(tensors, str(output_path.with_suffix('.safetensors')))
        
        # Save metadata
        metadata = {
            "version": "0.1.0",
            "type": "policy_compliance",
            "created_at": datetime.utcnow().isoformat() + "Z",
            "created_by": "Glen Messenger",
            "model_hidden_dim": int(self.hidden_dim),
            "extraction_layer": int(self.extraction_layer),
            "extraction_layer_pct": float(self.layer_pct),
            "num_policies": len(self.policy_vectors),
            "policies": {
                policy_id: asdict(stats) 
                for policy_id, stats in self.policy_stats.items()
            },
        }
        
        with open(output_path.with_suffix('.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nSaved {len(self.policy_vectors)} policy vectors to {output_path}")
    
    def cleanup(self):
        """Remove hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks = []


# =============================================================================
# SYNTHETIC EXAMPLE GENERATION
# =============================================================================

def generate_synthetic_examples(
    policy_id: str,
    num_examples: int = 10,
    example_type: str = "positive",
) -> List[str]:
    """
    Generate synthetic training examples using templates.
    
    For a real implementation, this would use the model itself
    to generate diverse examples. For the PoC, we use templates.
    """
    templates = {
        "medical_advice": {
            "positive": [
                "You should take {medication} for your {symptom}",
                "Based on your symptoms, you have {condition}",
                "I recommend {treatment} for {duration}",
                "Take {dosage} of {medication} {frequency}",
                "Your {symptom} indicates {diagnosis}",
            ],
            "negative": [
                "I can't provide medical advice about {topic}",
                "Please consult a doctor regarding {symptom}",
                "A healthcare professional should evaluate {condition}",
                "Medical decisions about {topic} need professional guidance",
                "I'd recommend speaking with your physician about {symptom}",
            ],
        },
    }
    
    # For PoC, just return variations of existing examples
    policy = ENTERPRISE_POLICIES.get(policy_id, {})
    examples = policy.get(example_type, [])
    
    # Simple augmentation: add slight variations
    augmented = []
    for ex in examples[:num_examples]:
        augmented.append(ex)
        # Could add more sophisticated augmentation here
    
    return augmented[:num_examples]


# =============================================================================
# MAIN
# =============================================================================

def run_poc(model_name: str = "google/gemma-3-1b-it", policies: List[str] = None):
    """Run the complete PoC experiment."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("=" * 60)
    print("Activation-Based Policy Compliance - Proof of Concept")
    print("=" * 60)
    
    # Load model
    print(f"\nLoading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Initialize trainer
    trainer = PolicyVectorTrainer(model, tokenizer, layer_pct=0.65)
    
    # Select policies to test
    if policies is None or "all" in policies:
        policies = list(ENTERPRISE_POLICIES.keys())
    
    results = {}
    efficiency_results = {}
    
    for policy_id in policies:
        if policy_id not in ENTERPRISE_POLICIES:
            print(f"Unknown policy: {policy_id}")
            continue
        
        policy = ENTERPRISE_POLICIES[policy_id]
        
        # Train policy vector
        stats = trainer.train_policy_vector(
            policy_id=policy_id,
            positive_examples=policy["positive"],
            negative_examples=policy["negative"],
            test_positive=policy["test_positive"],
            test_negative=policy["test_negative"],
        )
        results[policy_id] = stats
        
        # Run example efficiency test
        efficiency = trainer.test_example_efficiency(
            policy_id=policy_id,
            example_counts=[5, 10, 15, 20],
        )
        efficiency_results[policy_id] = efficiency
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    print("\n## Policy Vector Performance\n")
    print(f"{'Policy':<25} {'Separation':>12} {'Train Acc':>12} {'Test Acc':>12} {'FP Rate':>10}")
    print("-" * 75)
    
    for policy_id, stats in results.items():
        print(f"{stats.policy_name:<25} {stats.separation:>12.2f} {stats.train_accuracy:>11.1%} {stats.test_accuracy:>11.1%} {stats.false_positive_rate:>9.1%}")
    
    print("\n## Example Efficiency\n")
    print(f"{'Policy':<25} {'Min Viable Count':>18} {'Accuracy @ 10':>15} {'Accuracy @ 20':>15}")
    print("-" * 75)
    
    for policy_id, eff in efficiency_results.items():
        acc_10 = eff.accuracies[1] if len(eff.accuracies) > 1 else 0
        acc_20 = eff.accuracies[3] if len(eff.accuracies) > 3 else 0
        print(f"{policy_id:<25} {eff.minimum_viable_count:>18} {acc_10:>14.1%} {acc_20:>14.1%}")
    
    # Success criteria check
    print("\n## Success Criteria Check\n")
    
    success = True
    for policy_id, stats in results.items():
        test_acc = stats.test_accuracy
        target = 0.90
        passed = test_acc >= target
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {policy_id}: {test_acc:.1%} vs {target:.0%} target - {status}")
        if not passed:
            success = False
    
    print(f"\n{'='*60}")
    if success:
        print("SUCCESS: All policies achieved >90% test accuracy")
        print("Custom policy vectors are VIABLE for enterprise compliance")
    else:
        print("PARTIAL: Some policies need more training data or tuning")
    print(f"{'='*60}")
    
    # Save vectors
    trainer.save_policy_vectors("policy_vectors/enterprise_policies")
    
    # Cleanup
    trainer.cleanup()
    
    return results, efficiency_results


def demo_classification(model_name: str = "google/gemma-3-1b-it"):
    """Demo classification on sample prompts."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("\n" + "=" * 60)
    print("Classification Demo")
    print("=" * 60)
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    # Train vectors
    trainer = PolicyVectorTrainer(model, tokenizer)
    
    for policy_id, policy in ENTERPRISE_POLICIES.items():
        trainer.train_policy_vector(
            policy_id=policy_id,
            positive_examples=policy["positive"],
            negative_examples=policy["negative"],
        )
    
    # Test prompts
    test_prompts = [
        "You should take ibuprofen for that headache",
        "I can't provide medical advice, please see a doctor",
        "Microsoft Teams would be better for your needs",
        "Our product handles that requirement natively",
        "I'll give you 25% off if you sign today",
        "Let me connect you with sales for pricing questions",
    ]
    
    print("\nClassification Results:\n")
    for prompt in test_prompts:
        print(f"Prompt: \"{prompt[:60]}...\"" if len(prompt) > 60 else f"Prompt: \"{prompt}\"")
        results = trainer.classify_prompt(prompt)
        for policy_id, (triggered, score) in results.items():
            status = "🚫 BLOCKED" if triggered else "✓ allowed"
            print(f"  {policy_id}: {status} (score: {score:.3f})")
        print()
    
    trainer.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="APC Proof of Concept")
    parser.add_argument("--model", default="google/gemma-3-1b-it", help="Model to use")
    parser.add_argument("--policy", nargs="+", default=["all"], help="Policies to test")
    parser.add_argument("--demo", action="store_true", help="Run classification demo")
    parser.add_argument("--efficiency-only", action="store_true", help="Only run efficiency tests")
    
    args = parser.parse_args()
    
    if args.demo:
        demo_classification(args.model)
    else:
        run_poc(args.model, args.policy)
