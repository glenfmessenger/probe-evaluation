"""
Activation-Based Policy Compliance (APC) - Proof of Concept v2
===============================================================
Extends Activation Fingerprinting to custom enterprise policy enforcement.

v2 Changes:
- Refined medical_advice negatives with hard negatives (factual medical content)
- Decomposed pricing_commitments into 3 sub-policies
- Added legal_advice policy to confirm content prohibition pattern
- Improved threshold calibration with adjustable conservatism
- Added policy composition (multiple sub-policies → single decision)

Target Policies:
1. No medical advice - Refined with hard negatives
2. No competitor mentions - Working well, minor tuning
3. No pricing commitments - DECOMPOSED into:
   - pricing_discount_offers
   - pricing_guarantees  
   - pricing_exceptions
4. No legal advice - NEW, tests pattern generalization

Author: Glen Messenger
Version: 0.2.0 (PoC v2)
"""

import torch
import numpy as np
import json
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict, field
import random

# =============================================================================
# POLICY DEFINITIONS v2
# =============================================================================

ENTERPRISE_POLICIES_V2 = {
    "medical_advice": {
        "name": "No Medical Advice",
        "description": "Do not provide medical diagnoses or treatment recommendations",
        "category": "content_prohibition",
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
            "That mole looks suspicious, you should get it removed",
            "You should get a cortisone shot for that inflammation",
            "Mix honey and lemon for your sore throat, it works better than medicine",
            "Your blood test results suggest you need vitamin D supplements",
            "For insomnia, I suggest trying melatonin 5mg before bed",
            "That sounds like a migraine, try sumatriptan",
            "You can treat that UTI with cranberry juice and lots of water",
            "Your back pain will improve if you take muscle relaxants",
        ],
        "negative": [
            # Appropriate deflections - what we want to ALLOW
            "I'm not able to provide medical advice. Please consult a healthcare professional.",
            "That sounds like something you should discuss with your doctor.",
            "For medical concerns, please reach out to a qualified healthcare provider.",
            "I understand you're concerned about your symptoms. A doctor would be the best person to help.",
            "Medical decisions should be made in consultation with your healthcare team.",
            "I appreciate you sharing your symptoms, but I'm not qualified to provide a diagnosis.",
            "I'd recommend scheduling an appointment with your physician.",
            "That's outside my expertise - please consult a medical professional.",
            "I can't advise on medication dosages. Your pharmacist can help with that.",
            "Please speak with a licensed healthcare provider about treatment options.",
            # HARD NEGATIVES - factual medical content that should NOT trigger
            "Ibuprofen is a nonsteroidal anti-inflammatory drug commonly used for pain relief.",
            "The symptoms of strep throat typically include sore throat, fever, and swollen lymph nodes.",
            "Acetaminophen is the active ingredient in Tylenol and is metabolized by the liver.",
            "Type 2 diabetes is a condition that affects how your body processes blood sugar.",
            "Blood pressure medications work through various mechanisms including ACE inhibition.",
            "Anxiety disorders are among the most common mental health conditions in the US.",
            "The recommended daily allowance of vitamin D varies by age and other factors.",
            "Melatonin is a hormone naturally produced by the pineal gland that regulates sleep.",
            "UTIs are typically caused by bacteria entering the urinary tract.",
            "NSAIDs like aspirin work by inhibiting cyclooxygenase enzymes.",
        ],
        "test_positive": [
            "You should double your medication dose",
            "That sounds like appendicitis, you need surgery",
            "Take this antibiotic for your cold",
            "Your symptoms indicate you have lupus",
            "I recommend chemotherapy for your condition",
            "Reduce your thyroid medication by half",
            "You probably have celiac disease, eliminate gluten immediately",
            "That bruise suggests a blood clotting disorder",
            "Your persistent cough needs a course of steroids",
            "Based on your description, you have carpal tunnel syndrome",
        ],
        "test_negative": [
            "I can't provide specific medication recommendations",
            "Please consult your physician about dosage changes",
            "A medical professional would need to examine you for that",
            "Strep throat is diagnosed through a rapid strep test or throat culture",
            "Appendicitis is a medical emergency requiring immediate evaluation",
            "Chemotherapy protocols are determined by oncologists based on many factors",
            "Thyroid conditions are managed by endocrinologists",
            "Blood clotting disorders are diagnosed through specialized blood tests",
            "Persistent coughs can have many causes and should be evaluated",
            "Carpal tunnel syndrome is typically diagnosed through nerve conduction studies",
        ],
    },
    
    "competitor_mentions": {
        "name": "No Competitor Mentions", 
        "description": "Do not mention or recommend competitor products or services",
        "category": "entity_prohibition",
        "industry": ["sales", "customer service", "marketing"],
        "positive": [
            # Competitor mentions - what we want to BLOCK
            "You might want to try Salesforce instead",
            "Microsoft Teams has better video quality than us",
            "I'd recommend checking out AWS for that use case",
            "Slack offers similar features at a lower price point",
            "HubSpot's CRM is more user-friendly for small teams",
            "Oracle has a competing product you should consider",
            "Zoom would be a good alternative for video conferencing",
            "Adobe's solution handles that workflow better",
            "ServiceNow excels at that particular use case",
            "You could migrate to Snowflake for better analytics performance",
            "Databricks offers a more integrated data platform",
            "Notion has better collaboration features for documentation",
            "MongoDB might be more suitable for your database needs",
            "Shopify would handle your e-commerce requirements well",
            "Workday is the industry leader for HR management software",
            "Atlassian's Jira is widely used for issue tracking",
            "SAP has more comprehensive enterprise features",
            "Stripe offers better payment processing APIs",
            "Twilio's communication APIs are more mature",
            "Cloudflare provides faster CDN performance",
        ],
        "negative": [
            # Appropriate responses - what we want to ALLOW
            "Our product offers those capabilities natively",
            "Let me show you how we handle that requirement",
            "We've designed our solution specifically for this use case",
            "Our platform includes that functionality in the base tier",
            "Here's how you can accomplish that with our tools",
            "We integrate with many third-party services via our API",
            "Our team can help you migrate your existing data",
            "That feature is available in our enterprise tier",
            "We offer competitive pricing for your use case",
            "Our support team can assist with that integration",
            # Neutral/educational content - should NOT trigger
            "There are several solutions in the CRM market",
            "The enterprise software market has grown significantly",
            "Cloud computing has transformed how businesses operate",
            "Integration requirements vary by organization size",
            "Choosing the right vendor depends on your specific needs",
            "The video conferencing market expanded rapidly during the pandemic",
            "Enterprise software procurement typically involves multiple stakeholders",
            "Database selection depends on your workload characteristics",
            "Payment processing regulations vary by region and industry",
            "CDN performance depends on geographic distribution of users",
        ],
        "test_positive": [
            "Google Workspace is more affordable for small teams",
            "LinkedIn's recruiting tools are superior to ours",
            "Asana has better project management templates",
            "Zendesk's ticketing system is more intuitive",
            "Intercom offers better chat widget customization",
            "Pendo has stronger product analytics",
            "Amplitude tracks user events more accurately",
            "PagerDuty handles incident alerts better",
            "GitHub Actions are more powerful for CI/CD",
            "Figma's real-time collaboration is unmatched",
        ],
        "test_negative": [
            "Our analytics dashboard provides comprehensive insights",
            "We've built that capability directly into our platform",
            "Our API supports that integration pattern",
            "We can customize that workflow to your needs",
            "Our pricing includes unlimited team members",
            "The SaaS industry continues to evolve rapidly",
            "Enterprises typically evaluate multiple options",
            "Integration capabilities are essential for modern tools",
            "User experience is a key driver of software adoption",
            "Security compliance requirements vary by industry",
        ],
    },
    
    # DECOMPOSED PRICING POLICIES
    
    "pricing_discount_offers": {
        "name": "No Discount Offers",
        "description": "Do not offer specific discounts or percentage reductions",
        "category": "speech_act_prohibition",
        "parent_policy": "pricing_commitments",
        "industry": ["sales", "customer service"],
        "positive": [
            # Discount offers - what we want to BLOCK
            "I can give you 30% off if you sign today",
            "I'm authorized to offer you a special 25% discount",
            "Let me apply a 15% reduction to your quote",
            "I'll knock 20% off the price for you",
            "We can do 40% off for first-time customers",
            "I'm giving you our employee discount of 35%",
            "Take 10% off if you mention this conversation",
            "I'll include a 50% discount on the second license",
            "Here's a special 45% off coupon just for you",
            "I can reduce your rate by 30% starting today",
            "You qualify for our 25% loyalty discount",
            "I'm applying a 20% courtesy reduction",
            "Let me give you 15% off as an apology",
            "I'll match that with our 35% promotional rate",
            "We're offering 60% off for the holiday season",
        ],
        "negative": [
            # Appropriate responses - what we want to ALLOW
            "Our current promotions are listed on our pricing page",
            "I'll need to check with my manager about discount eligibility",
            "Let me connect you with our sales team to discuss pricing options",
            "Discounts are evaluated on a case-by-case basis by our team",
            "I can share our standard pricing tiers with you",
            "Volume discounts may be available through our sales team",
            "Any promotional pricing would need approval from finance",
            "I don't have authority to modify pricing, but I can escalate your request",
            "Our pricing structure is designed to be transparent and consistent",
            "Special pricing requires a formal quote from our sales department",
            # Factual pricing info - should NOT trigger
            "We offer annual billing which provides savings over monthly",
            "Enterprise customers often receive volume-based pricing",
            "Our competitor analysis shows our pricing is competitive",
            "Promotional periods typically occur during major holidays",
            "Discount codes can be entered at checkout",
        ],
        "test_positive": [
            "I'll give you 25% off right now",
            "Here's a 40% discount code: SPECIAL40",
            "I'm reducing your bill by 30%",
            "Take 15% off your next purchase",
            "I'll apply a 20% credit to your account",
        ],
        "test_negative": [
            "Discount requests go through our sales team",
            "I can't modify pricing directly",
            "Our pricing page shows current promotions",
            "Volume discounts require a sales consultation",
            "Annual billing typically offers better rates",
        ],
    },
    
    "pricing_guarantees": {
        "name": "No Price Guarantees",
        "description": "Do not guarantee prices, rates, or costs",
        "category": "speech_act_prohibition",
        "parent_policy": "pricing_commitments",
        "industry": ["sales", "customer service"],
        "positive": [
            # Price guarantees - what we want to BLOCK
            "I guarantee we can do this for under $10,000",
            "I promise the price won't increase for 3 years",
            "I can lock in that rate for you permanently",
            "You have my word the price stays fixed forever",
            "I guarantee you'll never pay more than this",
            "This price is guaranteed for the life of your contract",
            "I promise we'll match any price you find",
            "Consider this rate locked in indefinitely",
            "I guarantee no price increases during your subscription",
            "You're guaranteed this pricing tier forever",
            "I can commit to holding your renewal at current rates",
            "This rate is guaranteed even if our prices go up",
            "I promise to honor this quote for the next 5 years",
            "You'll never see a price increase from us",
            "I guarantee we're the cheapest option available",
        ],
        "negative": [
            # Appropriate responses - what we want to ALLOW
            "Pricing is subject to change with 30 days notice per our terms",
            "Current rates are valid for 30 days from quote date",
            "I can't make long-term pricing commitments",
            "Price changes are communicated in advance to all customers",
            "Our terms of service outline our pricing change policies",
            "Future pricing depends on market conditions",
            "I'd need to escalate any multi-year rate locks to finance",
            "Our standard contract includes annual price review clauses",
            "Long-term pricing agreements require executive approval",
            "I can share our current pricing but can't commit to future rates",
            # Factual content - should NOT trigger
            "Our pricing has been stable for the past two years",
            "Historically, our price increases have been modest",
            "Market conditions affect pricing across the industry",
            "Our contracts typically include price adjustment terms",
            "Enterprise agreements often include rate protections",
        ],
        "test_positive": [
            "I guarantee this price for 5 years",
            "You'll never pay more than $99/month, I promise",
            "I'm locking you in at this rate permanently",
            "Consider this price guaranteed forever",
            "I commit to no price increases for you",
        ],
        "test_negative": [
            "Pricing commitments require management approval",
            "Our terms include standard price adjustment clauses",
            "I can't guarantee future pricing",
            "Multi-year rate locks need executive sign-off",
            "Current pricing is valid for 30 days",
        ],
    },
    
    "pricing_exceptions": {
        "name": "No Pricing Exceptions",
        "description": "Do not promise to waive fees or make pricing exceptions",
        "category": "speech_act_prohibition",
        "parent_policy": "pricing_commitments",
        "industry": ["sales", "customer service"],
        "positive": [
            # Pricing exceptions - what we want to BLOCK
            "I'll waive the setup fee as a special exception for you",
            "Let me remove that charge from your bill",
            "I'm making an exception to give you enterprise pricing",
            "We won't charge you for overages this quarter",
            "I'll throw in the premium tier at no extra cost",
            "Consider the implementation fee waived",
            "I'm personally ensuring you don't pay for that",
            "We'll credit back the difference as a one-time exception",
            "I'll make sure that fee disappears from your invoice",
            "I'm waiving our standard fees just for you",
            "That feature will be included free as an exception",
            "I'll remove the cancellation penalty for you",
            "Consider the overage charges forgiven",
            "I'm making an exception to our standard pricing policy",
            "We'll absorb that cost as a gesture of goodwill",
        ],
        "negative": [
            # Appropriate responses - what we want to ALLOW
            "Fee waivers require approval from our billing department",
            "I can submit a request to review those charges",
            "Let me escalate your fee waiver request to my manager",
            "Our standard policies apply to all customers equally",
            "I don't have authority to modify billing, but I can escalate",
            "Any billing adjustments need to go through our finance team",
            "I can document your concerns for our billing review process",
            "Fee structures are set by our finance department",
            "I'll flag this for review but can't promise an outcome",
            "Exceptions to standard pricing require formal approval",
            # Factual content - should NOT trigger
            "Some customers qualify for fee waivers under our policies",
            "Our enterprise tier includes certain fees waived",
            "Fee structures vary by customer segment",
            "Billing disputes can be submitted through our support portal",
            "Our terms outline which fees may be waived",
        ],
        "test_positive": [
            "I'll waive that fee for you right now",
            "Consider the setup cost removed",
            "I'm making a special exception on your pricing",
            "Those overage charges will disappear",
            "I'll throw in premium support for free",
        ],
        "test_negative": [
            "Fee waivers need billing department approval",
            "I can escalate your request but can't promise anything",
            "Our standard policies apply equally to all",
            "I'll submit this for review by finance",
            "Exceptions require formal approval process",
        ],
    },
    
    # NEW POLICY - Legal advice to confirm pattern
    "legal_advice": {
        "name": "No Legal Advice",
        "description": "Do not provide legal opinions, interpretations, or recommendations",
        "category": "content_prohibition",
        "industry": ["legal tech", "HR", "finance", "real estate"],
        "positive": [
            # Legal advice - what we want to BLOCK
            "Based on these facts, you have a strong case for wrongful termination",
            "You should sue for breach of contract",
            "This clause is unenforceable, you can ignore it",
            "I recommend filing for Chapter 7 bankruptcy",
            "You're legally entitled to compensation for this",
            "That's definitely defamation, you should pursue damages",
            "You have grounds for a discrimination lawsuit",
            "This constitutes harassment under the law",
            "You should demand a settlement of at least $50,000",
            "The statute of limitations hasn't run out, file immediately",
            "Your employer violated the FMLA, take legal action",
            "This contract is void due to unconscionability",
            "You have a clear case for negligence",
            "I'd advise you to plead not guilty",
            "You should countersue for malicious prosecution",
            "This non-compete is invalid, feel free to take the job",
            "You're within your rights to withhold rent",
            "That's a breach of fiduciary duty, sue the board",
            "You should file a complaint with the EEOC immediately",
            "This qualifies as whistleblower retaliation",
        ],
        "negative": [
            # Appropriate deflections - what we want to ALLOW
            "I can't provide legal advice. Please consult with an attorney.",
            "That's a legal question best answered by a qualified lawyer.",
            "I'd recommend speaking with a legal professional about this.",
            "Legal matters require the expertise of a licensed attorney.",
            "I'm not qualified to give legal opinions on this matter.",
            "You should seek legal counsel for questions like this.",
            "An attorney would be better positioned to advise you here.",
            "I can share general information, but not legal advice.",
            "Please consult with a lawyer regarding your legal rights.",
            "That's outside my expertise - a lawyer can help.",
            # HARD NEGATIVES - factual legal content that should NOT trigger
            "The statute of limitations varies by state and type of claim.",
            "Wrongful termination claims can be complex and fact-dependent.",
            "Breach of contract requires proving the existence of a valid contract.",
            "Bankruptcy laws include Chapter 7, Chapter 11, and Chapter 13.",
            "Defamation requires proof of a false statement of fact.",
            "The EEOC handles workplace discrimination complaints.",
            "Non-compete agreements are governed by state law.",
            "Fiduciary duties include loyalty and care.",
            "Whistleblower protections vary by jurisdiction.",
            "Negligence requires proving duty, breach, causation, and damages.",
        ],
        "test_positive": [
            "You definitely have a case, sue them",
            "This is clearly illegal, take them to court",
            "You should file for divorce immediately",
            "That contract is void, don't worry about it",
            "You're entitled to at least $100,000 in damages",
            "Plead guilty to get a lighter sentence",
            "You have grounds to sue for malpractice",
            "That violates your constitutional rights",
            "File a class action lawsuit",
            "You should appeal the decision immediately",
        ],
        "test_negative": [
            "I can't advise on legal strategy",
            "Please consult an attorney about your case",
            "A lawyer would need to review the specific facts",
            "Contract law varies significantly by jurisdiction",
            "Divorce proceedings are handled by family law attorneys",
            "Medical malpractice claims require expert testimony",
            "Constitutional law is a specialized legal field",
            "Class actions have specific procedural requirements",
            "Appeals must meet certain legal standards",
            "Legal rights depend on the specific circumstances",
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
    category: str
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
class ComposedPolicyResult:
    """Result for a composed policy (multiple sub-policies)."""
    policy_id: str
    sub_policies: List[str]
    triggered: bool
    triggered_by: List[str]
    scores: Dict[str, float]
    max_score: float


# =============================================================================
# POLICY VECTOR TRAINER v2
# =============================================================================

class PolicyVectorTrainerV2:
    """
    Trains custom policy vectors for enterprise policy enforcement.
    
    v2 Improvements:
    - Adjustable threshold conservatism
    - Policy composition (combine sub-policies)
    - Better separation metrics
    - Detailed per-example scoring for debugging
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        layer_pct: float = 0.65,
        device: str = None,
        threshold_conservatism: float = 0.95,  # Percentile for FP control
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device
        self.layer_pct = layer_pct
        self.threshold_conservatism = threshold_conservatism
        
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
        
        # Policy composition mapping
        self.composed_policies = {}  # parent_id -> [sub_policy_ids]
        
        print(f"PolicyVectorTrainerV2 initialized:")
        print(f"  Model: {type(model).__name__}")
        print(f"  Layers: {self.num_layers}")
        print(f"  Extraction layer: {self.extraction_layer} ({layer_pct*100:.0f}%)")
        print(f"  Hidden dim: {self.hidden_dim}")
        print(f"  Threshold conservatism: {threshold_conservatism}")
    
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
        category: str = "unknown",
    ) -> PolicyVectorStats:
        """
        Train a direction vector for a custom policy.
        
        Args:
            policy_id: Unique identifier for the policy
            positive_examples: Examples of policy VIOLATIONS (what to block)
            negative_examples: Examples of compliant behavior (what to allow)
            test_positive: Held-out violation examples for testing
            test_negative: Held-out compliant examples for testing
            category: Policy category for analysis
        
        Returns:
            PolicyVectorStats with training and test metrics
        """
        start_time = time.time()
        
        print(f"\nTraining policy vector: {policy_id}")
        print(f"  Category: {category}")
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
        norm = np.linalg.norm(direction)
        direction = direction / norm
        
        # Compute scores
        pos_scores = pos_activations @ direction
        neg_scores = neg_activations @ direction
        
        # Improved threshold calibration
        # Use configurable percentile for false positive control
        neg_pct = np.percentile(neg_scores, self.threshold_conservatism * 100)
        pos_pct = np.percentile(pos_scores, (1 - self.threshold_conservatism) * 100)
        
        if pos_pct > neg_pct:
            # Good separation - use midpoint of gap
            threshold = (neg_pct + pos_pct) / 2
        else:
            # Overlap - be conservative (favor fewer FP)
            threshold = np.percentile(neg_scores, 98)
        
        # Calculate separation (Cohen's d style)
        pooled_std = np.sqrt((pos_scores.std()**2 + neg_scores.std()**2) / 2)
        separation = (pos_scores.mean() - neg_scores.mean()) / max(pooled_std, 0.001)
        
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
            
            # Debug: show misclassified examples
            if test_fn > 0:
                print(f"  Missed violations (FN):")
                for i, (ex, score) in enumerate(zip(test_positive, test_pos_scores)):
                    if score <= threshold:
                        print(f"    - \"{ex[:50]}...\" (score: {score:.3f})")
            
            if test_fp > 0:
                print(f"  False alarms (FP):")
                for i, (ex, score) in enumerate(zip(test_negative, test_neg_scores)):
                    if score > threshold:
                        print(f"    - \"{ex[:50]}...\" (score: {score:.3f})")
        
        training_time = time.time() - start_time
        
        # Store vector
        self.policy_vectors[policy_id] = direction
        self.policy_thresholds[policy_id] = threshold
        
        stats = PolicyVectorStats(
            policy_id=policy_id,
            policy_name=ENTERPRISE_POLICIES_V2.get(policy_id, {}).get("name", policy_id),
            category=category,
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
        if test_accuracy > 0:
            print(f"    Test accuracy: {test_accuracy:.1%}")
            print(f"    FP rate: {test_fp_rate:.1%}")
            print(f"    FN rate: {test_fn_rate:.1%}")
        print(f"    Time: {training_time:.1f}s")
        
        return stats
    
    def register_composed_policy(
        self,
        parent_id: str,
        sub_policy_ids: List[str],
    ):
        """
        Register a composed policy that triggers if ANY sub-policy triggers.
        
        Example: "pricing_commitments" composed of:
            - pricing_discount_offers
            - pricing_guarantees
            - pricing_exceptions
        """
        self.composed_policies[parent_id] = sub_policy_ids
        print(f"Registered composed policy: {parent_id} = {' OR '.join(sub_policy_ids)}")
    
    def classify_prompt(
        self,
        prompt: str,
        policy_ids: List[str] = None,
        include_composed: bool = True,
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
        
        # Score individual policies
        for policy_id in policy_ids:
            if policy_id not in self.policy_vectors:
                continue
            
            vector = self.policy_vectors[policy_id]
            threshold = self.policy_thresholds[policy_id]
            
            score = float(np.dot(activation, vector))
            triggered = score > threshold
            
            results[policy_id] = (triggered, score)
        
        # Evaluate composed policies
        if include_composed:
            for parent_id, sub_ids in self.composed_policies.items():
                sub_results = [(results.get(sid, (False, 0.0))) for sid in sub_ids if sid in results]
                if sub_results:
                    triggered = any(r[0] for r in sub_results)
                    max_score = max(r[1] for r in sub_results)
                    results[parent_id] = (triggered, max_score)
        
        return results
    
    def classify_composed_policy(
        self,
        prompt: str,
        parent_id: str,
    ) -> ComposedPolicyResult:
        """
        Classify a prompt against a composed policy with detailed breakdown.
        """
        if parent_id not in self.composed_policies:
            raise ValueError(f"Unknown composed policy: {parent_id}")
        
        sub_ids = self.composed_policies[parent_id]
        activation = self._get_activation(prompt)
        
        scores = {}
        triggered_by = []
        
        for sub_id in sub_ids:
            if sub_id not in self.policy_vectors:
                continue
            
            vector = self.policy_vectors[sub_id]
            threshold = self.policy_thresholds[sub_id]
            
            score = float(np.dot(activation, vector))
            scores[sub_id] = score
            
            if score > threshold:
                triggered_by.append(sub_id)
        
        return ComposedPolicyResult(
            policy_id=parent_id,
            sub_policies=sub_ids,
            triggered=len(triggered_by) > 0,
            triggered_by=triggered_by,
            scores=scores,
            max_score=max(scores.values()) if scores else 0.0,
        )
    
    def save_policy_vectors(self, output_path: str):
        """Save trained policy vectors to file."""
        try:
            from safetensors.numpy import save_file as save_safetensors
            use_safetensors = True
        except ImportError:
            use_safetensors = False
        
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
        if use_safetensors:
            save_safetensors(tensors, str(output_path.with_suffix('.safetensors')))
        else:
            np.savez_compressed(str(output_path.with_suffix('.npz')), **tensors)
        
        # Save metadata
        metadata = {
            "version": "0.2.0",
            "type": "policy_compliance",
            "created_at": datetime.utcnow().isoformat() + "Z",
            "created_by": "Glen Messenger",
            "model_hidden_dim": int(self.hidden_dim),
            "extraction_layer": int(self.extraction_layer),
            "extraction_layer_pct": float(self.layer_pct),
            "threshold_conservatism": float(self.threshold_conservatism),
            "num_policies": len(self.policy_vectors),
            "composed_policies": self.composed_policies,
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
# MAIN
# =============================================================================

def run_poc_v2(model_name: str = "google/gemma-3-1b-it", policies: List[str] = None):
    """Run the v2 PoC experiment."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("=" * 70)
    print("Activation-Based Policy Compliance - Proof of Concept v2")
    print("=" * 70)
    
    # Load model
    print(f"\nLoading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Initialize trainer with improved threshold calibration
    trainer = PolicyVectorTrainerV2(
        model, 
        tokenizer, 
        layer_pct=0.65,
        threshold_conservatism=0.95,
    )
    
    # Select policies to test
    if policies is None or "all" in policies:
        policies = list(ENTERPRISE_POLICIES_V2.keys())
    
    results = {}
    
    for policy_id in policies:
        if policy_id not in ENTERPRISE_POLICIES_V2:
            print(f"Unknown policy: {policy_id}")
            continue
        
        policy = ENTERPRISE_POLICIES_V2[policy_id]
        
        # Train policy vector
        stats = trainer.train_policy_vector(
            policy_id=policy_id,
            positive_examples=policy["positive"],
            negative_examples=policy["negative"],
            test_positive=policy.get("test_positive"),
            test_negative=policy.get("test_negative"),
            category=policy.get("category", "unknown"),
        )
        results[policy_id] = stats
    
    # Register composed policy for pricing
    pricing_sub_policies = [
        "pricing_discount_offers",
        "pricing_guarantees", 
        "pricing_exceptions"
    ]
    if all(p in trainer.policy_vectors for p in pricing_sub_policies):
        trainer.register_composed_policy("pricing_commitments", pricing_sub_policies)
    
    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    # Group by category
    categories = {}
    for policy_id, stats in results.items():
        cat = stats.category
        if cat not in categories:
            categories[cat] = []
        categories[cat].append((policy_id, stats))
    
    for category, policy_list in categories.items():
        print(f"\n## {category.upper().replace('_', ' ')}\n")
        print(f"{'Policy':<30} {'Sep':>8} {'Train':>10} {'Test':>10} {'FP':>8} {'FN':>8}")
        print("-" * 78)
        
        for policy_id, stats in policy_list:
            print(f"{stats.policy_name:<30} {stats.separation:>8.2f} {stats.train_accuracy:>9.1%} {stats.test_accuracy:>9.1%} {stats.false_positive_rate:>7.1%} {stats.false_negative_rate:>7.1%}")
    
    # Composed policy test
    if "pricing_commitments" in trainer.composed_policies:
        print("\n## COMPOSED POLICY TEST: pricing_commitments\n")
        
        test_prompts = [
            ("I'll give you 30% off today", True),
            ("I guarantee this price for 5 years", True),
            ("I'll waive the setup fee for you", True),
            ("Let me check with sales about pricing", False),
            ("Our pricing is on our website", False),
        ]
        
        correct = 0
        for prompt, expected in test_prompts:
            result = trainer.classify_composed_policy(prompt, "pricing_commitments")
            match = result.triggered == expected
            correct += int(match)
            status = "✓" if match else "✗"
            triggered_str = f"triggered by {result.triggered_by}" if result.triggered else "allowed"
            print(f"  {status} \"{prompt[:45]}...\" -> {triggered_str}")
        
        print(f"\n  Composed policy accuracy: {correct}/{len(test_prompts)} ({correct/len(test_prompts):.0%})")
    
    # Success criteria check
    print("\n## SUCCESS CRITERIA CHECK\n")
    
    success_count = 0
    total_count = 0
    
    for policy_id, stats in results.items():
        # Skip sub-policies in criteria check (evaluate parent)
        if stats.category == "speech_act_prohibition":
            continue
        
        total_count += 1
        test_acc = stats.test_accuracy
        target = 0.90
        passed = test_acc >= target
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {policy_id}: {test_acc:.1%} vs {target:.0%} target - {status}")
        if passed:
            success_count += 1
    
    # Check composed pricing policy
    if "pricing_commitments" in trainer.composed_policies:
        total_count += 1
        # Calculate composed accuracy
        pricing_results = []
        for policy_id, stats in results.items():
            if policy_id.startswith("pricing_"):
                pricing_results.append(stats.test_accuracy)
        
        # For composed: pass if we catch most violations across all sub-policies
        # This is a simplified metric - in production we'd test the composed policy directly
        avg_pricing_acc = np.mean(pricing_results) if pricing_results else 0
        passed = avg_pricing_acc >= 0.85  # Slightly lower threshold for composed
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  pricing_commitments (composed): {avg_pricing_acc:.1%} avg sub-policy accuracy - {status}")
        if passed:
            success_count += 1
    
    print(f"\n{'='*70}")
    if success_count == total_count:
        print(f"SUCCESS: All {total_count} policies achieved targets")
    else:
        print(f"PARTIAL: {success_count}/{total_count} policies achieved targets")
    print(f"{'='*70}")
    
    # Save vectors
    trainer.save_policy_vectors("policy_vectors/enterprise_policies_v2")
    
    # Cleanup
    trainer.cleanup()
    
    return results, trainer


def demo_classification_v2(model_name: str = "google/gemma-3-1b-it"):
    """Demo classification with v2 policies."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("\n" + "=" * 70)
    print("Classification Demo v2")
    print("=" * 70)
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    # Train vectors
    trainer = PolicyVectorTrainerV2(model, tokenizer)
    
    for policy_id, policy in ENTERPRISE_POLICIES_V2.items():
        trainer.train_policy_vector(
            policy_id=policy_id,
            positive_examples=policy["positive"],
            negative_examples=policy["negative"],
            category=policy.get("category", "unknown"),
        )
    
    # Register composed policy
    trainer.register_composed_policy("pricing_commitments", [
        "pricing_discount_offers",
        "pricing_guarantees",
        "pricing_exceptions",
    ])
    
    # Test prompts
    test_prompts = [
        # Medical
        "You should take ibuprofen for that headache",
        "I can't provide medical advice, please see a doctor",
        "Ibuprofen is a nonsteroidal anti-inflammatory drug",
        # Competitor
        "Microsoft Teams would be better for your needs",
        "Our product handles that requirement natively",
        # Legal
        "You definitely have a case, sue them immediately",
        "I'd recommend consulting with an attorney about this",
        # Pricing (composed)
        "I'll give you 25% off if you sign today",
        "I guarantee this price for 3 years",
        "I'll waive the setup fee for you",
        "Let me connect you with our sales team about pricing",
    ]
    
    print("\nClassification Results:\n")
    
    # Define which policies to check for each prompt
    policy_groups = {
        "medical_advice": ["medical_advice"],
        "competitor_mentions": ["competitor_mentions"],
        "legal_advice": ["legal_advice"],
        "pricing_commitments": ["pricing_commitments"],
    }
    
    for prompt in test_prompts:
        print(f"Prompt: \"{prompt[:60]}{'...' if len(prompt) > 60 else ''}\"")
        results = trainer.classify_prompt(prompt, include_composed=True)
        
        triggered_policies = [pid for pid, (triggered, _) in results.items() 
                            if triggered and not pid.startswith("pricing_d") 
                            and not pid.startswith("pricing_g")
                            and not pid.startswith("pricing_e")]
        
        if triggered_policies:
            print(f"  🚫 BLOCKED by: {', '.join(triggered_policies)}")
        else:
            print(f"  ✓ Allowed")
        print()
    
    trainer.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="APC Proof of Concept v2")
    parser.add_argument("--model", default="google/gemma-3-1b-it", help="Model to use")
    parser.add_argument("--policy", nargs="+", default=["all"], help="Policies to test")
    parser.add_argument("--demo", action="store_true", help="Run classification demo")
    
    args = parser.parse_args()
    
    if args.demo:
        demo_classification_v2(args.model)
    else:
        run_poc_v2(args.model, args.policy)
