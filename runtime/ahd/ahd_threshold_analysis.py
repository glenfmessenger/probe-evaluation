"""
AHD Threshold Analysis
======================

Analyze per-response confidence scores and find thresholds that achieve
acceptable hallucination rates.

Question: What threshold do we need to get hallucination rate below 50%? 20%? 10%?
"""

import json
import numpy as np

# Load results
import sys
filepath = sys.argv[1] if len(sys.argv) > 1 else 'simpleqa_gemma4b_v3_results.json'
with open(filepath, 'r') as f:
    data = json.load(f)

samples = data['samples']

print("=" * 80)
print("PER-RESPONSE CONFIDENCE SCORES")
print("=" * 80)
print()

# Separate correct and incorrect
correct = [s for s in samples if s['is_correct'] == True]
incorrect = [s for s in samples if s['is_correct'] == False]

print(f"Total: {len(samples)} | Correct: {len(correct)} | Incorrect: {len(incorrect)}")
print()

# Show some examples
print("CORRECT ANSWERS:")
print("-" * 80)
for s in correct:
    conf = s['ahd_confidence']
    icon = "🟢" if conf > 0.5 else "🟡" if conf > 0.3 else "🔴"
    print(f"{icon} Conf: {conf:.3f} | Q: {s['question'][:50]}...")
    print(f"   A: {s['model_answer'][:60]}...")
    print()

print("\nINCORRECT ANSWERS (sorted by confidence):")
print("-" * 80)
incorrect_sorted = sorted(incorrect, key=lambda x: x['ahd_confidence'], reverse=True)

print("\n🔴 HIGH CONFIDENCE BUT WRONG (most dangerous):")
for s in incorrect_sorted[:5]:
    print(f"   Conf: {s['ahd_confidence']:.3f} | Q: {s['question'][:45]}...")
    print(f"   Wrong A: {s['model_answer'][:50]}...")
    print(f"   Gold A: {s['gold_answer'][:50]}...")
    print()

print("\n🟢 LOW CONFIDENCE AND WRONG (AHD working correctly):")
for s in incorrect_sorted[-5:]:
    print(f"   Conf: {s['ahd_confidence']:.3f} | Q: {s['question'][:45]}...")
    print(f"   Wrong A: {s['model_answer'][:50]}...")
    print()

# Threshold analysis
print("\n" + "=" * 80)
print("THRESHOLD ANALYSIS: What threshold achieves acceptable hallucination rates?")
print("=" * 80)
print()

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

print(f"{'Threshold':>10} | {'Accepted':>8} | {'Correct':>7} | {'Wrong':>7} | {'Accuracy':>8} | {'Halluc%':>8} | {'Coverage':>8}")
print("-" * 85)

for thresh in thresholds:
    accepted = [s for s in samples if s['ahd_confidence'] >= thresh]
    if len(accepted) == 0:
        print(f"{thresh:>10.1f} | {'0':>8} | {'-':>7} | {'-':>7} | {'-':>8} | {'-':>8} | {'0%':>8}")
        continue
    
    acc_correct = sum(1 for s in accepted if s['is_correct'])
    acc_wrong = sum(1 for s in accepted if not s['is_correct'])
    accuracy = acc_correct / len(accepted) * 100
    halluc_rate = acc_wrong / len(accepted) * 100
    coverage = len(accepted) / len(samples) * 100
    
    # Highlight good thresholds
    marker = ""
    if halluc_rate < 50:
        marker = " ✓"
    if halluc_rate < 20:
        marker = " ✓✓"
    if halluc_rate < 10:
        marker = " ✓✓✓"
    
    print(f"{thresh:>10.1f} | {len(accepted):>8} | {acc_correct:>7} | {acc_wrong:>7} | {accuracy:>7.1f}% | {halluc_rate:>7.1f}% | {coverage:>7.1f}%{marker}")

print()
print("Key: ✓ = <50% halluc | ✓✓ = <20% halluc | ✓✓✓ = <10% halluc")

# Find optimal threshold for different targets
print("\n" + "=" * 80)
print("OPTIMAL THRESHOLDS FOR TARGET HALLUCINATION RATES")
print("=" * 80)
print()

targets = [50, 30, 20, 10, 5]
all_confidences = sorted([s['ahd_confidence'] for s in samples], reverse=True)

for target in targets:
    # Binary search for threshold
    best_thresh = None
    best_coverage = 0
    
    for thresh in np.arange(0.0, 1.0, 0.01):
        accepted = [s for s in samples if s['ahd_confidence'] >= thresh]
        if len(accepted) == 0:
            continue
        acc_wrong = sum(1 for s in accepted if not s['is_correct'])
        halluc_rate = acc_wrong / len(accepted) * 100
        coverage = len(accepted) / len(samples) * 100
        
        if halluc_rate <= target and coverage > best_coverage:
            best_thresh = thresh
            best_coverage = coverage
    
    if best_thresh is not None:
        accepted = [s for s in samples if s['ahd_confidence'] >= best_thresh]
        acc_correct = sum(1 for s in accepted if s['is_correct'])
        acc_wrong = len(accepted) - acc_correct
        print(f"Target ≤{target}% hallucination: threshold={best_thresh:.2f}, coverage={best_coverage:.1f}%, "
              f"accepted={len(accepted)} ({acc_correct} correct, {acc_wrong} wrong)")
    else:
        print(f"Target ≤{target}% hallucination: NOT ACHIEVABLE with this data")

# Show what abstention would look like
print("\n" + "=" * 80)
print("ABSTENTION MODE EXAMPLE")
print("=" * 80)
print()
print("If we set threshold=0.6 and abstain on low confidence:")
print()

thresh = 0.6
for s in samples[:10]:
    conf = s['ahd_confidence']
    if conf >= thresh:
        status = "✅ ANSWER" if s['is_correct'] else "❌ WRONG"
        print(f"{status} (conf={conf:.2f}): {s['model_answer'][:60]}...")
    else:
        print(f"🤷 ABSTAIN (conf={conf:.2f}): \"I'm not confident about this answer.\"")
    print(f"   Q: {s['question'][:60]}...")
    print()
