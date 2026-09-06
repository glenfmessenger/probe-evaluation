# AF vs Eval-Based Detection Comparison Report
**Model:** google/gemma-3-1b-it
**Test Date:** 2025-12-30 00:47:21
**Prompts Tested:** 19

## Performance Comparison
| Method | Accuracy | Harmful Detection | False Positives | Avg Latency | P95 Latency |
|--------|----------|-------------------|-----------------|-------------|-------------|
| af | 94.7% | 100.0% | 12.5% | 70.4ms | 91.1ms |
| llm_judge | 89.5% | 90.9% | 12.5% | 486.2ms | 508.0ms |
| rule_based | 73.7% | 54.5% | 0.0% | 0.1ms | 0.1ms |

## Speed Analysis
- **llm_judge**: 6.9x slower than AF
- **rule_based**: 0.0x slower than AF

## Performance by Attack Type
### af
- **direct**: 100.0% (5 samples)
- **jailbreak**: 100.0% (3 samples)
- **encoded**: 100.0% (3 samples)
- **normal**: 100.0% (5 samples)
- **educational**: 100.0% (2 samples)
- **creative**: 0.0% (1 samples)

### llm_judge
- **direct**: 100.0% (5 samples)
- **jailbreak**: 100.0% (3 samples)
- **encoded**: 66.7% (3 samples)
- **normal**: 100.0% (5 samples)
- **educational**: 100.0% (2 samples)
- **creative**: 0.0% (1 samples)

### rule_based
- **direct**: 60.0% (5 samples)
- **jailbreak**: 100.0% (3 samples)
- **encoded**: 0.0% (3 samples)
- **normal**: 100.0% (5 samples)
- **educational**: 100.0% (2 samples)
- **creative**: 100.0% (1 samples)
