# Taste (Continuously Learned by [CommandCode][cmd])

[cmd]: https://commandcode.ai/

# Communication
- Use Turkish for responses in this project. Confidence: 0.85
- Prioritize minimal, concise code (minimum LOC) over verbose implementations. Confidence: 0.70
- Provide deep, thorough engineer-level analysis before proposing solutions. Confidence: 0.75

# ML-Workflow
- Track calibration metrics (ECE, Brier) alongside AUROC/accuracy — the user prioritizes reliable probability estimates. Confidence: 0.80
- Use YAML config flags (e.g., use_ordinal_mobility: true) instead of hardcoding model behavior changes. Confidence: 0.70
- When an experimental change degrades performance, immediately revert to the last working checkpoint — do not iterate on broken approaches. The user values stable progress over adventurous exploration. Confidence: 0.75
- The user wants results-first communication: run experiments silently and report only the final outcome, not intermediate steps. Confidence: 0.65

